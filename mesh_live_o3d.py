"""Live human-mesh demo, MODEL-SWAPPABLE.

This is a thin wrapper around live_pipeline.py: it reuses the ENTIRE pipeline
(YOLO person mask -> mesh -> depth metric grounding -> camera-visible-surface
refine -> head-hold -> decoupled cloud@~15Hz / mesh open3d viz) and only
swaps out the mesh MODEL. All downstream maths is SMPL-topology (6890 verts, 24
joints, faces, J_regressor), so any SMPL-outputting model plugs in by providing
just two functions with these signatures:
    load(device)                       -> (model_holder, tf, faces)
    infer(model_holder, tf, rgb, bbox_xyxy, device)
                                       -> (verts[6890,3] metric root-rel,
                                           joints[24,3], pelvis_px[2], sigma|None)
live_pipeline.main() calls those by the names `load_model` / `run_model`, so we
simply bind them on the module before invoking it.

Engines:
    --engine tokenhmr  TokenHMR (CVPR 2024, tokenised SMPL pose)   [default]
    --engine fastsam3d Fast SAM 3D Body (MHR, TensorRT backbone worker)

    DISPLAY=:0 ~/anaconda3/envs/human-pose/bin/python mesh_live_o3d.py --engine tokenhmr
"""
# PyTorch sizes its intra-op thread pool to the core count (12 here), but this
# pipeline runs its model on the GPU: those threads only serve small CPU ops
# like the SMPL skinning, where 12-way parallelism costs more in synchronisation
# than it saves.  Worse, they saturate the CPU and starve the depth fit running
# in the same process -- measured on an Orin, `fit` went 41.2 -> 8.3 ms and the
# mesh rate 8.9 -> 14.9 Hz purely from capping this, with the GPU inference
# itself unchanged (33.3 vs 33.1 ms).  Set before torch is imported.
#
# One thread, not a few: the pool earns nothing at any size (frame_total 70.1 ms
# with four threads, 69.5 with one) while the three extra threads each sat at
# ~85% -- 2.5 cores of waste.  OMP_WAIT_POLICY=PASSIVE does not change that, so
# they are doing real work, just work worth nothing to a GPU-resident model.
# Those cores are better left to whatever else the robot is running: TokenHMR
# went from 5.9 to 1.3 cores with the frame time unchanged.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys, types, argparse, time
from pathlib import Path
import numpy as np
import torch

# Belt and braces: OMP_NUM_THREADS only takes effect if it is set before the
# OpenMP runtime initialises, which a different entry point might not honour.
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

import live_pipeline as PIPE    # full pipeline: viz, ROS, threading, SMPL maths


def _sync_current_cuda_stream():
    """Wait only for this inference stream, not unrelated GPU workers."""
    torch.cuda.current_stream().synchronize()


# TokenHMR emits no per-joint uncertainty, so the shared depth refine gets a
# flat SMPL-24 reliability prior (smaller sigma => larger vertex weight).
SIGMA_DEFAULT = 0.020

# TokenHMR re-estimates the body SHAPE (betas) from scratch every frame, but a
# person's build does not actually change -- so any frame-to-frame movement in
# betas is estimation noise, and it shows up as the mesh visibly swelling and
# shrinking, worst while someone turns and the silhouette evidence for shape is
# weakest.  Smoothing betas alone fixes the scale wobble without touching pose,
# which must stay per-frame to keep the demo responsive.  The running mean
# converges over the first BETAS_WARM frames, then holds with an EMA.
BETAS_SMOOTHING = True
BETAS_ALPHA = 0.05           # weight of the newest frame once warmed up
BETAS_WARM = 20              # frames of plain averaging before the EMA takes over
BETAS_STATE = {"mean": None, "n": 0}


def reset_betas_state():
    BETAS_STATE["mean"] = None
    BETAS_STATE["n"] = 0


def smooth_betas(betas):
    """Running mean -> EMA on the shape coefficients."""
    if not BETAS_SMOOTHING:
        return betas
    st = BETAS_STATE
    if st["mean"] is None:
        st["mean"] = betas.clone()
        st["n"] = 1
    elif st["n"] < BETAS_WARM:
        st["n"] += 1
        st["mean"] += (betas - st["mean"]) / st["n"]
    else:
        st["mean"] += BETAS_ALPHA * (betas - st["mean"])
    return st["mean"]

HERE = Path(__file__).resolve().parent
TOKENHMR_ROOT = HERE / "third_party" / "TokenHMR"
# Escape hatch for benchmarking the PyTorch path on a machine that has an engine.
TOKENHMR_ALLOW_TRT = os.environ.get("TOKENHMR_NO_TRT", "0") != "1"


def _install_smpl_tables(smpl):
    """Expose an engine's SMPL topology to the shared RGB-D postprocessor."""
    faces = np.asarray(smpl.faces).astype(np.int32)
    lbs = np.asarray(smpl.lbs_weights.detach().cpu())
    PIPE.VERT_JOINT = lbs.argmax(1)
    PIPE.HEAD_CAP = PIPE.VERT_JOINT == 15
    PIPE.PART_MASKS = {
        name: np.isin(PIPE.VERT_JOINT, ids)
        for name, ids in PIPE.PART_JOINTS.items()
    }
    PIPE.PART_MASKS["head"] = lbs[:, 15] > 0.05
    PIPE.J_REGRESSOR = np.asarray(smpl.J_regressor.detach().cpu())[:24]
    vt = np.asarray(smpl.v_template.detach().cpu())
    Jr = PIPE.J_REGRESSOR @ vt
    neck_r, R_r = PIPE._torso_frame_smpl(Jr)
    PIPE.HEAD_LOCAL_REST = (vt[PIPE.HEAD_CAP] - neck_r) @ R_r
    PIPE.PART_LOCAL_REST = {
        name: (vt[PIPE.PART_MASKS[name]] - neck_r) @ R_r
        for name in PIPE.PART_JOINTS
    }
    return faces


# --- TokenHMR engine ---------------------------------------------------------
def load_tokenhmr_engine(device):
    sys.path.insert(0, str(TOKENHMR_ROOT))
    # Import the local runner before temporarily chdir'ing into TokenHMR;
    # Python's empty sys.path entry follows cwd and would otherwise stop
    # resolving this sibling module.
    from tokenhmr_trt_runtime import TokenHMRTensorRT, ENGINE_PATH
    old_cwd = os.getcwd()
    try:
        os.chdir(TOKENHMR_ROOT)
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        from tokenhmr.lib.configs import get_config
        from tokenhmr.lib.models.smpl_wrapper import SMPL

        # Two ways to run the regressor:
        #
        #   TensorRT (preferred).  The engine owns every learned weight, so the
        #   2.6 GB training checkpoint is never touched -- only the non-learned
        #   SMPL layer is rebuilt from the config/body-model files.  This
        #   changes no inference weights or post-processing mathematics.
        #
        #   PyTorch fallback.  Used when no engine is present.  An engine is
        #   tied to one GPU architecture AND one TensorRT version, so a machine
        #   that cannot build one (e.g. JetPack 6.0 ships TensorRT 8.6, which
        #   rejects this graph) would otherwise be unable to run at all.  Here
        #   the full checkpoint IS loaded and run_model takes its non-TRT
        #   branch.  Same maths, slower.
        use_trt = ENGINE_PATH.exists() and TOKENHMR_ALLOW_TRT
        if use_trt:
            cfg = get_config("data/checkpoints/model_config.yaml")
            cfg.defrost()
            if (cfg.MODEL.BACKBONE.TYPE == "vit"
                    and "BBOX_SHAPE" not in cfg.MODEL):
                cfg.MODEL.BBOX_SHAPE = [192, 256]
            cfg.freeze()
            smpl_cfg = {k.lower(): v for k, v in dict(cfg.SMPL).items()}
            smpl = SMPL(**smpl_cfg).to(device).eval()
            model = types.SimpleNamespace(smpl=smpl)
            trt_runner = TokenHMRTensorRT(ENGINE_PATH)
        else:
            why = ("disabled by TOKENHMR_NO_TRT" if ENGINE_PATH.exists()
                   else f"no engine at {ENGINE_PATH}")
            print(f"TokenHMR: running the regressor in PyTorch ({why}); "
                  f"build an engine with export_tokenhmr_trt.py for full speed",
                  flush=True)
            os.chdir(old_cwd)          # load_full_model does its own chdir
            from export_tokenhmr_trt import load_full_model
            model, cfg = load_full_model(device)
            os.chdir(TOKENHMR_ROOT)
            trt_runner = None
    finally:
        os.chdir(old_cwd)
    faces = _install_smpl_tables(model.smpl)
    if trt_runner is not None:
        print(f"loaded TokenHMR TensorRT FP16 regressor: {ENGINE_PATH.name}",
              flush=True)
        print(f"loaded TokenHMR minimal runtime (checkpoint skipped); "
              f"input {cfg.MODEL.IMAGE_SIZE}px, mesh faces {faces.shape}",
              flush=True)
    else:
        print(f"loaded TokenHMR PyTorch regressor from checkpoint; "
              f"input {cfg.MODEL.IMAGE_SIZE}px, mesh faces {faces.shape}",
              flush=True)
    return {"model": model, "cfg": cfg, "trt": trt_runner,
            "focal": float(cfg.EXTRA.FOCAL_LENGTH)}, None, faces


def run_tokenhmr(eng, tf, rgb, bbox_xyxy, device, is_bgr=False):
    from tokenhmr.lib.datasets.vitdet_dataset import ViTDetDataset
    t0 = time.perf_counter()
    # Dataset follows OpenCV convention; the shared pipeline supplies RGB.
    # is_bgr lets a caller that already holds the original OpenCV frame hand it
    # over untouched.  A ROS node has BGR to begin with, so converting it to RGB
    # only for this function to convert it straight back cost two full-frame
    # copies -- 16 ms per fit at 1920x1536 -- for a value never otherwise used.
    # ViTDetDataset can now consume the pipeline's native RGB frame directly.
    # The old RGB->BGR full-frame copy followed by BGR->RGB on the 256px crop
    # was mathematically redundant and cost ~10-18 ms at 1920x1536.
    source = np.ascontiguousarray(rgb)
    boxes = np.asarray(bbox_xyxy, np.float32).reshape(1, 4)
    item = ViTDetDataset(eng["cfg"], source, boxes,
                         input_is_rgb=not is_bgr)[0]
    t0b = time.perf_counter()          # crop/normalise vs host->device split
    batch = {
        k: (torch.as_tensor(v)[None].to(device) if not isinstance(v, int)
            else torch.tensor([v], device=device))
        for k, v in item.items()
    }
    t1 = time.perf_counter()
    if device.type == "cuda": _sync_current_cuda_stream()
    t2 = time.perf_counter()
    with torch.inference_mode():
        if eng["trt"] is not None:
            global_orient, body_pose, betas, pred_cam = eng["trt"](batch["img"])
            smpl_out = eng["model"].smpl(
                global_orient=global_orient.float(),
                body_pose=body_pose.float(),
                betas=smooth_betas(betas.float()),
                pose2rot=False)
            verts_t = smpl_out.vertices
            joints44_t = smpl_out.joints
            # Weak-perspective (s, tx, ty) over the 256 px crop -> a translation
            # in the crop's own camera.  cfg.EXTRA.FOCAL_LENGTH is the focal
            # length the network was trained with, expressed for that crop.
            cam_t = torch.stack(
                [pred_cam[:, 1], pred_cam[:, 2],
                 2.0 * eng["focal"] / (eng["cfg"].MODEL.IMAGE_SIZE *
                                        pred_cam[:, 0] + 1e-9)], dim=-1)
            pelvis_cam = joints44_t[:, 39] + cam_t
            root2d_t = ((eng["focal"] / eng["cfg"].MODEL.IMAGE_SIZE) *
                        pelvis_cam[:, :2] / pelvis_cam[:, 2:])
        else:
            with torch.amp.autocast(device_type=device.type,
                                    enabled=device.type == "cuda"):
                out = eng["model"](batch)
            verts_t = out["pred_vertices"]
            root2d_t = out["pred_keypoints_2d"][:, 39]
    if device.type == "cuda": _sync_current_cuda_stream()
    t3 = time.perf_counter()
    verts = verts_t[0].float().cpu().numpy()
    joints = (PIPE.J_REGRESSOR @ verts).astype(np.float32)
    # TokenHMR keypoints are crop-normalised. ViTDetDataset uses a square crop,
    # so crop-normalised root -> original pixel is center + root*box_size.
    root2d = root2d_t[0].float().cpu().numpy()
    pelvis_px = (np.asarray(item["box_center"], np.float32) +
                 root2d * float(item["box_size"]))
    sigma = np.full(24, SIGMA_DEFAULT, np.float32)
    # Hand the pipeline everything it needs to turn this weak-perspective guess
    # into a real camera-frame translation (see PIPE.weak_persp_root): the
    # translation itself, the crop it belongs to, and the focal length it
    # assumes.  Fast SAM gets an equivalent quantity for free -- its worker runs
    # with the real intrinsics and returns pred_cam_t in the camera frame --
    # which is why its placement is markedly better.
    if eng["trt"] is not None:
        PIPE.MODEL_CAM_T = {
            "cam_t": cam_t[0].float().cpu().numpy(),
            "box_center": np.asarray(item["box_center"], np.float32),
            "box_size": float(item["box_size"]),
            "crop_px": float(eng["cfg"].MODEL.IMAGE_SIZE),
            "crop_focal": float(eng["focal"]),
        }
    else:
        PIPE.MODEL_CAM_T = None
    t4 = time.perf_counter()
    PIPE.MODEL_PROFILE = {"model_pre": (t1-t0)*1e3, "model_pre_crop": (t0b-t0)*1e3,
                        "model_pre_h2d": (t1-t0b)*1e3, "model_gpu": (t3-t2)*1e3,
                        "model_post": (t4-t3)*1e3}
    return verts.astype(np.float32), joints, pelvis_px.astype(np.float32), sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine",
                    choices=["tokenhmr", "fastsam3d", "wham"],
                    default="tokenhmr")
    # The display cloud shares the GIL, the CPU and the GPU upload path with the
    # latency-critical mesh thread.  Exposing both knobs makes the trade-off
    # measurable in one command instead of an edit-and-rerun.
    ap.add_argument("--cloud-hz", type=float, default=None,
                    help="display point-cloud rate (default 15; lower frees the mesh thread)")
    ap.add_argument("--run-seconds", type=float, default=0.0,
                    help="close the window automatically after N seconds")
    ap.add_argument("--seg-view", action="store_true",
                    help="side window showing the person mask driving the fit")
    ap.add_argument("--seg-view-width", type=int, default=480)
    ap.add_argument("--seg-pick", action="store_true",
                    help="show every detected person in the segmentation window "
                         "and fit the one you click (implies --seg-view)")
    ap.add_argument("--person-cloud-stride", type=int, default=None,
                    help="sampling stride for the emphasised person surface; "
                         "finer than the room costs upload bandwidth")
    ap.add_argument("--display-stride", type=int, default=None,
                    help="display point-cloud pixel stride (default 6; higher is sparser)")
    args, _ = ap.parse_known_args()
    if args.engine == "fastsam3d":
        # Fast SAM 3D uses MHR topology (18,439 vertices), a Python-3.11 worker
        # and calibrated camera-space output, so it cannot be substituted into
        # the synchronous SMPL runner hook below.  Keep one public launcher and
        # dispatch to its purpose-built RGB-D pipeline, forwarding every other
        # CLI option unchanged.
        forwarded = [sys.argv[0]]
        skip = False
        for arg in sys.argv[1:]:
            if skip:
                skip = False
                continue
            if arg == "--engine":
                skip = True
                continue
            if arg.startswith("--engine="):
                continue
            forwarded.append(arg)
        sys.argv = forwarded
        import fastsam3d_live_o3d as fast_live
        print("=== ENGINE: Fast SAM 3D TRT raw realtime (MHR + RGB-D) ===",
              flush=True)
        return fast_live.main()
    if args.run_seconds:
        PIPE.RUN_SECONDS = args.run_seconds
    if args.seg_view or args.seg_pick:
        PIPE.SEG_VIEW = True
        PIPE.SEG_VIEW_WIDTH = args.seg_view_width
    if args.seg_pick:
        PIPE.SEG_PICK = True
    if args.cloud_hz is not None:
        PIPE.CLOUD_HZ = args.cloud_hz
    if args.display_stride is not None:
        PIPE.DISPLAY_STRIDE = args.display_stride
    if args.person_cloud_stride is not None:
        PIPE.PERSON_CLOUD_STRIDE = args.person_cloud_stride
    if args.engine == "tokenhmr":
        PIPE.DISPLAY_MODEL_NAME = "TokenHMR"
        PIPE.load_model = load_tokenhmr_engine    # rebind the two model hooks main() calls
        PIPE.run_model = run_tokenhmr
        # Raw low-latency mode: never replace a current body part with cached
        # geometry, and never average it with an older pose. Keep only metric
        # grounding + one depth z-fit so the result remains in RGB-D space.
        PIPE.UPPER_BODY_4DOF_REFINE = False
        PIPE.TEMPORAL_SMOOTHING = False
        PIPE.HOLD_BODY_PARTS = False
        PIPE.FAST_VISIBILITY = True
        PIPE.UPPER_REFINE_STATE["valid"] = False
        PIPE.reset_shape_state = reset_betas_state
        print("=== ENGINE: TokenHMR TRT raw realtime (no hold / no EMA / fast visibility) ===",
              flush=True)
    else:
        ap.error("WHAM is a sequence model and cannot use the single-frame live hook. "
                 "Use third_party/WHAM/demo.py for video inference; a live WHAM mode "
                 "requires a temporal frame/keypoint/feature buffer.")
    PIPE.main()


if __name__ == "__main__":
    main()
