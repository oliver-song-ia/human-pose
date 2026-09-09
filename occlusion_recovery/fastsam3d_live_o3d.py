#!/usr/bin/env python3
"""Fast SAM 3D Body + Gemini RGB-D camera-visible-surface live demo.

Fast SAM runs in its Python 3.11 conda environment in a ZMQ worker.  This
process stays in ``human-pose`` (Python 3.10, ROS 2/Open3D), selects one person
with the project's YOLO segmentation model, and applies the same HPR-front-
surface, depth-only registration used by the HMR2/TokenHMR demo.

Run (after starting the Gemini ROS driver):
  DISPLAY=:0 conda run -n human-pose python fastsam3d_live_o3d.py
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import zmq

import live_pipeline as PIPE
import mesh_live_o3d as ML

FAST_ROOT = Path(os.environ.get(
    "FAST_SAM_3D_ROOT",
    "/media/oliver/9a72b131-ff4a-4fc1-a6d5-53ef9c8524e1/code/Fast-SAM-3D-Body"))
WORKER = FAST_ROOT / "ros_realtime/mesh_worker.py"
sys.path.insert(0, str(FAST_ROOT))  # shared length-prefixed ZMQ wire helper
YOLO_ENGINE = Path(os.environ.get(
    "HUMAN_POSE_YOLO",
    "/home/oliver/Documents/semantic_perception/yolo26m-seg-custom_20260903_rtx4070.engine"))
YOLO_PT = Path(os.environ.get(
    "HUMAN_POSE_YOLO_PT",
    "/home/oliver/Documents/semantic_perception/yolo26m-seg-custom_20260903.pt"))
MESH_VIS = np.array([0.10, 0.85, 0.55])
MESH_HID = np.array([0.20, 0.24, 0.42])
TOKEN_VIS = np.array([1.00, 0.35, 0.05])
TOKEN_HID = np.array([0.55, 0.12, 0.65])
FLIP = np.array([1.0, -1.0, -1.0])
# MHR70 body landmark indices. Pelvis is appended as index 70 from the two hips.
MHR_BODY_IDS = np.array([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                         15, 18, 41, 62, 69, 70], np.int32)
MHR_BODY_BONES = np.array([
    [70, 9], [9, 11], [11, 13], [13, 15],
    [70, 10], [10, 12], [12, 14], [14, 18],
    [70, 69], [69, 5], [5, 7], [7, 62],
    [69, 6], [6, 8], [8, 41], [69, 0],
], np.int32)


def start_worker(frames_endpoint, results_endpoint, layer_dtype, image_size,
                 use_fast_trt=True, verbose_inference=False, extra_env=()):
    trt_engine = FAST_ROOT / "checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine"
    trt_env = (f"export USE_TRT_BACKBONE=1 TRT_BACKBONE_PATH={trt_engine!s}"
               if use_fast_trt else "export USE_TRT_BACKBONE=0")
    # The estimator already times backbone / decoder / pre / post per frame and
    # prints them; --quiet-inference is what swallows that breakdown.  Dropping
    # the flag is the whole instrument -- it costs a few prints per frame.
    quiet = "" if verbose_inference else "--quiet-inference"
    # The estimator is driven almost entirely by environment variables
    # (USE_COMPILE, MHR_USE_CUDA_GRAPH, BODY_INTERM_PRED_LAYERS, ...), so let the
    # caller append or override any of them without editing this launcher.
    overrides = "".join(f"export {kv}\n" for kv in extra_env)
    # The worker needs an interpreter that can import sam_3d_body.  By default
    # that is the project's conda env; FAST_SAM_3D_PYTHON names an interpreter
    # directly instead, which is how a machine without conda (a Jetson, say)
    # runs the worker out of a plain venv.
    worker_python = os.environ.get("FAST_SAM_3D_PYTHON", "")
    if worker_python:
        env_setup = ""
    else:
        # conda's nvidia pip wheels ship their CUDA libs outside the linker path
        env_setup = (
            "source /home/oliver/anaconda3/etc/profile.d/conda.sh\n"
            "conda activate fast_sam_3d_body\n"
            "NVLIBS=$(find \"$CONDA_PREFIX/lib/python3.11/site-packages/nvidia\" "
            "-mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:' 2>/dev/null)\n"
            "export LD_LIBRARY_PATH=\"${NVLIBS}${LD_LIBRARY_PATH:-}\"\n")
        worker_python = "python"
    command = f"""
{env_setup}export GPU_HAND_PREP=1 LAYER_DTYPE={layer_dtype} SKIP_KEYPOINT_PROMPT=1 IMG_SIZE={image_size}
# CUDA graphs measured at -2.4 ms on the body decoder (24.15 -> 21.70) with an
# unchanged numerical path; revert with --worker-env MHR_USE_CUDA_GRAPH=0.
# USE_COMPILE stays 0: the estimator's multi-person compile warmup feeds the
# backbone a batch of 3 while the TRT engine is fixed at (1,3,384,384).
export USE_COMPILE=0 MHR_USE_CUDA_GRAPH=1 KEYPOINT_PROMPT_INTERM_INTERVAL=999
export BODY_INTERM_PRED_LAYERS=0,1,2 MHR_NO_CORRECTIVES=1
{trt_env}
{overrides}cd {FAST_ROOT!s}
exec {worker_python} {WORKER!s} --external-boxes --no-overlay {quiet} --reliable --frames-endpoint {frames_endpoint} --results-endpoint {results_endpoint}
"""
    log = open("/tmp/fastsam3d_rgbd_worker.log", "w", buffering=1)
    return subprocess.Popen(["bash", "-lc", command], stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True), log


def decode(msg):
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    if msg.encoding in ("rgb8", "bgr8"):
        im = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return np.ascontiguousarray(im if msg.encoding == "rgb8" else im[..., ::-1])
    if msg.encoding == "16UC1":
        return rows[:, :msg.width * 2].copy().view(np.uint16).reshape(msg.height, msg.width)
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def build_mesh(verts, faces, visible, visible_color=MESH_VIS,
               hidden_color=MESH_HID):
    """Two uniformly coloured submeshes: camera-visible faces, and the rest.

    A face is visible only when all of its vertices pass the image-space
    z-buffer *and* its outward normal faces the camera.

    The colour boundary has to be HARD -- Open3D interpolates per-vertex
    colours, which turned boundary triangles into green/grey speckle.  This
    used to be solved by duplicating all three corners of every triangle so
    each corner could carry its own face colour, but on MHR that inflates
    18,439 vertices into 110,622 and cost 6.5 ms per frame.  Splitting the
    faces into two meshes that each carry ONE uniform colour gives the same
    hard boundary from the original shared vertices (measured 6.5 -> ~1 ms).
    The visible trade-off is that shared vertices average their face normals,
    so the body is smooth-shaded now instead of faceted.
    """
    verts = np.asarray(verts, np.float32)
    faces = np.asarray(faces, np.int32)
    visible_mask = np.zeros(len(verts), dtype=bool)
    visible_mask[np.asarray(visible, dtype=np.int64)] = True
    # `all vertices pass the z-buffer` AND `normal faces the camera`.  The first
    # test is a gather-and-reduce; the second needs a cross product per face.
    # Evaluating the cheap test first and the geometric one only on the faces
    # that survive it gives an identical result (False & x == False) over ~10%
    # as many faces -- the full-mesh version was 3.5 ms of the 8.6 ms build.
    candidate = visible_mask[faces].all(axis=1)
    face_visible = np.zeros(len(faces), dtype=bool)
    if candidate.any():
        cf = faces[candidate]
        v0, v1, v2 = verts[cf[:, 0]], verts[cf[:, 1]], verts[cf[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        # 3x the centroid: a positive scale cannot change the sign of the dot
        face_visible[candidate] = np.einsum(
            "ij,ij->i", normals, v0 + v1 + v2) < 0.0

    draw_verts = o3d.utility.Vector3dVector(
        np.ascontiguousarray(verts * FLIP, np.float64))
    out = []
    for keep, colour in ((face_visible, visible_color),
                         (~face_visible, hidden_color)):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = draw_verts
        mesh.triangles = o3d.utility.Vector3iVector(
            np.ascontiguousarray(faces[keep], np.int32))
        mesh.paint_uniform_color(colour)
        mesh.compute_vertex_normals()
        out.append(mesh)
    return tuple(out)


def build_skeleton(joints):
    """MHR70 body skeleton in the same display frame as the mesh."""
    joints = np.asarray(joints, np.float32)
    pelvis = 0.5 * (joints[9] + joints[10])
    points = np.concatenate([joints[:70], pelvis[None]], axis=0) * FLIP
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, np.float64))
    lines.lines = o3d.utility.Vector2iVector(MHR_BODY_BONES)
    lines.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.72, 0.08], (len(MHR_BODY_BONES), 1)))
    dots = o3d.geometry.PointCloud()
    dots.points = o3d.utility.Vector3dVector(
        np.ascontiguousarray(points[MHR_BODY_IDS], np.float64))
    dots.colors = o3d.utility.Vector3dVector(
        np.tile([1.0, 0.18, 0.05], (len(MHR_BODY_IDS), 1)))
    return lines, dots


def build_yolo_debug(rgb, det, misses):
    """BGR debug image showing exactly the box/mask driving Fast SAM."""
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if det is None:
        cv2.putText(out, f"NO PERSON  misses={misses}", (24, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3,
                    cv2.LINE_AA)
        return out
    box, mask = det
    m = np.asarray(mask, dtype=bool)
    if m.shape == out.shape[:2]:
        tint = np.zeros_like(out); tint[..., 1] = 255
        out[m] = (0.55 * out[m] + 0.45 * tint[m]).astype(np.uint8)
    x1, y1, x2, y2 = np.rint(box).astype(int)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(out, "PERSON MASK", (max(0, x1), max(28, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                cv2.LINE_AA)
    return out


def worker_input_roi(rgb, box, K, pad):
    """Crop worker input while preserving the original optical camera frame.

    ``pad`` is a fraction of the box's longest side on every edge.  Fast SAM's
    body transform expands the detection to a padded square, so using the
    longest side here prevents that internal crop from running outside the
    transmitted image for ordinary in-frame detections.
    """
    h, w = rgb.shape[:2]
    box = np.asarray(box, dtype=np.float32).reshape(4)
    side = max(float(box[2] - box[0]), float(box[3] - box[1]), 1.0)
    margin = max(0.0, float(pad)) * side
    x0 = max(0, int(np.floor(box[0] - margin)))
    y0 = max(0, int(np.floor(box[1] - margin)))
    x1 = min(w, int(np.ceil(box[2] + margin)))
    y1 = min(h, int(np.ceil(box[3] + margin)))
    if x1 <= x0 or y1 <= y0:
        return rgb, box, np.asarray(K, dtype=np.float32)
    roi = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    roi_box = box - np.array([x0, y0, x0, y0], dtype=np.float32)
    roi_K = np.array(K, dtype=np.float32, copy=True)
    roi_K[0, 2] -= x0
    roi_K[1, 2] -= y0
    return roi, roi_box, roi_K


def fast_visible_vertices(verts, K, image_shape, cell=4, thickness=0.035):
    """Stable O(N) camera-front visibility for the dense 18k-vertex MHR mesh.

    Open3D hidden-point-removal constructs a spherical hull; on MHR its runtime
    varies sharply with pose and was the source of multi-second latency spikes.
    This image-space z-buffer keeps vertices on the nearest surface in each
    small pixel cell. ``thickness`` retains neighbouring vertices from the same
    physical surface instead of producing a sparse single-vertex shell.
    """
    h, w = image_shape
    z = verts[:, 2]
    valid = np.isfinite(verts).all(1) & (z > 1e-5)
    ids = np.flatnonzero(valid)
    if not len(ids):
        return ids
    vv = verts[ids]
    u = np.rint(K[0, 0] * vv[:, 0] / vv[:, 2] + K[0, 2]).astype(np.int32)
    v = np.rint(K[1, 1] * vv[:, 1] / vv[:, 2] + K[1, 2]).astype(np.int32)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ids, u, v, zz = ids[inside], u[inside], v[inside], vv[inside, 2]
    if not len(ids):
        return ids
    gw = (w + cell - 1) // cell
    gh = (h + cell - 1) // cell
    bins = (v // cell) * gw + (u // cell)
    nearest = np.full(gw * gh, np.inf, np.float32)
    np.minimum.at(nearest, bins, zz)
    return ids[zz <= nearest[bins] + thickness]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-endpoint", default="tcp://127.0.0.1:5699")
    ap.add_argument("--results-endpoint", default="tcp://127.0.0.1:5700")
    ap.add_argument("--no-worker", action="store_true")
    ap.add_argument("--mesh-hz", type=float, default=10.0)
    ap.add_argument("--mesh-ema-alpha", type=float, default=1.0,
                    help="current-frame mesh weight; 1.0 disables temporal smoothing")
    ap.add_argument("--cloud-hz", type=float, default=15.0)
    ap.add_argument("--fast-layer-dtype", choices=("fp16", "bf16"), default="fp16",
                    help="Fast SAM backbone autocast dtype")
    ap.add_argument("--fast-image-size", type=int, choices=(384, 448, 512), default=384,
                    help="Fast SAM input size; use 448 if fine-detail quality regresses")
    ap.add_argument("--disable-fast-trt", action="store_true",
                    help="disable the fixed 384px TensorRT backbone and use PyTorch")
    ap.add_argument("--max-mesh-age-ms", type=float, default=500.0,
                    help="drop cold/stalled results older than this")
    ap.add_argument("--detection-miss-limit", type=int, default=8,
                    help="consecutive YOLO misses before hiding the last mesh")
    ap.add_argument("--yolo-window", action="store_true",
                    help="show the optional YOLO mask debug window")
    ap.add_argument("--no-yolo-window", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--with-tokenhmr", action="store_true",
                    help="overlay TokenHMR (orange) with Fast SAM 3D (green)")
    ap.add_argument("--render-hz", type=float, default=20.0,
                    help="scene update rate; the renderer's steady GPU load "
                         "competes with the worker's bursty inference")
    ap.add_argument("--no-gui", action="store_true",
                    help="run the pipeline without the Open3D window; the "
                         "renderer holds the GPU continuously, which starves "
                         "the worker's bursty submissions")
    ap.add_argument("--seg-hz", type=float, default=8.0,
                    help="detection rate for the picker panel; lower leaves more "
                         "GPU for the mesh model")
    ap.add_argument("--seg-pick", action="store_true",
                    help="show every detected person in the mask window and fit "
                         "the one you click (implies --yolo-window)")
    ap.add_argument("--hide-person-cloud", action="store_true",
                    help="drop the measured person points instead of emphasising "
                         "them (they can be mistaken for the predicted back)")
    ap.add_argument("--person-cloud-stride", type=int, default=2,
                    help="sampling stride for the person surface; finer than the "
                         "room so the measured body reads clearly (0 = same)")
    ap.add_argument("--run-seconds", type=float, default=0.0,
                    help="close the window automatically after N seconds so a "
                         "latency run is reproducible without hand timing")
    ap.add_argument("--worker-env", action="append", default=[], metavar="KEY=VAL",
                    help="extra environment variable for the Fast SAM worker; "
                         "repeatable, applied after the built-in exports "
                         "(e.g. --worker-env USE_COMPILE=1)")
    ap.add_argument("--jpeg-worker-input", action="store_true",
                    help="send the worker a JPEG instead of the raw ROI "
                         "(needed only for a worker on another machine)")
    ap.add_argument("--no-pipeline-detect", action="store_true",
                    help="send the worker this frame's box instead of the previous "
                         "frame's, serialising YOLO ahead of the model again")
    ap.add_argument("--worker-verbose", action="store_true",
                    help="let the Fast SAM worker print its per-stage breakdown "
                         "(backbone / decoder / pre / post) into its log")
    ap.add_argument("--worker-roi-pad", type=float, default=0.25,
                    help="worker JPEG ROI padding as a fraction of the person's longest side")
    ap.add_argument("--full-frame-worker-input", action="store_true",
                    help="send the full RGB frame to Fast SAM (compatibility/debug fallback)")
    args = ap.parse_args()
    if args.seg_pick:
        args.yolo_window = True          # the picker needs its window
    if not 0.0 < args.mesh_ema_alpha <= 1.0:
        ap.error("--mesh-ema-alpha must be in (0, 1]")
    if args.fast_image_size != 384 and not args.disable_fast_trt:
        ap.error("the TensorRT backbone is fixed at 384px; use --disable-fast-trt "
                 "with --fast-image-size 448/512")

    worker = log = None
    if not args.no_worker:
        worker, log = start_worker(args.frames_endpoint, args.results_endpoint,
                                   args.fast_layer_dtype, args.fast_image_size,
                                   not args.disable_fast_trt, args.worker_verbose,
                                   args.worker_env)
        print(f"Fast SAM worker pid={worker.pid}; log=/tmp/fastsam3d_rgbd_worker.log", flush=True)
        print(f"Fast SAM config: {args.fast_image_size}px / {args.fast_layer_dtype} / "
              "allocator cache retained", flush=True)
        print("Fast SAM display: raw current frame / no hold / "
              f"EMA alpha={args.mesh_ema_alpha:.2f} / strict face visibility",
              flush=True)

    ctx = zmq.Context.instance()
    frames = ctx.socket(zmq.PUSH); frames.setsockopt(zmq.SNDHWM, 1)
    frames.bind(args.frames_endpoint)
    results = ctx.socket(zmq.PULL); results.setsockopt(zmq.RCVHWM, 1)
    results.connect(args.results_endpoint)

    from ultralytics import YOLO
    yolo_path = YOLO_ENGINE if YOLO_ENGINE.exists() else YOLO_PT
    yolo = YOLO(str(yolo_path))
    person_cls = next((i for i, n in yolo.names.items() if str(n).lower() == "person"), 0)
    direct_yolo = None
    if PIPE.YOLO_DIRECT and str(yolo_path).endswith(".engine"):
        from yolo_trt_runtime import YoloSegTRT
        direct_yolo = YoloSegTRT(str(yolo_path), conf=0.25, person_class=person_cls)
        print("YOLO: direct TensorRT runner (ultralytics wrapper bypassed)", flush=True)

    import rclpy
    from sensor_msgs.msg import CameraInfo, Image
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    rclpy.init(); node = rclpy.create_node("fastsam3d_rgbd_live")
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST)
    state = {"rgb": None, "depth": None, "K": None, "stamp": 0, "running": True,
             "mesh": None, "mesh_stamp": None,
             "token_mesh": None, "token_mesh_stamp": None, "cloud": None,
             "skeleton": None,
             "profile": None, "error": None, "person_detected": False,
             "yolo_viz": None, "seg_det": None, "seg_stamp": 0,
             "cloud_exclude_mask": None, "person_mask": None,
             "sensor_lag": None,
             "profile_rows": []}      # unified latency rows (see PIPE.UNIFIED_KEYS)

    def on_rgb(m):
        # `stamp` is arrival in THIS node -- the shared measurement origin.  The
        # header stamp additionally exposes camera driver + transport, which no
        # other bucket on either demo can see.
        hs = m.header.stamp
        now_ns = time.time_ns()
        state.update(rgb=m, stamp=now_ns,
                     sensor_lag=now_ns * 1e-6 - (hs.sec * 1e3 + hs.nanosec * 1e-6))
    node.create_subscription(Image, "/camera/color/image_raw", on_rgb, qos)
    node.create_subscription(Image, "/camera/depth/image_raw",
                             lambda m: state.update(depth=m), qos)
    node.create_subscription(CameraInfo, "/camera/color/camera_info",
                             lambda m: state.update(K=np.array(m.k, np.float32).reshape(3, 3)), qos)
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    def cloud_loop():
        """Depth display producer, deliberately independent of mesh inference.

        It snapshots only the newest ROS messages and never waits for YOLO,
        JPEG/ZMQ, or Fast SAM.  Intermediate camera frames are dropped when the
        renderer cannot keep up, keeping latency bounded instead of queueing.
        """
        period = 1.0 / max(args.cloud_hz, 1.0)
        last_stamp = 0
        seg_yolo = None
        # The panel exists so a human can see and click people; 8 Hz is plenty
        # for that, and every extra detection contends with Fast SAM for the
        # GPU (measured: 15 Hz panel pushed the mesh from 335 to 480 ms).
        seg_period = 1.0 / max(args.seg_hz, 0.5)
        last_seg = 0.0
        if args.seg_pick and direct_yolo is not None:
            # The picker panel must not run at the mesh rate: Fast SAM takes
            # ~300 ms per frame, so a panel driven by the inference loop updates
            # 3x a second and feels frozen while you are trying to click a
            # moving person.  Give it its own detector on the cloud thread,
            # which already runs at ~15 Hz on freshly decoded frames.
            from yolo_trt_runtime import YoloSegTRT
            seg_yolo = YoloSegTRT(str(yolo_path), conf=0.25,
                                  person_class=person_cls)
        while state["running"]:
            started = time.monotonic()
            rgb_msg, dep_msg, K, stamp = state["rgb"], state["depth"], state["K"], state["stamp"]
            if (rgb_msg is not None and dep_msg is not None and K is not None
                    and stamp != last_stamp):
                last_stamp = stamp
                rgb = decode(rgb_msg)
                now_seg = time.monotonic()
                if seg_yolo is not None and now_seg - last_seg >= seg_period:
                    last_seg = now_seg
                    t_seg = time.perf_counter()
                    seg_people = seg_yolo(rgb, all_people=True)
                    # Resolve the click here so the selection reacts at panel
                    # rate; the inference loop reads the resulting target.
                    seg_chosen = PIPE.pick_person(seg_people)
                    state["yolo_viz"] = PIPE.build_seg_view_multi(
                        rgb, seg_people, seg_chosen,
                        (time.perf_counter() - t_seg) * 1000)
                    # Publish it for the inference loop: running a second
                    # detector there would double the YOLO work and the two
                    # engines would contend for the GPU (measured: mesh
                    # 335 -> 480 ms).  One detection per frame, shared.
                    state["seg_det"] = (
                        (seg_people[seg_chosen][0], seg_people[seg_chosen][1])
                        if seg_chosen is not None else None)
                    state["seg_stamp"] = stamp
                    # Emphasise the person actually being fitted, updated at
                    # panel rate so the highlight follows a click immediately.
                    state["person_mask"] = (seg_people[seg_chosen][1]
                                            if seg_chosen is not None else None)
                depth = decode(dep_msg).astype(np.float32) * 0.001
                if rgb.shape[:2] == depth.shape:
                    # Same builder as the TokenHMR demo, so both look identical.
                    state["cloud"], _ = PIPE.build_scene_cloud(
                        depth, rgb, K,
                        person_mask=(None if args.hide_person_cloud
                                     else state["person_mask"]),
                        person_stride=args.person_cloud_stride)
            time.sleep(max(0.0, period - (time.monotonic() - started)))

    def inference_loop():
        last_stamp = 0; last_send = 0.0; ema = None; prev_box = None
        profile_rows = state["profile_rows"]   # shared list; main() prints it
        alpha = args.mesh_ema_alpha; result_count = 0
        detection_misses = 0
        token_eng = token_tf = token_faces = token_ema = None
        token_failed = False
        while state["running"]:
            if worker is not None and worker.poll() is not None:
                state["error"] = (
                    f"Fast SAM worker exited with code {worker.returncode}; "
                    "see /tmp/fastsam3d_rgbd_worker.log")
                print(f"ERROR: {state['error']}", flush=True)
                return
            rgb_msg, dep_msg, K, stamp = state["rgb"], state["depth"], state["K"], state["stamp"]
            if rgb_msg is None or dep_msg is None or K is None or stamp == last_stamp:
                time.sleep(0.005); continue
            last_stamp = stamp
            t_decode0 = time.perf_counter()
            rgb, depth = decode(rgb_msg), decode(dep_msg).astype(np.float32) * 0.001
            decode_ms = (time.perf_counter() - t_decode0) * 1000
            if rgb.shape[:2] != depth.shape: continue
            if time.monotonic() - last_send < 1.0 / args.mesh_hz: continue
            t0 = time.perf_counter()
            capture_wait_ms = (time.time_ns() - stamp) * 1e-6
            sensor_lag_ms = state["sensor_lag"]
            from ros_realtime.wire import pack, unpack

            def dispatch(send_box):
                """ROI crop -> JPEG -> ZMQ.  False if the frame is unusable."""
                if args.full_frame_worker_input:
                    w_rgb, w_box, w_K = rgb, send_box, K
                else:
                    w_rgb, w_box, w_K = worker_input_roi(rgb, send_box, K,
                                                         args.worker_roi_pad)
                raw_shape = None
                if args.jpeg_worker_input:
                    ok, enc = cv2.imencode(".jpg",
                                           cv2.cvtColor(w_rgb, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok:
                        return False
                    payload = enc.tobytes()
                else:
                    # The worker is on this machine: shipping the raw ROI skips a
                    # JPEG encode here and a decode there (~3.5 ms together) and
                    # hands the model an unquantised image.  ~0.7 MB over
                    # loopback is cheaper than either half of that round trip.
                    w_rgb = np.ascontiguousarray(w_rgb, np.uint8)
                    payload = w_rgb.tobytes()
                    raw_shape = list(w_rgb.shape)
                meta = json.dumps({"stamp_sec": int(stamp // 1_000_000_000),
                                   "stamp_nsec": int(stamp % 1_000_000_000),
                                   "frame_id": rgb_msg.header.frame_id,
                                   "camera_intrinsics": np.asarray(w_K).tolist(),
                                   "raw_shape": raw_shape,
                                   "bbox": np.asarray(w_box).tolist()}).encode()
                # PUB/SUB can still hold one completed response while this loop is
                # doing YOLO/fit/render work. Drop it before issuing a new request,
                # then accept only the response carrying this exact capture stamp.
                # Previously an old mesh was paired with the current depth/mask;
                # the suspicious 3 ms "wait" for a 100 ms worker exposed this bug.
                while results.poll(timeout=0):
                    results.recv()
                frames.send(pack([meta, payload]))
                return True

            # The worker only needs a BOX, so issuing the request on the PREVIOUS
            # frame's box lets Fast SAM's ~58 ms run concurrently with this
            # frame's ~15 ms detection instead of behind it.  One frame of motion
            # is exactly what the 25% ROI padding is there to absorb, and the
            # mask the depth fit uses is still this frame's -- only the crop
            # window is a frame old, never the mesh or the depth it lands on.
            piped = (not args.no_pipeline_detect) and prev_box is not None
            t_dispatch = None
            if piped:
                d0 = time.perf_counter()
                if not dispatch(prev_box):
                    continue
                t_dispatch = time.perf_counter(); dispatch_ms = (t_dispatch-d0)*1000
                last_send = time.monotonic()
            t_yolo0 = time.perf_counter()
            people = None
            if args.seg_pick:
                # Reuse the detection the panel thread already made for this
                # frame -- it runs at cloud rate on the same images, so the mesh
                # loop pays no YOLO cost at all here.
                people = []                       # marks "panel owns the view"
                det = state["seg_det"]
                yolo_speed = {}
                t_yolo = time.perf_counter()
            elif direct_yolo is not None:
                # Straight to TensorRT: RGB in, (box, mask) out.  Ultralytics'
                # wrapper costs 0.2 ms on an RTX 4070 but tens of ms on a Jetson.
                det = direct_yolo(rgb)
                yolo_speed = {}
                t_yolo = time.perf_counter()
            else:
                # predict, not track: `largest_person` ranks by box area and no
                # track id is read anywhere downstream, so ByteTrack was overhead.
                res = yolo.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   classes=[person_cls], conf=0.25, verbose=False)[0]
                yolo_speed = getattr(res, "speed", None) or {}
                t_yolo = time.perf_counter()
                det = PIPE.largest_person(res, rgb.shape[:2])
            if det is None:
                detection_misses += 1
                if people is None:
                    state["yolo_viz"] = build_yolo_debug(rgb, None, detection_misses)
                if detection_misses == args.detection_miss_limit:
                    state["person_detected"] = False
                    state["mesh"] = None
                    state["token_mesh"] = None
                    state["cloud_exclude_mask"] = None
                    state["person_mask"] = None
                    ema = token_ema = None
                    prev_box = None     # re-anchor the crop on the next detection
                    print(f"person detection lost for {detection_misses} consecutive "
                          "checks; hiding stale mesh", flush=True)
                continue
            if detection_misses >= args.detection_miss_limit:
                print(f"person detection reacquired after {detection_misses} misses",
                      flush=True)
            detection_misses = 0
            state["person_detected"] = True
            if people is None:
                state["yolo_viz"] = build_yolo_debug(rgb, det, 0)
            box, mask = det
            prev_box = box
            # Dilate slightly to cover segmentation erosion and inter-frame
            # motion before removing the observed front-surface person points.
            state["person_mask"] = np.asarray(mask, bool)
            state["cloud_exclude_mask"] = None if not args.hide_person_cloud else cv2.dilate(
                np.asarray(mask, np.uint8), np.ones((11, 11), np.uint8),
                iterations=1).astype(bool)
            t_detect = time.perf_counter()
            detect_ms = (t_detect - t_yolo0) * 1000
            if not piped:
                if not dispatch(box):
                    continue
                t_dispatch = time.perf_counter()
                dispatch_ms = (t_dispatch - t_detect) * 1000
                last_send = time.monotonic()
            # Fast SAM is now executing in the worker process. Run TokenHMR on
            # this exact RGB/box concurrently instead of serialising the two
            # model latencies. Both results are published only for this stamp.
            token_raw = None
            token_model_ms = 0.0
            if token_eng is not None:
                token_t0 = time.perf_counter()
                try:
                    token_raw = ML.run_tokenhmr(token_eng, token_tf, rgb, box,
                                                token_device)
                    token_model_ms = (time.perf_counter() - token_t0) * 1000
                except Exception as exc:
                    print(f"WARNING: TokenHMR inference failed: {exc}", flush=True)
            deadline = time.monotonic() + 2.0
            matched = None
            while state["running"] and time.monotonic() < deadline:
                remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
                if not results.poll(timeout=remaining_ms):
                    break
                candidate = unpack(results.recv())
                candidate_h = json.loads(candidate[0])
                returned_stamp = (int(candidate_h["stamp_sec"]) * 1_000_000_000
                                  + int(candidate_h["stamp_nsec"]))
                if returned_stamp == stamp:
                    matched = candidate, candidate_h
                    break
            if matched is None:
                print(f"WARNING: no matching Fast SAM result for request {stamp}",
                      flush=True)
                continue
            payload, h = matched
            header, faces_b, verts_b = payload[:3]
            joints_b = payload[3] if len(payload) >= 5 else None
            t_worker = time.perf_counter()
            result_age_ms = (time.time_ns() - stamp) * 1e-6
            if result_age_ms > args.max_mesh_age_ms:
                print(f"dropping stale Fast SAM result: age={result_age_ms:.1f} ms",
                      flush=True)
                ema = None
                continue
            if h["n_persons"] < 1: continue
            faces_np = np.frombuffer(faces_b, np.int32).reshape(-1, 3).copy()
            verts = np.frombuffer(verts_b, np.float32).reshape(-1, 3)[:h["verts_per_person"]].copy()
            joints = None
            if joints_b is not None and h.get("joints_per_person", 0) >= 70:
                joints = np.frombuffer(joints_b, np.float32).reshape(-1, 3)[
                    :h["joints_per_person"]].copy()
            vis = fast_visible_vertices(verts, K, depth.shape)
            f1 = time.perf_counter()
            cloud = PIPE.person_cloud(depth, mask, K)
            f2 = time.perf_counter()
            fit_joints = (joints if joints is not None else
                          np.zeros((1, 3), np.float32))
            verts, fit_joints = PIPE.refine_to_cloud(
                verts, fit_joints, cloud, visible_idx=vis)
            if joints is not None:
                joints = fit_joints
            t_fit = time.perf_counter()
            ema = verts if ema is None or ema.shape != verts.shape else alpha * verts + (1-alpha) * ema
            vis = fast_visible_vertices(ema, K, depth.shape)
            g1 = time.perf_counter()
            new_geometry = build_mesh(ema, faces_np, vis)
            g2 = time.perf_counter()
            state["mesh_stamp"] = stamp
            state["mesh"] = new_geometry
            state["skeleton"] = build_skeleton(joints) if joints is not None else None
            g3 = time.perf_counter()
            sub = {"fit_visible": (f1-t_worker)*1000, "fit_cloud": (f2-f1)*1000,
                   "fit_icp": (t_fit-f2)*1000, "geo_visible": (g1-t_fit)*1000,
                   "geo_mesh": (g2-g1)*1000, "geo_skeleton": (g3-g2)*1000,
                   "cloud_pts": float(len(cloud)), "vis_verts": float(len(vis))}

            if token_raw is not None:
                token_verts, token_joints, pelvis_px, _ = token_raw
                root = PIPE.metric_root(pelvis_px, depth, K, mask)
                if root is not None:
                    token_verts, token_joints = PIPE.ground(token_verts,
                                                          token_joints, root)
                    token_vis = fast_visible_vertices(token_verts, K, depth.shape)
                    token_verts, token_joints = PIPE.refine_to_cloud(
                        token_verts, token_joints, cloud, visible_idx=token_vis)
                    token_ema = (token_verts if token_ema is None else
                                 alpha * token_verts + (1-alpha) * token_ema)
                    token_vis = fast_visible_vertices(token_ema, K, depth.shape)
                    state["token_mesh"] = build_mesh(
                        token_ema, token_faces, token_vis, TOKEN_VIS, TOKEN_HID)
                    state["token_mesh_stamp"] = stamp
            t_geometry = time.perf_counter()
            end_to_end_ms = (time.time_ns() - stamp) * 1e-6
            state["profile"] = (h.get("infer_ms", 0.0),
                                (time.perf_counter()-t0)*1000, end_to_end_ms)
            result_count += 1
            profile_rows.append({
                **sub,
                "sensor_lag": sensor_lag_ms,
                "capture_wait": capture_wait_ms,
                "decode": decode_ms,
                "detect": detect_ms,
                "detect_gpu": yolo_speed.get("inference"),
                "dispatch": dispatch_ms,
                # model_core is the worker's self-reported inference; model_total
                # adds the ZMQ round trip; model_exposed is the part still on the
                # critical path once detection runs concurrently with it.
                "model_core": h.get("infer_ms", 0.0),
                "model_total": (t_worker - t_dispatch) * 1000,
                "model_exposed": (t_worker - t_detect) * 1000,
                "fit": (t_fit - t_worker) * 1000,
                "geometry": (t_geometry - t_fit) * 1000,
                "frame_total": (t_geometry - t0) * 1000,
                "end_to_end": end_to_end_ms,
            })
            if result_count % 10 == 0:
                print(f"mesh latency: worker={h.get('infer_ms', 0.0):.1f} ms  "
                      f"pipeline={(time.perf_counter()-t0)*1000:.1f} ms  "
                      f"frame-age={end_to_end_ms:.1f} ms  "
                      f"stages[yolo={(t_yolo-t0)*1000:.1f}, "
                      f"wait={(t_worker-t_yolo)*1000:.1f}, "
                      f"fit={(t_fit-t_worker)*1000:.1f}, "
                      f"geometry={(t_geometry-t_fit)*1000:.1f}, "
                      f"token={token_model_ms:.1f}]", flush=True)

            # Delay the one-time TokenHMR construction until Fast SAM has
            # answered once, so both large CUDA runtimes are not initialised at
            # the same instant. The already-published green mesh stays visible
            # during this warm-up; following frames contain both meshes.
            if args.with_tokenhmr and token_eng is None and not token_failed:
                try:
                    import torch
                    token_device = torch.device("cuda")
                    print("Loading TokenHMR comparison model...", flush=True)
                    token_eng, token_tf, token_faces = ML.load_tokenhmr_engine(
                        token_device)
                    print("Mesh colors: Fast SAM 3D=green, TokenHMR=orange",
                          flush=True)
                except Exception as exc:
                    token_failed = True
                    print(f"ERROR: TokenHMR disabled: {exc}", flush=True)

    cloud_thread = threading.Thread(target=cloud_loop, daemon=True)
    inference_thread = threading.Thread(target=inference_loop, daemon=True)
    cloud_thread.start(); inference_thread.start()

    if args.no_gui:
        # Diagnostic / headless: everything runs except the viewer.  Filament
        # renders continuously at vsync, and on a Jetson that steady GPU load
        # competes with the worker, which submits one burst per inference.
        print("running without the Open3D window", flush=True)
        end = time.monotonic() + (args.run_seconds or 1e9)
        try:
            while state["running"] and time.monotonic() < end:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        state["running"] = False
        cloud_thread.join(timeout=1.0)
        inference_thread.join(timeout=2.5)
        if state["profile_rows"]:
            PIPE.unified_summary(state["profile_rows"], "Fast SAM 3D")
        if worker is not None:
            worker.terminate()
            try: worker.wait(timeout=5)
            except subprocess.TimeoutExpired: worker.kill()
        frames.close(linger=0); results.close(linger=0)
        node.destroy_node(); rclpy.try_shutdown()
        if log is not None: log.close()
        return

    # The legacy Visualizer ignores mesh alpha.  Use the same GUI renderer and
    # transparency material as the TokenHMR live demo so the two engines have
    # directly comparable presentation.
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = gui.Application.instance
    app.initialize()
    title = ("Fast SAM 3D (green) + TokenHMR (orange) + RGB-D"
             if args.with_tokenhmr else
             "Fast SAM 3D Body + RGB-D front-surface fit")
    win = app.create_window(title, 1280, 860)
    sw = gui.SceneWidget()
    sw.scene = rendering.Open3DScene(win.renderer)
    sw.scene.set_background([1, 1, 1, 1])
    win.add_child(sw)

    m_pts = rendering.MaterialRecord()
    m_pts.shader = "defaultUnlit"
    m_pts.point_size = PIPE.CLOUD_POINT_SIZE
    m_mesh = rendering.MaterialRecord()
    m_mesh.shader = "defaultLitTransparency"
    m_mesh.base_color = [1, 1, 1, 0.7]
    m_bone = rendering.MaterialRecord()
    m_bone.shader = "unlitLine"
    m_bone.line_width = 4.0
    m_joint = rendering.MaterialRecord()
    m_joint.shader = "defaultUnlit"
    m_joint.point_size = 8.0

    gui_state = {"camera_set": False, "render_count": 0,
                 "update_pending": False}

    def replace_geometry(name, geometry, material):
        scene = sw.scene
        if scene.has_geometry(name):
            scene.remove_geometry(name)
        if geometry is not None:
            scene.add_geometry(name, geometry, material)

    def replace_split_mesh(name, pair, material):
        """build_mesh returns (visible_faces, hidden_faces); swap both halves."""
        for suffix, geometry in zip(("_vis", "_hid"),
                                    pair if pair is not None else (None, None)):
            replace_geometry(name + suffix, geometry, material)

    def update_scene():
        """Upload only the newest producer result on the Open3D GUI thread."""
        try:
            if args.yolo_window and not args.no_yolo_window and state["yolo_viz"] is not None:
                title = "YOLO person mask -> Fast SAM"
                cv2.imshow(title, state["yolo_viz"])
                if args.seg_pick and not gui_state.get("seg_cb"):
                    cv2.setMouseCallback(title, PIPE._on_seg_click)
                    gui_state["seg_cb"] = True
                cv2.waitKey(1)
            if state["cloud"] is not None:
                cloud = state["cloud"]
                state["cloud"] = None
                replace_geometry("cloud", cloud, m_pts)
                if not gui_state["camera_set"] and cloud.has_points():
                    bb = sw.scene.bounding_box
                    sw.setup_camera(60, bb, bb.get_center())
                    gui_state["camera_set"] = True
            if state["mesh"] is not None:
                render_t0 = time.perf_counter()
                mesh = state["mesh"]
                state["mesh"] = None
                replace_split_mesh("mesh", mesh, m_mesh)
                gui_state["render_count"] += 1
                render_age_ms = ((time.time_ns() - state["mesh_stamp"]) * 1e-6
                                 if state["mesh_stamp"] is not None else 0.0)
                if gui_state["render_count"] % 10 == 0 or render_age_ms > 1000:
                    print(f"render latency: update={(time.perf_counter()-render_t0)*1000:.1f} ms  "
                          f"frame-age={render_age_ms:.1f} ms", flush=True)
            if state["token_mesh"] is not None:
                token_mesh = state["token_mesh"]
                state["token_mesh"] = None
                replace_split_mesh("token_mesh", token_mesh, m_mesh)
            if state["skeleton"] is not None:
                bones, joint_dots = state["skeleton"]
                state["skeleton"] = None
                replace_geometry("bones", bones, m_bone)
                replace_geometry("joint_dots", joint_dots, m_joint)
            elif not state["person_detected"]:
                for name in ("mesh_vis", "mesh_hid", "token_mesh_vis",
                             "token_mesh_hid", "bones", "joint_dots"):
                    if sw.scene.has_geometry(name):
                        sw.scene.remove_geometry(name)
        finally:
            gui_state["update_pending"] = False

    def render_loop():
        """Post scene updates at --render-hz, and only when there is new content.

        This used to post every 10 ms regardless.  Nothing produces geometry
        that fast -- the mesh arrives a few times a second and the cloud at
        cloud_hz -- so most of those posts re-uploaded unchanged geometry, and
        the resulting steady GPU load starved the Fast SAM worker, whose own
        submissions come in one burst per inference: measured on an Orin, the
        worker's inference read 264 ms with the viewer running and 157 ms
        without it.
        """
        period = 1.0 / max(args.render_hz, 1.0)
        while state["running"]:
            started = time.monotonic()
            if not gui_state["update_pending"] and (
                    state["cloud"] is not None or state["mesh"] is not None
                    or state["token_mesh"] is not None
                    or state["skeleton"] is not None
                    or not state["person_detected"]
                    or (args.yolo_window and state["yolo_viz"] is not None)):
                gui_state["update_pending"] = True
                app.post_to_main_thread(win, update_scene)
            time.sleep(max(0.005, period - (time.monotonic() - started)))

    render_thread = threading.Thread(target=render_loop, daemon=True)
    render_thread.start()
    if args.run_seconds > 0:
        def _autostop():
            time.sleep(args.run_seconds)
            if state["running"]:
                print(f"--run-seconds {args.run_seconds:.0f} elapsed; closing",
                      flush=True)
                app.post_to_main_thread(win, win.close)
        threading.Thread(target=_autostop, daemon=True).start()
    try:
        app.run()
    finally:
        state["running"] = False
        if args.yolo_window and not args.no_yolo_window:
            cv2.destroyAllWindows()
        # Let producer threads leave their current short operation before CUDA,
        # ZMQ, Open3D and ROS objects are destroyed. Exiting with a live C++
        # worker thread caused "terminate called without an active exception".
        cloud_thread.join(timeout=1.0)
        inference_thread.join(timeout=2.5)
        render_thread.join(timeout=1.0)
        if state["profile_rows"]:
            PIPE.unified_summary(state["profile_rows"], "Fast SAM 3D")
        if worker is not None:
            worker.terminate()
            try: worker.wait(timeout=5)
            except subprocess.TimeoutExpired: worker.kill()
        frames.close(linger=0); results.close(linger=0)
        node.destroy_node(); rclpy.try_shutdown()
        if log is not None: log.close()


if __name__ == "__main__":
    main()
