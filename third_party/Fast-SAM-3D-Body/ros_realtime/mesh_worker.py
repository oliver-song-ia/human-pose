#!/usr/bin/env python
"""Fast SAM 3D Body inference worker.

Runs in the `fast_sam_3d_body` conda env. Receives JPEG frames over ZMQ from
ros_mesh_bridge.py, runs mesh recovery, and publishes decimated meshes back.

Frames in : SUB  tcp://127.0.0.1:5599   [meta_json, jpeg]
Meshes out: PUB  tcp://127.0.0.1:5600   [meta_json, faces_i32, verts_f32, overlay_jpeg]
"""
import argparse, contextlib, json, os, sys, time
import numpy as np
import zmq
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
from ros_realtime.wire import pack, unpack


def build_estimator(args):
    from notebook.utils import setup_sam_3d_body
    return setup_sam_3d_body(
        detector_name=None if args.external_boxes else "yolo_pose",
        detector_model=args.detector_model,
        # RGB-D live callers provide calibrated K.  The legacy autonomous RViz
        # path can use the estimator's built-in default focal length instead of
        # keeping a second large MoGe model resident on an 8 GB GPU.
        fov_name=None,
        local_checkpoint_path=args.checkpoint,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-endpoint", default="tcp://127.0.0.1:5599")
    ap.add_argument("--results-endpoint", default="tcp://127.0.0.1:5600")
    ap.add_argument("--checkpoint", default="./checkpoints/sam-3d-body-dinov3")
    ap.add_argument("--detector-model", default="./checkpoints/yolo/yolo11m-pose.pt")
    ap.add_argument("--target-faces", type=int, default=0,
                    help="decimate the mesh to about this many faces (0 = full resolution)")
    ap.add_argument("--overlay", action="store_true", default=True,
                    help="draw 2D skeleton overlay and ship it alongside the mesh")
    ap.add_argument("--no-overlay", action="store_true",
                    help="do not render/transport the unused 2-D JPEG overlay")
    ap.add_argument("--quiet-inference", action="store_true",
                    help="suppress verbose per-layer model timing prints")
    ap.add_argument("--reliable", action="store_true",
                    help="use PUSH/PULL for a reliable single-request live client")
    ap.add_argument("--external-boxes", action="store_true",
                    help="caller always supplies bbox; do not load a duplicate detector")
    args = ap.parse_args()
    if args.no_overlay:
        args.overlay = False

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.PULL if args.reliable else zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 1)
    if not args.reliable:
        sub.setsockopt(zmq.CONFLATE, 1)      # legacy multi-client RViz mode
        sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(args.frames_endpoint)
    pub = ctx.socket(zmq.PUSH if args.reliable else zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 2)
    pub.bind(args.results_endpoint)

    estimator = build_estimator(args)
    # The RGB-D live path never requests an overlay.  Importing the visualizer
    # pulls in matplotlib and constructing it does work that is otherwise
    # completely unused.  Keep it lazy for the legacy RViz path.
    skel = None
    if args.overlay:
        from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
        from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
        skel = SkeletonVisualizer(line_width=2, radius=4)
        skel.set_pose_meta(mhr70_pose_info)

    faces_full = np.asarray(estimator.faces, np.int32)
    collapse = None     # built once; the mesh topology is identical on every frame
    faces_out = faces_full
    faces_bytes = faces_out.tobytes()

    print(f"[worker] ready. faces={faces_full.shape[0]}  waiting for frames ...", flush=True)
    n_done, t_win, fps = 0, time.perf_counter(), 0.0
    profile_sum = np.zeros(5, dtype=np.float64)
    profile_n = 0
    cached_K_key, cached_cam_int = None, None
    quiet_sink = open(os.devnull, "w") if args.quiet_inference else None

    while True:
        try:
            meta_raw, payload = unpack(sub.recv())
        except KeyboardInterrupt:
            break
        frame_t0 = time.perf_counter()
        meta = json.loads(meta_raw)
        # A sender on this machine can ship the ROI as raw contiguous RGB and
        # skip the JPEG round trip entirely: the extra bytes cost far less over
        # loopback than encode+decode (~3.5 ms combined), and the model sees the
        # unquantised image.  `raw_shape` in the meta selects it; without it the
        # original JPEG path is used unchanged.
        raw_shape = meta.get("raw_shape")
        if raw_shape:
            # frombuffer is read-only; copy so downstream crops can write.
            img_rgb = np.frombuffer(payload, np.uint8).reshape(raw_shape).copy()
            decode_done = preprocess_done = time.perf_counter()
        else:
            img = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            decode_done = time.perf_counter()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # estimator expects RGB
            preprocess_done = time.perf_counter()
        t0 = preprocess_done
        try:
            # A caller that already runs a person segmenter (the RGB-D Open3D
            # demo does) can ship its selected box.  Reusing it avoids a second
            # YOLO pass and guarantees that the mesh and depth mask describe the
            # same person.  Keep the old autonomous detector path for RViz.
            bbox = meta.get("bbox")
            boxes = (np.asarray(bbox, np.float32).reshape(1, 4)
                     if bbox is not None else None)
            # The RGB-D caller has calibrated camera intrinsics.  Supplying
            # them avoids running the ~300 ms monocular MoGe FOV estimator on
            # every frame and gives the mesh the same camera model as depth.
            K = meta.get("camera_intrinsics")
            if K is None:
                cam_int = None
            else:
                # Camera calibration is normally constant for the lifetime of
                # the stream.  Avoid rebuilding and copying this CUDA tensor on
                # every frame, while still handling a calibration update.
                K_key = tuple(float(x) for row in K for x in row)
                if K_key != cached_K_key:
                    cached_cam_int = torch.tensor(
                        K_key, dtype=torch.float32, device="cuda").reshape(1, 3, 3)
                    cached_K_key = K_key
                cam_int = cached_cam_int
            if quiet_sink is None:
                outputs = estimator.process_one_image(
                    img_rgb, bboxes=boxes, cam_int=cam_int,
                    inference_type="body", hand_box_source="body_decoder")
            else:
                with contextlib.redirect_stdout(quiet_sink):
                    outputs = estimator.process_one_image(
                        img_rgb, bboxes=boxes, cam_int=cam_int,
                        inference_type="body", hand_box_source="body_decoder")
        except Exception as e:                      # a bad frame must not kill the stream
            print(f"[worker] inference failed: {e}", flush=True)
            continue
        inference_done = time.perf_counter()
        infer_ms = (inference_done - t0) * 1000.0

        verts_list, joints_list, overlay = [], [], None
        for person in outputs:
            v = person["pred_vertices"] + person["pred_cam_t"][None, :]   # camera optical frame
            if not np.isfinite(v).all():
                continue
            if args.target_faces and collapse is None:
                collapse, faces_out = build_decimation(
                    v.astype(np.float32), faces_full, args.target_faces)
                faces_bytes = faces_out.tobytes()
                print(f"[worker] mesh decimated {faces_full.shape[0]} -> {faces_out.shape[0]} faces",
                      flush=True)
            v = v.astype(np.float32, copy=False)
            verts_list.append(collapse(v) if collapse is not None else v)
            # MHR exposes 70 body/hand landmarks. Put them in the same camera
            # optical frame as the vertices before transport; the RGB-D caller
            # will apply the exact same depth-only registration to both.
            j = (person["pred_keypoints_3d"] +
                 person["pred_cam_t"][None, :]).astype(np.float32, copy=False)
            joints_list.append(j)

        if args.overlay:
            overlay = img
            for person in outputs:
                kp = person["pred_keypoints_2d"]
                kp = np.concatenate([kp, np.ones((kp.shape[0], 1))], -1)
                overlay = skel.draw_skeleton(overlay, kp)
                b = person["bbox"].astype(int)
                cv2.rectangle(overlay, (b[0], b[1]), (b[2], b[3]), (0, 220, 0), 2)

        n_done += 1
        now = time.perf_counter()
        if now - t_win >= 1.0:
            fps = n_done / (now - t_win)
            n_done, t_win = 0, now

        if len(verts_list) == 1:
            verts = verts_list[0]
        elif verts_list:
            verts = np.concatenate(verts_list, axis=0, dtype=np.float32)
        else:
            verts = np.empty((0, 3), np.float32)
        joints = (np.concatenate(joints_list, axis=0, dtype=np.float32)
                  if joints_list else np.empty((0, 3), np.float32))
        header = json.dumps({
            "stamp_sec": meta["stamp_sec"], "stamp_nsec": meta["stamp_nsec"],
            "frame_id": meta["frame_id"], "n_persons": len(verts_list),
            "verts_per_person": int(verts.shape[0] // max(len(verts_list), 1)),
            "n_faces": int(faces_out.shape[0]), "infer_ms": round(infer_ms, 1),
            "joints_per_person": int(joints.shape[0] // max(len(joints_list), 1)),
            "fps": round(fps, 2),
            "focal": float(outputs[0]["focal_length"]) if outputs else 0.0,
        }).encode()

        parts = [header, faces_bytes, verts.tobytes(), joints.tobytes()]
        if overlay is not None:
            ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 75])
            parts.append(buf.tobytes() if ok else b"")
        else:
            parts.append(b"")
        post_done = time.perf_counter()
        pub.send(pack(parts))
        send_done = time.perf_counter()
        timings = np.array([
            decode_done - frame_t0,
            preprocess_done - decode_done,
            inference_done - preprocess_done,
            post_done - inference_done,
            send_done - post_done,
        ]) * 1000.0
        profile_sum += timings
        profile_n += 1
        avg = profile_sum / profile_n
        print(f"\r[worker] {len(verts_list)} person(s)  infer {infer_ms:6.1f} ms  {fps:4.1f} fps"
              f"  avg decode/rgb/infer/post/send="
              f"{avg[0]:.1f}/{avg[1]:.1f}/{avg[2]:.1f}/{avg[3]:.1f}/{avg[4]:.1f} ms",
              end="", flush=True)


def build_decimation(verts, faces, target_faces):
    """Decimate once and reuse the edge-collapse mapping on every later frame.

    The collapse mapping only ever merges vertices that share an edge, so a
    decimated vertex can never straddle two body parts -- mapping decimated
    vertices back by nearest neighbour does exactly that, and the stray vertex
    then drags a web of triangles between e.g. an arm and the torso as the
    person moves.
    """
    import fast_simplification as fsim
    reduction = max(0.0, min(0.95, 1.0 - target_faces / faces.shape[0]))
    _, _, collapses = fsim.simplify(verts, faces, target_reduction=reduction,
                                    return_collapses=True)
    _, faces_dec, mapping = fsim.replay_simplification(verts, faces, collapses)
    mapping = np.asarray(mapping, np.int64)
    n_out = int(mapping.max()) + 1
    counts = np.maximum(np.bincount(mapping, minlength=n_out), 1).astype(np.float32)

    def collapse(v):
        acc = np.stack([np.bincount(mapping, weights=v[:, k], minlength=n_out)
                        for k in range(3)], axis=1)
        return (acc / counts[:, None]).astype(np.float32)

    return collapse, np.asarray(faces_dec, np.int32)


if __name__ == "__main__":
    main()
