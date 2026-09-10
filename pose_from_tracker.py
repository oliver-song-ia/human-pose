#!/usr/bin/env python3
"""Body pose for the person wave_detector has locked.

Division of labour: detection, segmentation and identity belong upstream --
wave_detector.py picks one person out of the scene and follows them.  This node
runs no detector and decides nothing about who anyone is; it turns that one
person's pixels into a body and publishes it.

  in   /wave_detector/tracked_human_mask   mono8, full resolution, 0 or 255.
                                           Both the gate and the mask: no white
                                           pixels means nobody is locked, and
                                           then nothing here runs.
       the camera's colour/depth/info topics, paired to the mask BY STAMP.

  out  ~/human_pose    geometry_msgs/PoseStamped  pelvis position, +x = facing
       ~/human_mesh    visualization_msgs/Marker  TRIANGLE_LIST, the body
       ~/human_facing  visualization_msgs/Marker  ARROW along the facing axis
       ~/human_joints  visualization_msgs/Marker  LINE_LIST, the SMPL skeleton

Notes on the contract:

  * "Nobody locked" is a blank mask, not a silence.  wave_detector publishes an
    all-zero mask twice a second while no one is locked, precisely so a
    consumer can tell "nobody" from "the publisher died".  This node treats
    both as idle -- a blank mask, or no mask at all for --tracker-timeout.

  * The mask and the colour frame must be paired by stamp.  The mask indexes
    the frame it was computed from; applying it to whichever colour frame is
    newest shifts it by a frame of motion, worst exactly when the person moves.

  * Temporal filtering (shape smoothing, root-jump rejection) is on, and is
    legitimate here in a way it is not against a per-frame detector: the lock
    upstream is a real identity, held across the person turning, lowering their
    arm or being briefly occluded.  This node does not create that identity, it
    just relies on it -- and drops all of it the moment the lock goes away.

The bounding box comes from the mask's own extent.  wave_detector publishes no
box, and re-deriving one from the mask is exact rather than approximate: the
mask is what the box would have been drawn around.
"""
import os
# Before torch: the intra-op pool otherwise saturates the CPU that the fit and
# the ROS callbacks need.  Same reason, same numbers as mesh_live_o3d.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import live_pipeline as PIPE
import mesh_live_o3d as ML

# Which axis points at the sky, per output frame.  A camera OPTICAL frame is
# x right, y DOWN, z forward, so up there is -y; the floor-aligned world frame
# has z up.  Getting this wrong does not move the heading -- that is the
# quaternion's +x axis either way -- but it rolls the pose about it.
UP_IN_FRAME = {"world": np.array([0.0, 0.0, 1.0]),
               "camera": np.array([0.0, -1.0, 0.0])}

# Frames of colour and depth history to keep, in the middleware queue and in
# the buffer alike.  It has to cover the upstream node's own latency: the mask
# arrives naming a frame that is already ~300 ms old, which is ~9 frames at
# 30 Hz.  45 leaves room for a hiccup without holding a second of images.
CAM_QUEUE = 45


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


class StampBuffer:
    """The last few messages, retrievable by the stamp they carry.

    The mask names the exact colour frame it was computed from, and that frame
    is a few tens of milliseconds behind whatever has arrived since.  A short
    history is what lets the mask be applied to its own frame instead of to a
    newer one it does not describe.
    """

    def __init__(self, size=45):
        self.q = deque(maxlen=size)
        self.lock = threading.Lock()

    def push(self, msg):
        with self.lock:
            self.q.append((stamp_ns(msg.header), msg))

    def exact(self, key_ns):
        with self.lock:
            for ns, msg in reversed(self.q):
                if ns == key_ns:
                    return msg
        return None

    def nearest(self, key_ns, tol_ns):
        """Closest message within `tol_ns`, for a stream on its own clock."""
        best, best_d = None, None
        with self.lock:
            for ns, msg in reversed(self.q):
                d = abs(ns - key_ns)
                if best_d is None or d < best_d:
                    best, best_d = msg, d
        return best if best_d is not None and best_d <= tol_ns else None


def mask_box(mask):
    """Tight xyxy box around the mask, or None if it is blank."""
    cols = mask.any(axis=0)
    rows = mask.any(axis=1)
    if not cols.any():
        return None
    x = np.flatnonzero(cols)
    y = np.flatnonzero(rows)
    return np.array([x[0], y[0], x[-1] + 1, y[-1] + 1], np.float32)


def transform_points(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def tf_to_matrix(tr):
    from scipy.spatial.transform import Rotation
    q, t = tr.transform.rotation, tr.transform.translation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def facing_quaternion(forward, up):
    """Quaternion whose +x axis is `forward`, in the frame `forward` lives in.

    +x forward is the ROS body convention, so the pose reads as a heading
    without the consumer knowing how it was built.
    """
    from scipy.spatial.transform import Rotation
    x = np.asarray(forward, np.float64)
    n = np.linalg.norm(x)
    if n < 1e-6:
        return None
    x = x / n
    up = np.asarray(up, np.float64)
    if abs(float(np.dot(up, x))) > 0.99:          # facing straight up or down
        up = np.array([0.0, 1.0, 0.0])
    z = up - x * float(np.dot(up, x))
    zn = np.linalg.norm(z)
    if zn < 1e-6:
        return None
    z = z / zn
    return Rotation.from_matrix(
        np.stack([x, np.cross(z, x), z], axis=1)).as_quat()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracked-mask",
                    default="/wave_detector/tracked_human_mask",
                    help="mono8 mask of the one locked person")
    ap.add_argument("--ns", default="/pose",
                    help="namespace the results are published under")
    ap.add_argument("--min-mask-px", type=int, default=600,
                    help="treat a mask smaller than this as nobody locked")
    ap.add_argument("--tracker-timeout", type=float, default=2.0,
                    help="seconds without any mask before going idle; upstream "
                         "publishes a blank one twice a second, so silence "
                         "means the publisher stopped, not that nobody is there")
    ap.add_argument("--depth-tolerance-ms", type=float, default=60.0,
                    help="how far the depth frame may sit from the mask's stamp")
    ap.add_argument("--max-hz", type=float, default=15.0)
    ap.add_argument("--mesh-hz", type=float, default=10.0,
                    help="TRIANGLE_LIST is ~1 MB a message; publish it no "
                         "faster than this even when fitting faster")
    ap.add_argument("--publish-frame", choices=("camera", "world"),
                    default="camera",
                    help="'camera' matches wave_detector's own frame and needs "
                         "no TF; 'world' needs semantic_perception running to "
                         "broadcast it, and falls back to the camera frame")
    ap.add_argument("--world-frame", default="semantic_world")
    ap.add_argument("--run-seconds", type=float, default=0.0)
    ap.add_argument("--verbose", action="store_true")
    PIPE.add_camera_args(ap)
    args = ap.parse_args()
    PIPE.apply_camera_args(args)

    import rclpy
    import rclpy.time
    import tf2_ros
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Point, PoseStamped
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    eng, tf_model, faces = ML.load_tokenhmr_engine(device)

    rclpy.init()
    node = rclpy.create_node("pose_from_tracker")
    # The gate wants only the newest mask, so depth=1 is right for it.  The
    # camera streams are the opposite: the mask names ONE frame by stamp, and
    # with depth=1 the middleware keeps only the newest, so every frame between
    # two callbacks is discarded before it can be buffered and the named frame
    # is usually gone.  Measured: 100 of 278 masks unpairable at depth=1.
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
    cam_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=CAM_QUEUE)
    tf_buffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf_buffer, node)

    rgb_buf, depth_buf = StampBuffer(CAM_QUEUE), StampBuffer(CAM_QUEUE)
    st = {"K": None, "cam_frame": None, "mask": None, "mask_t": 0.0}

    def on_mask(m):
        st["mask"] = m
        st["mask_t"] = time.monotonic()

    def on_rgb(m):
        st["cam_frame"] = m.header.frame_id or st["cam_frame"]
        rgb_buf.push(m)

    node.create_subscription(Image, args.tracked_mask, on_mask, qos)
    PIPE.subscribe_camera(
        node, cam_qos, on_rgb, depth_buf.push,
        lambda m: st.update(K=np.array(m.k, np.float32).reshape(3, 3)))

    ns = args.ns.rstrip("/")
    pub_pose = node.create_publisher(PoseStamped, f"{ns}/human_pose", 1)
    pub_mesh = node.create_publisher(Marker, f"{ns}/human_mesh", 1)
    pub_facing = node.create_publisher(Marker, f"{ns}/human_facing", 1)
    pub_joints = node.create_publisher(Marker, f"{ns}/human_joints", 1)
    print(f"pose_from_tracker: gate={args.tracked_mask} -> "
          f"{ns}/human_{{pose,mesh,facing,joints}}", flush=True)

    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    tri = np.asarray(faces, np.int32).reshape(-1)
    bones = np.asarray(PIPE.SMPL_BONES, np.int32).reshape(-1)
    root_stab = PIPE.RootStabiliser(PIPE.MAX_ROOT_SPEED, PIPE.ROOT_JUMP_GRACE)
    shown = [False]
    last_mesh_t = [0.0]

    def marker(mid, mtype, frame_id, stamp, scale, colour):
        m = Marker()
        m.header.frame_id, m.header.stamp = frame_id, stamp
        m.ns, m.id, m.action, m.type = "human", mid, Marker.ADD, mtype
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color = colour
        return m

    def to_points(a):
        return [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in a]

    def clear():
        """Delete the markers once the lock is gone, so a body does not sit in
        the view after the person it belongs to has been released."""
        if not shown[0]:
            return
        for pub, mid in ((pub_mesh, 0), (pub_facing, 1), (pub_joints, 2)):
            m = Marker()
            m.header.frame_id = st["cam_frame"] or args.world_frame
            m.header.stamp = node.get_clock().now().to_msg()
            m.ns, m.id, m.action = "human", mid, Marker.DELETE
            pub.publish(m)
        shown[0] = False

    def go_idle():
        clear()
        # The lock is gone, so every filter that assumed one person is stale.
        root_stab.reset()
        ML.reset_betas_state()

    period = 1.0 / max(args.max_hz, 0.1)
    depth_tol_ns = int(args.depth_tolerance_ms * 1e6)
    t_start = time.monotonic()
    fits = idles = skips = 0
    skip_rgb, skip_depth = [0], [0]
    # A colour frame older than the buffer holds is gone for good.
    buffer_span_ns = int(rgb_buf.q.maxlen / 30.0 * 1e9)
    last_key = None

    try:
        while rclpy.ok():
            loop_t0 = time.monotonic()
            if args.run_seconds and loop_t0 - t_start > args.run_seconds:
                break

            mask_msg, K = st["mask"], st["K"]
            if (mask_msg is None or K is None
                    or loop_t0 - st["mask_t"] > args.tracker_timeout):
                idles += 1
                go_idle()
                time.sleep(period)
                continue

            key = stamp_ns(mask_msg.header)
            if key == last_key:
                time.sleep(period / 4)          # nothing new since the last fit
                continue

            mask = PIPE.decode(mask_msg)
            box = mask_box(mask)
            if box is None or int(mask.astype(bool).sum()) < args.min_mask_px:
                idles += 1                      # blank mask: nobody is locked
                go_idle()
                last_key = key
                time.sleep(period)
                continue

            rgb_msg = rgb_buf.exact(key)
            dep_msg = depth_buf.nearest(key, depth_tol_ns)
            if rgb_msg is None or dep_msg is None:
                # Retry briefly: the frame this mask names may still be in
                # flight.  Give up on it once it is older than the buffer can
                # hold, so a mask whose frame was missed cannot spin forever.
                skips += 1
                skip_rgb[0] += rgb_msg is None
                skip_depth[0] += dep_msg is None
                if (node.get_clock().now().nanoseconds - key) > buffer_span_ns:
                    last_key = key
                time.sleep(period / 4)
                continue
            last_key = key

            rgb = np.ascontiguousarray(PIPE.decode(rgb_msg))
            depth_m = PIPE.decode_depth(dep_msg)
            if not (depth_m.shape[:2] == rgb.shape[:2] == mask.shape[:2]):
                node.get_logger().warn(
                    f"shape mismatch colour {rgb.shape[:2]} depth "
                    f"{depth_m.shape[:2]} mask {mask.shape[:2]}; depth must be "
                    "registered to colour", throttle_duration_sec=10.0)
                time.sleep(period)
                continue
            mask = (mask > 0).astype(np.uint8)

            # ---- the fit -------------------------------------------------
            verts, joints, pelvis_px, sigma = ML.run_tokenhmr(
                eng, tf_model, rgb, box, device)
            forward = PIPE.body_forward(joints)
            root = PIPE.metric_root(pelvis_px, depth_m, K, mask,
                                    PIPE.root_offset_for(forward))
            if root is None:
                time.sleep(period)
                continue
            root = root_stab(PIPE.blend_root(root, PIPE.MODEL_CAM_T, K))
            if root is None:
                time.sleep(period)
                continue
            verts_m, joints_m = PIPE.ground(verts, joints, root)
            vis_idx = PIPE.visible_vertices(verts_m, K, mask, depth_m)
            verts_m, joints_m = PIPE.refine_to_cloud(
                verts_m, joints_m, PIPE.person_cloud(depth_m, mask, K),
                PIPE.vertex_fit_weights(sigma), visible_idx=vis_idx)
            forward = PIPE.body_forward(joints_m)

            # ---- output frame --------------------------------------------
            out_frame, out_up = st["cam_frame"], UP_IN_FRAME["camera"]
            if args.publish_frame == "world":
                try:
                    T = tf_to_matrix(tf_buffer.lookup_transform(
                        args.world_frame, st["cam_frame"], rclpy.time.Time()))
                    verts_m = transform_points(T, verts_m)
                    joints_m = transform_points(T, joints_m)
                    if forward is not None:
                        forward = T[:3, :3] @ np.asarray(forward)
                    out_frame, out_up = args.world_frame, UP_IN_FRAME["world"]
                except Exception as exc:
                    # The camera frame is still correct -- TF relates the two --
                    # so degrade rather than drop the frame.
                    node.get_logger().warn(
                        f"no TF {st['cam_frame']} -> {args.world_frame} "
                        f"({exc}); publishing in the camera frame",
                        throttle_duration_sec=10.0)

            # ---- publish --------------------------------------------------
            stamp = mask_msg.header.stamp
            pelvis = joints_m[0]
            ps = PoseStamped()
            ps.header.frame_id, ps.header.stamp = out_frame, stamp
            ps.pose.position.x = float(pelvis[0])
            ps.pose.position.y = float(pelvis[1])
            ps.pose.position.z = float(pelvis[2])
            q = facing_quaternion(forward, out_up) if forward is not None else None
            if q is None:
                ps.pose.orientation.w = 1.0
            else:
                (ps.pose.orientation.x, ps.pose.orientation.y,
                 ps.pose.orientation.z, ps.pose.orientation.w) = (
                    float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            pub_pose.publish(ps)

            if loop_t0 - last_mesh_t[0] >= 1.0 / max(args.mesh_hz, 0.1):
                last_mesh_t[0] = loop_t0
                mm = marker(0, Marker.TRIANGLE_LIST, out_frame, stamp,
                            (1.0, 1.0, 1.0),
                            ColorRGBA(r=0.10, g=0.85, b=0.55, a=0.95))
                mm.points = to_points(verts_m[tri])
                pub_mesh.publish(mm)

            if forward is not None:
                am = marker(1, Marker.ARROW, out_frame, stamp,
                            (0.035, 0.07, 0.09),
                            ColorRGBA(r=1.0, g=0.32, b=0.05, a=1.0))
                am.points = to_points(
                    [pelvis,
                     pelvis + np.asarray(forward) * PIPE.FACING_ARROW_LENGTH])
                pub_facing.publish(am)

            jm = marker(2, Marker.LINE_LIST, out_frame, stamp,
                        (0.018, 0.0, 0.0),
                        ColorRGBA(r=0.95, g=0.95, b=0.35, a=1.0))
            jm.points = to_points(joints_m[bones])
            pub_joints.publish(jm)
            shown[0] = True

            fits += 1
            if args.verbose and fits % 30 == 0:
                age = (node.get_clock().now().nanoseconds - key) / 1e6
                print(f"fits={fits} idle={idles} "
                      f"skip={skips} (rgb {skip_rgb[0]} / depth {skip_depth[0]}) "
                      f"{(time.monotonic()-loop_t0)*1e3:.0f} ms/fit  "
                      f"frame-age {age:.0f} ms", flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - loop_t0)))
    except KeyboardInterrupt:
        pass
    finally:
        clear()
        print(f"pose_from_tracker: {fits} fits, {idles} idle, {skips} "
              f"unpaired (rgb {skip_rgb[0]} / depth {skip_depth[0]})", flush=True)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
