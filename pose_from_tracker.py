#!/usr/bin/env python3
"""Bodies for the people semantic_perception segments.

Division of labour: detection and segmentation are upstream, identity belongs
to tracker.py, and choosing who matters belongs to whatever policy is watching
-- wave_detector, today.  This node fits bodies and does not decide anything
about them, so it depends on the segmenter alone and keeps running whether or
not anybody has been designated.

Everyone visible enough to fit gets fitted, and their joints go out together.
Only the designated track is grounded against depth, refined onto its own
point cloud and drawn: that half costs more than the fit, and a body nobody
asked for does not need to stand on the floor.

The loop is ordered by what the output is for, because everything in front of
a message is latency added to it:

  1. the target, fitted alone -- a second person in frame must not put their
     own fit in front of the body somebody is acting on
  2. everyone else, for the gesture policy, at --others-hz while there is a
     target (they are secondary then); every frame when there is not
  3. ~/joints, everybody's raw joints.  This is the gesture policy's input and
     the regressor has already answered it: whether a wrist is above a crown
     is settled inside one body, and nothing below can change it.  So it goes
     out before any of it.
  4. the target grounded against depth and registered onto its own point
     cloud, then ~/human_pose, ~/human_joints and ~/human_facing.  Only the
     target: nobody else's metric placement is being acted on.
  5. ~/human_mesh last, and only when something is subscribed to it.  It is a
     picture: 2500 decimated triangles are still 7500 Point objects, and a
     body drawn for nobody would push the next frame's skeleton back.

Measured on an RTX 4070 with two people in frame, camera stamp to the
skeleton on the wire, not counting the camera driver's own 30-45 ms:
51 ms median / 72 p90 headless, 66 / 77 with RViz attached.

  in   /tracker/instances      vision_msgs/Detection2DArray -- the gate: class,
                               score and instance id per detection
       /tracker/instance_mask  16UC1, pixel = instance id, same stamp
       /tracker/tracks         instance id to track id, as "<track>:<instance>"
       /tracker/target         std_msgs/Int32, the track to draw, -1 for none
       the camera's colour/depth/info topics, paired BY STAMP.

  out  ~/human_pose    geometry_msgs/PoseStamped  pelvis position, +x = facing
       ~/human_joints  visualization_msgs/Marker  LINE_LIST, the SMPL skeleton
       ~/human_facing  visualization_msgs/Marker  ARROW along the facing axis
       ~/joints        sensor_msgs/PointCloud2, 25 points per body -- the 24
                       SMPL joints then the crown -- with w carrying the
                       instance id.  Everyone who was fitted this frame.
       ~/people        visualization_msgs/MarkerArray  everyone: a skeleton
                       each, and a label saying how much of them is in view.
                       For looking at, not for acting on -- the bodies in it
                       are placed by depth at the pelvis and not refined.
       ~/human_mesh    visualization_msgs/Marker  TRIANGLE_LIST, the body,
                       published last and only to a live subscriber

Notes on the contract:

  * "Nobody designated" is a -1 on /tracker/target, not a silence, and the
    markers are deleted rather than left standing.  A consumer can tell
    "nobody" from "the publisher died"; this node treats an instance array
    older than --tracker-timeout as the second.

  * A frame the target could not be fitted on is neither.  It is a tracks
    message that did not arrive for that stamp, or a mask a hair too small,
    and --idle-hold keeps the last body up across it -- 33 ms stale beats
    blinking, and the temporal filters keep the state that one dropped frame
    must not throw away.

  * The mask and the colour frame must be paired by stamp.  The mask indexes
    the frame it was computed from; applying it to whichever colour frame is
    newest shifts it by a frame of motion, worst exactly when the person moves.

  * Temporal filtering (shape smoothing, root-jump rejection) is on, and is
    legitimate here in a way it is not against a per-frame detector: the track
    upstream is a real identity, held across the person turning, lowering their
    arm or being briefly occluded.  This node does not create that identity, it
    just relies on it -- and drops all of it the moment the target goes away.

  * The mask says which pixels, the detection says which box.  Both arrive for
    the same stamp, so neither has to be re-derived from the other: the box a
    crop is taken from is the box the segmenter actually detected.
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
from collections import Counter, deque
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


# What a person's mask covers of their own detection box with nothing in front
# of them, measured over 81 person-frames of this camera.  The occlusion gate
# reads a fraction of it.
NOMINAL_FILL = 0.39

# What a person is, in metres, for turning a distance into "how many pixels
# tall should they be".  Only the ratio matters, so the exact figure does not.
PERSON_HEIGHT_M = 1.7

# Everyone's bodies go out as one cloud: 24 SMPL joints then the crown, w
# carrying the instance id they belong to.  The crown is the highest mesh
# vertex within HEAD_RADIUS_M of the head joint -- it travels with the joints
# because a gesture policy needs it and does not want the 6890-vertex mesh to
# get it.  The highest vertex of the whole body would be the raised hand.
JOINTS_PER_BODY = 24
CROWN_POINT = 1
BODY_POINTS = JOINTS_PER_BODY + CROWN_POINT
HEAD_RADIUS_M = 0.25
J_HEAD = 15


def visible_fraction(mask_px: int, box_area: float) -> float:
    """What the mask covers of its own box; 1.0 for a solid silhouette.

    This catches something standing in front of a person: the detector still
    draws the box around the whole of them, and the mask comes back with a
    hole in it.  It cannot catch a person who is merely CROPPED -- the box is
    drawn round whatever is visible, so somebody showing only their head fills
    that box and reads 1.00 here.  visible_height_fraction is for that.
    """
    if box_area <= 0.0:
        return 0.0
    return min(1.0, (mask_px / box_area) / NOMINAL_FILL)


def visible_height_fraction(mask: np.ndarray, depth_m: np.ndarray,
                            fy: float) -> float:
    """How much of a person's own length is in view, 1.0 for all of them.

    Depth is what makes this possible: the camera says how many pixels a
    PERSON_HEIGHT_M person spans at the distance the mask actually sits, and
    the mask's longest extent says how much of that is there.  Measured live,
    real people -- including seated ones with their legs behind a desk -- read
    0.65 to 0.70, while a 48x17 fragment of a person read 0.12.

    The longest extent rather than the height, so somebody lying down is not
    rejected for being short, and the test only ever grows more permissive
    when it is unsure.
    """
    if fy <= 0.0:
        return 1.0
    rows, cols = mask.any(axis=1), mask.any(axis=0)
    ys, xs = np.flatnonzero(rows), np.flatnonzero(cols)
    if not len(ys) or not len(xs):
        return 1.0
    extent = float(max(ys[-1] - ys[0] + 1, xs[-1] - xs[0] + 1))
    # A median over every fourth pixel: this only needs the distance to the
    # person, not a depth image.
    z = depth_m[::4, ::4][mask[::4, ::4] > 0]
    z = z[(z > 0.3) & (z < 10.0)]
    if not len(z):
        return 1.0
    expected = fy * PERSON_HEIGHT_M / float(np.median(z))
    return 1.0 if expected <= 0.0 else float(extent / expected)


def crown_of(verts: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """The top of the head: the highest mesh vertex near the head joint.

    Not the highest vertex of the body, which is a raised hand -- the very
    thing a gesture policy wants to compare against this.  Y is down.
    """
    near = verts[np.linalg.norm(verts - joints[J_HEAD], axis=1) < HEAD_RADIUS_M]
    return near[near[:, 1].argmin()] if len(near) else joints[J_HEAD]


def decimate_smpl(verts: np.ndarray, faces: np.ndarray, target_faces: int,
                  max_rounds: int = 40) -> np.ndarray:
    """A coarser SMPL topology, as flat TRIANGLE_LIST indices into `verts`.

    RViz's TRIANGLE_LIST has no index buffer, so every triangle spells out its
    three corners and SMPL's 13776 faces become 41328 Point objects -- more
    pure Python per frame than the fit and the grounding together.  Coarser
    costs less, and the body has to stay driven by the original vertices,
    since those are what the next pose moves.

    So the decimation runs along the mesh's own edges: an edge (u, w) is
    collapsed by merging w *into* u, never into a new point between them.
    Every surviving vertex is therefore an original SMPL vertex, and every
    surviving triangle's corners were neighbours on the original surface.

    The obvious alternative -- quadric decimation, then map each new vertex
    back to the nearest SMPL vertex -- is what webbed the limbs to the torso:
    "nearest" crosses from an arm to the body wherever the two surfaces come
    close, and the triangle then stretches between them when the arm moves.
    Distance cannot tell an arm from the chest behind it; connectivity can.

    Only the ordering of collapses depends on the pose this is measured on, so
    any fitted body will do.
    """
    import heapq

    v = np.asarray(verts, np.float64)
    F = np.asarray(faces, np.int32).copy()
    alive = np.ones(len(F), bool)
    count = len(F)
    if target_faces <= 0 or target_faces >= count:
        return F.reshape(-1)

    for _ in range(max_rounds):
        if count <= target_faces:
            break
        # Neighbours and edges, rebuilt from the faces still alive.  A pass
        # can only consume the edges it started with, so passes repeat until
        # one makes no progress.
        nbr, vfaces = {}, {}
        for fi in np.flatnonzero(alive):
            f = F[fi]
            for a in f:
                vfaces.setdefault(int(a), set()).add(int(fi))
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[0], f[2])):
                nbr.setdefault(int(a), set()).add(int(b))
                nbr.setdefault(int(b), set()).add(int(a))
        heap = [(float(np.linalg.norm(v[a] - v[b])), a, b)
                for a in nbr for b in nbr[a] if a < b]
        heapq.heapify(heap)

        before, dead = count, set()
        while count > target_faces and heap:
            _, u, w = heapq.heappop(heap)
            if u in dead or w in dead:
                continue
            # On a closed surface an edge's endpoints share exactly the two
            # opposite corners.  More than that and merging them folds the
            # surface onto itself.
            if len(nbr[u] & nbr[w]) > 2:
                continue
            if len(nbr[u]) < len(nbr[w]):
                u, w = w, u
            dead.add(w)
            for fi in vfaces[w]:
                if not alive[fi]:
                    continue
                F[fi] = np.where(F[fi] == w, u, F[fi])
                f = F[fi]
                if f[0] == f[1] or f[1] == f[2] or f[0] == f[2]:
                    alive[fi] = False
                    count -= 1
                else:
                    vfaces[u].add(fi)
            for x in nbr[w]:
                if x != u and x not in dead:
                    nbr[u].add(x)
                    nbr[x].discard(w)
                    nbr[x].add(u)
            nbr[u].discard(w)
        if count == before:
            break
    return F[alive].reshape(-1)


class PointPool:
    """Reusable geometry_msgs/Point objects to fill a Marker with.

    Building a Point costs far more than setting one: 41328 fresh ones is
    51 ms, the same objects mutated in place is 15 ms.  publish() serializes
    before it returns, so last frame's buffer is free to overwrite.
    """

    def __init__(self):
        from geometry_msgs.msg import Point
        self._Point = Point
        self._pts = []

    def fill(self, a: np.ndarray):
        # One numpy-to-float conversion for the whole array beats three per
        # point, and is most of the difference on its own.
        rows = a.tolist()
        if len(self._pts) < len(rows):
            self._pts.extend(self._Point()
                             for _ in range(len(rows) - len(self._pts)))
        pts = self._pts[:len(rows)]
        for p, r in zip(pts, rows):
            p.x, p.y, p.z = r
        return pts


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
    ap.add_argument("--instances", default="/tracker/instances",
                    help="the gate: class, score and id per segmented instance")
    ap.add_argument("--instance-mask", default="/tracker/instance_mask",
                    help="16UC1, pixel = instance id, from the same frame")
    ap.add_argument("--tracks", default="/tracker/tracks",
                    help="instance id to track id, written as '<track>:<instance>'")
    ap.add_argument("--target", default="/tracker/target",
                    help="the track id whose body is grounded, refined and "
                         "drawn; everyone else gets joints only")
    ap.add_argument("--min-visible-height", type=float, default=0.35,
                    help="skip anybody less of whose own length is in view "
                         "than this, measured against how tall a person is at "
                         "the distance depth puts them.  Real people read "
                         "0.65-0.70 here, seated ones included; a head-sized "
                         "fragment reads 0.12, and fitting one hands TokenHMR "
                         "a crop it can only hallucinate a body from")
    ap.add_argument("--max-occlusion", type=float, default=0.8,
                    help="skip the fit for anybody hidden by more than this "
                         "fraction of what an unoccluded person's mask covers")
    ap.add_argument("--ns", default="/pose",
                    help="namespace the results are published under")
    ap.add_argument("--max-persons", type=int, default=4,
                    help="most people fitted per frame; the rest wait for the "
                         "next one, so a crowd cannot stall the loop")
    ap.add_argument("--human-class", default="human",
                    help="the class_id semantic_perception labels people with")
    ap.add_argument("--min-mask-px", type=int, default=600,
                    help="skip anybody whose mask is smaller than this: too "
                         "few pixels to crop a body out of")
    ap.add_argument("--tracker-timeout", type=float, default=2.0,
                    help="seconds without any detections before going idle; "
                         "upstream publishes every frame, empty ones included, "
                         "so silence means the publisher stopped rather than "
                         "that the room is empty")
    ap.add_argument("--depth-tolerance-ms", type=float, default=60.0,
                    help="how far the depth frame may sit from the mask's stamp")
    ap.add_argument("--max-hz", type=float, default=30.0,
                    help="cap on fits per second; the camera runs at 30")
    ap.add_argument("--mesh-hz", type=float, default=10.0,
                    help="cap on how often the mesh marker goes out.  It is a "
                         "picture, not an input: drawing it every frame makes "
                         "the loop slower than the camera, and then every "
                         "skeleton queues behind a mesh")
    ap.add_argument("--root-max-speed", type=float, default=2.0,
                    help="metres per second the pelvis may move before it is "
                         "treated as an outlier and clamped.  Walking is ~1.5")
    ap.add_argument("--root-grace", type=int, default=10,
                    help="consecutive outlier frames before a jump is believed "
                         "and the mesh is allowed to teleport there.  The "
                         "depth-plus-model root is noisy -- measured, it moves "
                         "86 cm between frames at p99 -- and at the stock 3 "
                         "frames any excursion lasting 0.15 s got through and "
                         "was published as a body jumping across the room.  At "
                         "10 it has to hold for half a second, which an "
                         "outlier does not and a person walking out of an "
                         "occlusion does")
    ap.add_argument("--idle-hold", type=float, default=0.5,
                    help="keep the last body on screen for this long when the "
                         "target cannot be fitted; below it a dropped frame "
                         "reads as a departure and the body blinks")
    ap.add_argument("--people-hz", type=float, default=10.0,
                    help="how often ~/people is redrawn; it is a picture and "
                         "is skipped entirely when nothing subscribes to it")
    ap.add_argument("--label-size", type=float, default=0.12,
                    help="height of the per-person text label, in metres")
    ap.add_argument("--mesh-faces", type=int, default=2500,
                    help="decimate the body to this many triangles before "
                         "publishing it; 0 keeps SMPL's 13776, which costs "
                         "41328 Point objects a frame")
    ap.add_argument("--others-hz", type=float, default=25.0,
                    help="how often the people who are not the target are "
                         "fitted, while there is a target; with nobody "
                         "designated everyone is fitted every frame anyway.  "
                         "It is the rate at which a NEW waver is noticed, so "
                         "it is not free to lower: at 10 Hz a second person "
                         "was sampled at 7.8 Hz and needed 0.38 s to clear "
                         "--wave-min-frames, against 0.17 s at 25, and the "
                         "difference cost 1.2 ms of skeleton latency with one "
                         "other person in frame.  The cost scales with how "
                         "many there are, which --max-persons caps")
    ap.add_argument("--publish-frame", choices=("camera", "world"),
                    default="camera",
                    help="'camera' is the optical frame itself and needs "
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
    from visualization_msgs.msg import Marker, MarkerArray

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
    st = {"K": None, "cam_frame": None, "instances": None, "instances_t": 0.0,
          "labels": {}, "tracks": {}, "target": -1}

    # The fit loop sleeps between frames, and what it is waiting for is one of
    # these callbacks.  Polling for them on a timer costs half the poll
    # interval on every frame, in the middle of the path being measured.
    arrived = threading.Event()

    def on_instances(m):
        st["instances"] = m
        st["instances_t"] = time.monotonic()
        arrived.set()

    def on_labels(m):
        """The 16UC1 instance mask, kept by stamp beside its detections."""
        st["labels"][stamp_ns(m.header)] = np.frombuffer(
            m.data, np.uint16).reshape(m.height, m.width)
        while len(st["labels"]) > CAM_QUEUE:
            del st["labels"][next(iter(st["labels"]))]
        arrived.set()

    def on_tracks(m):
        """instance id -> track id, for this stamp.

        tracker.py writes Detection2D.id as "<track>:<instance>", so the two
        numbers travel together and either side can be joined on.
        """
        mapping = {}
        for det in m.detections:
            parts = str(det.id).split(":")
            if len(parts) == 2 and parts[1].isdigit():
                mapping[int(parts[1])] = int(parts[0])
        st["tracks"][stamp_ns(m.header)] = mapping
        while len(st["tracks"]) > CAM_QUEUE:
            del st["tracks"][next(iter(st["tracks"]))]

    def on_target(m):
        st["target"] = int(m.data)

    def on_rgb(m):
        st["cam_frame"] = m.header.frame_id or st["cam_frame"]
        rgb_buf.push(m)

    from std_msgs.msg import Int32
    from vision_msgs.msg import Detection2DArray
    node.create_subscription(Detection2DArray, args.instances, on_instances, qos)
    node.create_subscription(Image, args.instance_mask, on_labels, cam_qos)
    node.create_subscription(Detection2DArray, args.tracks, on_tracks, cam_qos)
    node.create_subscription(Int32, args.target, on_target, 10)
    PIPE.subscribe_camera(
        node, cam_qos, on_rgb, depth_buf.push,
        lambda m: st.update(K=np.array(m.k, np.float32).reshape(3, 3)))

    ns = args.ns.rstrip("/")
    pub_pose = node.create_publisher(PoseStamped, f"{ns}/human_pose", 1)
    pub_mesh = node.create_publisher(Marker, f"{ns}/human_mesh", 1)
    pub_facing = node.create_publisher(Marker, f"{ns}/human_facing", 1)
    pub_joints = node.create_publisher(Marker, f"{ns}/human_joints", 1)
    pub_people = node.create_publisher(MarkerArray, f"{ns}/people", 1)
    from sensor_msgs.msg import PointCloud2
    pub_bodies = node.create_publisher(PointCloud2, f"{ns}/joints", qos)
    # The cloud the registration actually fits to -- eroded, de-spiked and
    # depth-banded by person_cloud.  Published because the alternative is
    # judging the mesh by eye against /tracker/target_cloud, which is every
    # masked pixel the sensor returned including the silhouette band the fit
    # deliberately throws away: the mesh then looks wrong against points
    # nothing is trying to match it to.
    pub_fit_cloud = node.create_publisher(PointCloud2, f"{ns}/fit_cloud", qos)
    print(f"pose_from_tracker: {args.instances} -> {ns}/joints (everyone), "
          f"{ns}/human_{{pose,mesh,facing,joints}} (the target from "
          f"{args.target})", flush=True)

    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    # Filled in from the first body fitted; the pose only orders the
    # collapses, it cannot change what is legal to collapse.
    tri = [None]
    bones = np.asarray(PIPE.SMPL_BONES, np.int32).reshape(-1)
    root_stab = PIPE.RootStabiliser(args.root_max_speed, args.root_grace)
    shown = [False]
    last_mesh_t = [0.0]
    last_others_t = [0.0]
    last_drawn = [0.0]
    last_people_t = [0.0]
    people_shown = [set()]        # marker ids currently up in the "people" ns
    last_target = [None]
    why = Counter()
    prev_root, prev_raw = [None], [None]
    jumps = {"in": deque(maxlen=600), "out": deque(maxlen=600)}
    # What the depth fit was given and what it did with it.  A registration
    # that declines to run leaves the mesh wherever the regressor put it, which
    # is the failure that took longest to find, so it is counted out loud.
    fit_why, root_why = Counter(), Counter()
    fit_dz = deque(maxlen=600)
    fit_match = deque(maxlen=600)
    root_gap = deque(maxlen=600)
    last_logged = [0]
    mesh_pool, bone_pool = PointPool(), PointPool()

    def marker(mid, mtype, frame_id, stamp, scale, colour):
        m = Marker()
        m.header.frame_id, m.header.stamp = frame_id, stamp
        m.ns, m.id, m.action, m.type = "human", mid, Marker.ADD, mtype
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = scale
        m.color = colour
        return m

    def publish_bodies(bodies, header):
        """Everyone's raw joints, one cloud, w = the instance id.

        This is the machine-readable output the gesture policy reads; it is
        published whether or not anybody is designated, because deciding who
        matters is somebody else's job.
        """
        import array
        from sensor_msgs.msg import PointCloud2, PointField
        data = np.zeros((len(bodies) * BODY_POINTS, 4), np.float32)
        for k, (inst_id, joints, crown) in enumerate(bodies):
            lo = k * BODY_POINTS
            data[lo:lo + JOINTS_PER_BODY, :3] = joints[:JOINTS_PER_BODY]
            data[lo + JOINTS_PER_BODY, :3] = crown
            data[lo:lo + BODY_POINTS, 3] = float(inst_id)
        msg = PointCloud2()
        msg.header.stamp = header.stamp
        msg.header.frame_id = st["cam_frame"] or ""
        msg.height, msg.width = 1, len(data)
        msg.is_dense = True
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * len(data)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="w", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = array.array("B", data.tobytes())
        pub_bodies.publish(msg)

    def publish_fit_cloud(cloud, frame, stamp):
        """What refine_to_cloud was given, as XYZ, for a viewer to check."""
        if pub_fit_cloud.get_subscription_count() == 0 or not len(cloud):
            return
        import array
        from sensor_msgs.msg import PointField
        pts = np.asarray(cloud, np.float32)
        msg = PointCloud2()
        msg.header.frame_id, msg.header.stamp = frame, stamp
        msg.height, msg.width = 1, len(pts)
        msg.is_dense, msg.is_bigendian = True, False
        msg.point_step, msg.row_step = 12, 12 * len(pts)
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.data = array.array("B", pts.tobytes())
        pub_fit_cloud.publish(msg)

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

    def clear_people():
        """Take down everybody's markers, for when there is nobody left.

        Not part of clear(): that fires on a target switch too, and the other
        people in the room are still standing there when the target changes.
        This is only for a frame with no humans in it at all, which is the one
        case publish_people never sees -- the loop turns back before it.
        """
        if not people_shown[0]:
            return
        arr = MarkerArray()
        for mid in people_shown[0]:
            m = Marker()
            m.header.frame_id = st["cam_frame"] or args.world_frame
            m.header.stamp = node.get_clock().now().to_msg()
            m.ns, m.id, m.action = "people", mid, Marker.DELETE
            arr.markers.append(m)
        pub_people.publish(arr)
        people_shown[0] = set()

    def go_idle():
        """Nothing to draw this frame.  Usually that is not news.

        A frame the target could not be fitted on is not a departure: it is a
        tracks message that did not arrive for this stamp, or a mask a hair
        too small.  Deleting the markers for that made the body blink in and
        out at half the frame rate, and resetting the temporal filters threw
        away smoothing state over one dropped frame.

        Departure is the tracker's call, and it says so by publishing -1.
        Until then the last body stays up, and only --idle-hold of continuous
        silence takes it down.
        """
        if st["target"] >= 0 and time.monotonic() - last_drawn[0] < args.idle_hold:
            return
        clear()
        # Whoever this was is gone, so every filter that assumed one person
        # is now stale.
        root_stab.reset()
        ML.reset_betas_state()

    def instance_of(det):
        return int(det.id) if str(det.id).isdigit() else 0

    def fit_one(det, labels, rgb, depth_m, fy):
        """TokenHMR for one detection, and how much of them was in view.

        Returns (visible, fit).  `visible` is the more limiting of the two
        gates, which is what a viewer wants to be told -- it is reported for
        everybody, including the people too hidden to fit, because "12% of
        them is showing" is the answer to "why is there no body here".  `fit`
        is None for those, and otherwise the instance id, the joints, the mesh
        and the extras only the target's grounding needs.
        """
        if det is None:
            return 0.0, None
        bb = det.bbox
        box = np.array([bb.center.position.x - bb.size_x / 2.0,
                        bb.center.position.y - bb.size_y / 2.0,
                        bb.center.position.x + bb.size_x / 2.0,
                        bb.center.position.y + bb.size_y / 2.0], np.float32)
        mask = (labels == instance_of(det)).astype(np.uint8)
        px = int(mask.sum())
        area = max(float(bb.size_x * bb.size_y), 1.0)
        fill = visible_fraction(px, area)
        height = visible_height_fraction(mask, depth_m, fy)
        visible = min(fill, height)
        if (px < args.min_mask_px or fill < 1.0 - args.max_occlusion
                or height < args.min_visible_height):
            return visible, None
        verts, joints, pelvis_px, sigma = ML.run_tokenhmr(
            eng, tf_model, rgb, box, device)
        return visible, (instance_of(det), np.asarray(joints, np.float32),
                         np.asarray(verts, np.float32), verts, joints,
                         pelvis_px, sigma, mask)

    def place_target(got, depth_m, K, stamp, prof):
        """Stand the target's body on the floor, refine it, and publish it.

        Everything in here is for the one designated person: it costs more
        than the fit that produced them, and nobody is looking at the rest.
        The pose and the skeleton go out from here -- they are what downstream
        reads, so nothing slower is allowed in front of them.  The mesh is
        handed back instead of published, for the caller to send last.
        Returns None when depth could not place them, which is a frame lost
        rather than an error.
        """
        _, _, _, verts, joints, pelvis_px, sigma, mask = got

        t_stage = time.monotonic()
        forward = PIPE.body_forward(joints)
        raw_root = PIPE.metric_root(pelvis_px, depth_m, K, mask,
                                    PIPE.root_offset_for(forward))
        root = raw_root
        if root is not None:
            blended = PIPE.blend_root(root, PIPE.MODEL_CAM_T, K)
            root = root_stab(blended)
            # What the depth said before anything smoothed it, against what
            # came out: a body that teleports on screen is one of these two
            # moving, and they are not the same failure.
            if root is not None and prev_root[0] is not None:
                jumps["out"].append(float(np.linalg.norm(root - prev_root[0])))
            if prev_raw[0] is not None:
                jumps["in"].append(float(np.linalg.norm(blended - prev_raw[0])))
            prev_raw[0] = blended
            if root is not None:
                prev_root[0] = root.copy()
        if root is None:
            prof["ground"] += (time.monotonic() - t_stage) * 1e3
            return None
        verts_m, joints_m = PIPE.ground(verts, joints, root)
        t_sub = time.monotonic()
        # Hidden-point removal, and it is the expensive half of grounding.  The
        # z-buffer alternative in live_pipeline (FAST_VISIBILITY) costs 0.2 ms
        # against this 5, and is off for a reason worth writing down: at 4-pixel
        # cells 6890 vertices almost never share one, so it barely rejects
        # anything -- measured against the true surface normals it returns 44%
        # back-facing vertices where HPR returns 5%.  Coarser cells trade that
        # for throwing away the front surface: 8% back-facing costs 31% of the
        # vertices HPR keeps.  No setting is both.
        vis_idx = PIPE.visible_vertices(verts_m, K, mask, depth_m)
        prof["g_vis"] += (time.monotonic() - t_sub) * 1e3
        t_sub = time.monotonic()
        fit_cloud = PIPE.person_cloud(depth_m, mask, K)
        publish_fit_cloud(fit_cloud, st["cam_frame"] or args.world_frame, stamp)
        verts_m, joints_m = PIPE.refine_to_cloud(
            verts_m, joints_m, fit_cloud,
            PIPE.vertex_fit_weights(sigma), visible_idx=vis_idx)
        prof["g_refine"] += (time.monotonic() - t_sub) * 1e3
        lr, lroot = PIPE.LAST_REFINE, PIPE.LAST_ROOT
        fit_why[lr["why"]] += 1
        fit_dz.append(abs(lr["dz"]))
        fit_match.append(lr["matched"] / max(lr["cloud"], 1))
        root_why[lroot["why"]] += 1
        root_gap.append(abs(lroot["model"] - lroot["depth"]))
        forward = PIPE.body_forward(joints_m)
        prof["ground"] += (time.monotonic() - t_stage) * 1e3

        # ---- output frame ------------------------------------------------
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

        # ---- publish -------------------------------------------------------
        # Order is priority.  The pelvis pose, the skeleton and the facing
        # arrow are a few hundred bytes each and are what a consumer acts on;
        # the mesh is a picture, costs more than all of them together, and is
        # therefore sent last, by the caller, after everyone else is fitted.
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

        t_stage = time.monotonic()
        if forward is not None:
            am = marker(1, Marker.ARROW, out_frame, stamp,
                        (0.035, 0.07, 0.09),
                        ColorRGBA(r=1.0, g=0.32, b=0.05, a=1.0))
            am.points = to_points(
                [pelvis, pelvis + np.asarray(forward) * PIPE.FACING_ARROW_LENGTH])
            pub_facing.publish(am)

        jm = marker(2, Marker.LINE_LIST, out_frame, stamp,
                    (0.018, 0.0, 0.0),
                    ColorRGBA(r=0.95, g=0.95, b=0.35, a=1.0))
        jm.points = bone_pool.fill(joints_m[bones])
        pub_joints.publish(jm)
        shown[0] = True
        last_drawn[0] = time.monotonic()
        prof["markers"] += (time.monotonic() - t_stage) * 1e3
        # The skeleton's own age, which is the number that matters: this is
        # what downstream consumes, and it is on the wire now.
        prof["pose_age"].append(
            (node.get_clock().now().nanoseconds
             - (int(stamp.sec) * 10 ** 9 + int(stamp.nanosec))) / 1e6)
        return verts_m, out_frame, stamp

    def publish_people(people, track_of, target_instance, depth_m, K, stamp,
                       loop_t0, prof):
        """Everybody in frame: their skeleton, and how much of them is showing.

        The target already gets its own markers, drawn from a body that was
        grounded against depth and refined onto its own point cloud.  These
        are the cheap version for everyone else -- the raw fit, translated so
        the pelvis sits where depth says it is, which costs ~1 ms and puts the
        skeleton on the person rather than wherever the model guessed.  No
        refinement: nobody is acting on these, they are here to be looked at.

        The percentage goes out for people who were NOT fitted too.  That is
        the case worth seeing: a label reading 12% with no skeleton under it
        says the body is missing because the person is, not because anything
        broke.
        """
        if pub_people.get_subscription_count() == 0 or K is None:
            return
        if loop_t0 - last_people_t[0] < 1.0 / max(args.people_hz, 0.1):
            return
        t_stage = time.monotonic()
        last_people_t[0] = loop_t0
        frame = st["cam_frame"] or args.world_frame
        arr = MarkerArray()
        drawn = set()

        def mid_of(track, inst, second=False):
            """A marker id that means the same person next frame.

            Instance ids are frame-local -- the detector renumbers them
            whenever it reorders -- so keying markers on them made a person's
            skeleton and label change id under RViz for no reason, and forced
            a DELETEALL of the whole namespace every publish to clean up after
            it.  Track ids are the only identity that survives a frame
            boundary.  Somebody not yet tracked gets a high, clearly separate
            id off their instance; it is wrong next frame either way, but it
            cannot collide with a real track.
            """
            base = track * 2 if track is not None else 100000 + inst * 2
            return base + (1 if second else 0)

        for inst, visible, fit in people:
            track = track_of.get(inst)
            is_target = target_instance is not None and inst == target_instance
            # The target's own skeleton is already drawn, in its own colour,
            # from a better fit; here it only needs its label.
            colour = (ColorRGBA(r=0.95, g=0.95, b=0.35, a=1.0) if is_target
                      else ColorRGBA(r=0.35, g=0.75, b=0.95, a=0.9))
            anchor = None
            if fit is not None:
                _, j, v, verts, joints, pelvis_px, sigma, mask = fit
                forward = PIPE.body_forward(joints)
                root = PIPE.metric_root(pelvis_px, depth_m, K, mask,
                                        PIPE.root_offset_for(forward))
                if root is not None:
                    verts_m, joints_m = PIPE.ground(verts, joints, root)
                    anchor = crown_of(verts_m, joints_m)
                    if not is_target:
                        sk = marker(mid_of(track, inst), Marker.LINE_LIST,
                                    frame, stamp, (0.012, 0.0, 0.0), colour)
                        sk.ns = "people"
                        sk.points = to_points(joints_m[bones])
                        arr.markers.append(sk)
                        drawn.add(sk.id)
            if anchor is None:
                continue          # no depth for them: nowhere to put a label
            txt = marker(mid_of(track, inst, second=True),
                         Marker.TEXT_VIEW_FACING, frame, stamp,
                         (0.0, 0.0, args.label_size), colour)
            txt.ns = "people"
            drawn.add(txt.id)
            name = f"#{track}" if track is not None else f"i{inst}"
            txt.text = f"{name}  {visible * 100:.0f}%"
            # Y is down in the optical frame, so up is -y: the label floats
            # above the head rather than inside it.
            txt.pose.position.x = float(anchor[0])
            txt.pose.position.y = float(anchor[1] - args.label_size * 2.0)
            txt.pose.position.z = float(anchor[2])
            arr.markers.append(txt)

        # Only what actually left is deleted.  Wiping the namespace and
        # redrawing it at --people-hz meant every person's marker was destroyed
        # and recreated ten times a second; now a person who is still here is
        # an in-place update, and a DELETE means somebody really has gone.
        for mid in people_shown[0] - drawn:
            gone_m = Marker()
            gone_m.header.frame_id, gone_m.header.stamp = frame, stamp
            gone_m.ns, gone_m.id, gone_m.action = "people", mid, Marker.DELETE
            arr.markers.append(gone_m)
        people_shown[0] = drawn
        if arr.markers:
            pub_people.publish(arr)
        prof["people"] += (time.monotonic() - t_stage) * 1e3

    def publish_mesh(verts_m, out_frame, stamp, loop_t0, prof):
        """The body as a picture, sent after everything a machine reads.

        On this loop, not a thread of its own: filling a Marker is pure Python
        and holds the GIL, so a drawing thread does not overlap with the fit,
        it interleaves with it.  Measured: the fit went from 13 ms to 30 and
        the skeleton's p90 from 80 ms to 156.  Cheap and in line beats
        expensive and parallel here.
        """
        if pub_mesh.get_subscription_count() == 0:
            return
        if loop_t0 - last_mesh_t[0] < 1.0 / max(args.mesh_hz, 0.1):
            return
        t_stage = time.monotonic()
        last_mesh_t[0] = loop_t0
        if tri[0] is None:
            t0 = time.monotonic()
            tri[0] = decimate_smpl(verts_m, faces, args.mesh_faces)
            node.get_logger().info(
                f"mesh decimated to {len(tri[0]) // 3} faces "
                f"({len(tri[0])} points, SMPL has {len(faces)}) in "
                f"{(time.monotonic() - t0) * 1e3:.0f} ms")
        mm = marker(0, Marker.TRIANGLE_LIST, out_frame, stamp,
                    (1.0, 1.0, 1.0),
                    ColorRGBA(r=0.10, g=0.85, b=0.55, a=0.95))
        mm.points = mesh_pool.fill(verts_m[tri[0]])
        pub_mesh.publish(mm)
        prof["mesh"] += (time.monotonic() - t_stage) * 1e3
        prof["mesh_age"].append(
            (node.get_clock().now().nanoseconds
             - (int(stamp.sec) * 10 ** 9 + int(stamp.nanosec))) / 1e6)

    def wait_for_input(timeout):
        """Sleep until the next detections or mask land, or timeout."""
        arrived.wait(timeout)
        arrived.clear()

    period = 1.0 / max(args.max_hz, 0.1)
    depth_tol_ns = int(args.depth_tolerance_ms * 1e6)
    t_start = time.monotonic()
    fits = idles = skips = 0
    skip_rgb, skip_depth = [0], [0]
    # Where a fitted frame's time goes.  Kept always, not behind --verbose:
    # the loop is a latency budget and this is the only view of it.
    PROF_KEYS = ("decode", "fit", "ground", "mesh", "markers", "others",
                 "joints", "people", "g_vis", "g_refine")

    def new_prof():
        p = {k: 0.0 for k in PROF_KEYS}
        p["pose_age"], p["mesh_age"] = [], []
        return p

    prof = new_prof()
    # A colour frame older than the buffer holds is gone for good.
    buffer_span_ns = int(rgb_buf.q.maxlen / 30.0 * 1e9)
    last_key = None

    try:
        while rclpy.ok():
            loop_t0 = time.monotonic()
            if args.run_seconds and loop_t0 - t_start > args.run_seconds:
                break

            inst_msg, K = st["instances"], st["K"]
            if (inst_msg is None or K is None
                    or loop_t0 - st["instances_t"] > args.tracker_timeout):
                idles += 1
                go_idle()
                wait_for_input(period)
                continue

            key = stamp_ns(inst_msg.header)
            if key == last_key:
                wait_for_input(period)          # nothing new since the last fit
                continue

            labels = st["labels"].get(key)
            humans = [d for d in inst_msg.detections
                      if d.results and d.results[0].hypothesis.class_id == args.human_class]
            if labels is None or not humans:
                idles += 1
                go_idle()
                clear_people()
                last_key = key
                wait_for_input(period)
                continue

            rgb_msg = rgb_buf.exact(key)
            dep_msg = depth_buf.nearest(key, depth_tol_ns)
            if rgb_msg is None or dep_msg is None:
                # Retry briefly: the frame these detections name may still be
                # in flight.  Give up once it is older than the buffer can
                # hold, so a frame that was missed cannot spin forever.
                skips += 1
                skip_rgb[0] += rgb_msg is None
                skip_depth[0] += dep_msg is None
                if (node.get_clock().now().nanoseconds - key) > buffer_span_ns:
                    last_key = key
                wait_for_input(period)
                continue
            last_key = key

            t_stage = time.monotonic()
            rgb = np.ascontiguousarray(PIPE.decode(rgb_msg))
            depth_m = PIPE.decode_depth(dep_msg)
            if not (depth_m.shape[:2] == rgb.shape[:2] == labels.shape[:2]):
                node.get_logger().warn(
                    f"shape mismatch colour {rgb.shape[:2]} depth "
                    f"{depth_m.shape[:2]} mask {labels.shape[:2]}; depth must "
                    "be registered to colour", throttle_duration_sec=10.0)
                wait_for_input(period)
                continue

            # Which instance the target track is, this frame.  Identity is the
            # tracker's business; all this node does is look the answer up.
            track_of = st["tracks"].get(key, {})
            target_instance = next(
                (i for i, t in track_of.items() if t == st["target"]), None)
            if st["target"] < 0:
                why["none designated"] += 1
            elif not track_of:
                why["no tracks for this stamp"] += 1
            elif target_instance is None:
                why["target not tracked this frame"] += 1

            # A different person is a different body, and every filter here
            # assumes one.  The root stabiliser exists to reject a pelvis that
            # jumps, which is exactly what a switch looks like to it, so it
            # would drag the new body toward where the old one stood; the
            # shape estimate would blend the two builds over its warm-up.  The
            # markers go too: holding the previous person's body for
            # --idle-hold while the target is somebody else is the wrong body
            # in the wrong place.
            if st["target"] != last_target[0]:
                if last_target[0] is not None:
                    node.get_logger().info(
                        f"target #{last_target[0]} -> #{st['target']}: "
                        "resetting the body filters")
                last_target[0] = st["target"]
                clear()
                root_stab.reset()
                ML.reset_betas_state()

            # ---- the target first, and alone ---------------------------------
            # Whoever is designated is the only body anybody is looking at, so
            # nothing else is fitted before it: a second person in frame used
            # to put their own 13 ms in front of the mesh on screen.  Everyone
            # else follows, below, for the gesture policy.
            prof["decode"] += (time.monotonic() - t_stage) * 1e3
            crowd = humans[:args.max_persons]
            target_det = next(
                (d for d in crowd if instance_of(d) == target_instance), None)

            t_stage = time.monotonic()
            bodies = []
            fy = float(K[1, 1]) if K is not None else 0.0
            people = []                      # (instance, visible, fit or None)
            vis, got = (fit_one(target_det, labels, rgb, depth_m, fy)
                        if target_det else (0.0, None))
            if target_det is not None:
                people.append((instance_of(target_det), vis, got))
            prof["fit"] += (time.monotonic() - t_stage) * 1e3
            placed = None
            if got is None:
                # Nobody designated, or the designated one is not fittable this
                # frame.  The others are still worth fitting, below.
                if st["target"] >= 0 and target_instance is not None:
                    why["target too hidden to fit"] += 1
                idles += 1
                go_idle()
            else:
                bodies.append((got[0], got[1], crown_of(got[2], got[1])))

            # ---- everybody else, for the gesture policy ----------------------
            # Raw joints are all a raised hand needs -- it is a comparison
            # inside one body -- and the policy debounces over frames anyway,
            # so these do not have to keep up with the camera while somebody
            # is designated.  With nobody designated there is no target
            # latency to protect and everyone is a candidate, so the rate
            # limit lifts: otherwise /pose/joints would go empty on the frames
            # in between, and the policy would be reading a room that keeps
            # emptying out.
            t_stage = time.monotonic()
            if (target_det is None
                    or loop_t0 - last_others_t[0] >= 1.0 / max(args.others_hz, 0.1)):
                last_others_t[0] = loop_t0
                for det in crowd:
                    if det is target_det:
                        continue
                    vis, other = fit_one(det, labels, rgb, depth_m, fy)
                    people.append((instance_of(det), vis, other))
                    if other is not None:
                        bodies.append(
                            (other[0], other[1], crown_of(other[2], other[1])))
            prof["others"] += (time.monotonic() - t_stage) * 1e3

            # ---- everybody's joints, before anything is registered ----------
            # The gesture policy reads these, and a raised hand is a
            # comparison inside one body: it is answered by the pose the
            # regressor already returned, and none of the depth registration
            # below can change the answer.  So it goes out first -- putting
            # ~14 ms of grounding and refinement in front of the wave decision
            # bought nothing.
            t_stage = time.monotonic()
            publish_bodies(bodies, inst_msg.header)
            prof["joints"] += (time.monotonic() - t_stage) * 1e3

            # ---- and only now, the one person we are following --------------
            # Standing a body on the floor and registering it onto its own
            # point cloud is the expensive half, and it is done for the target
            # alone: nobody else's metric placement is being acted on.
            if got is not None:
                placed = place_target(got, depth_m, K, inst_msg.header.stamp,
                                      prof)
                if placed is not None:
                    fits += 1
                else:
                    why["depth could not place the target"] += 1

            # Last, and only now: the pictures.  Everything a consumer acts
            # on has already gone out.
            publish_people(people, track_of, target_instance, depth_m, K,
                           inst_msg.header.stamp, loop_t0, prof)
            if placed is not None:
                publish_mesh(*placed, loop_t0, prof)

            if fits and fits % 30 == 0 and fits != last_logged[0]:
                last_logged[0] = fits
                age = (node.get_clock().now().nanoseconds - key) / 1e6
                stages = "  ".join(f"{k} {prof[k] / 30.0:.1f}" for k in PROF_KEYS)
                def ages(v, label):
                    return (f"{label} {np.median(v):.0f}/{np.percentile(v, 90):.0f} "
                            f"(p50/p90) ms  " if len(v) else "")
                print(f"[PoseProfile] {stages}  |  "
                      f"{(time.monotonic() - loop_t0) * 1e3:.0f} ms/fit  "
                      f"{ages(prof['pose_age'], 'skeleton-age')}"
                      f"{ages(prof['mesh_age'], 'mesh-age')}"
                      f"frame-age {age:.0f} ms  "
                      f"fits={fits} idle={idles} skip={skips} "
                      f"(rgb {skip_rgb[0]} / depth {skip_depth[0]})  "
                      + "  ".join(f"{k}={v}" for k, v in why.most_common()),
                      flush=True)
                why.clear()
                if len(fit_dz) > 10:
                    dz = np.asarray(fit_dz) * 100
                    mt = np.asarray(fit_match) * 100
                    rg = np.asarray(root_gap) * 100
                    print(f"[Fit] depth-fit moved the mesh p50 "
                          f"{np.median(dz):.1f} p90 {np.percentile(dz,90):.0f} "
                          f"max {dz.max():.0f} cm  on p50 {np.median(mt):.0f}% "
                          f"of the cloud  |  "
                          + "  ".join(f"{k}={v}" for k, v in fit_why.most_common())
                          + f"  |  model-vs-depth range p50 {np.median(rg):.0f} "
                            f"p90 {np.percentile(rg,90):.0f} cm  "
                          + "  ".join(f"{k}={v}" for k, v in root_why.most_common()),
                          flush=True)
                    fit_why.clear(); root_why.clear()
                if len(jumps["out"]) > 30:
                    a, b = np.asarray(jumps["in"]), np.asarray(jumps["out"])
                    print(f"[RootJump] depth said p50 {np.median(a)*100:.1f} "
                          f"p99 {np.percentile(a,99)*100:.0f} max "
                          f"{a.max()*100:.0f} cm  |  published p50 "
                          f"{np.median(b)*100:.1f} p99 {np.percentile(b,99)*100:.0f} "
                          f"max {b.max()*100:.0f} cm  (over {len(b)} frames)",
                          flush=True)
                prof = new_prof()
            left = period - (time.monotonic() - loop_t0)
            if left > 0:
                time.sleep(left)
            arrived.clear()
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
