"""Live SMPL body-mesh recovery on the Gemini 335L RGB-D stream, visualised in
a rotatable open3d 3D window.

The mesh MODEL is not part of this module: a launcher binds `load_model` /
`run_model` (see mesh_live_o3d.py) and calls main().  Everything else -- the
per-frame pipeline, the maths and the viewer -- lives here.

Pipeline per frame:
  YOLO-seg  -> person box + mask
  model     -> full SMPL mesh (6890 verts) + joints, camera frame, METRIC
               scale but only WEAK-PERSPECTIVE (monocular) global translation.
  depth     -> the real metric root: the predicted mesh is root-relative, so we
               place its pelvis at the sensor-observed distance (median
               person-mask depth + torso half-thickness) back-projected with the
               real K.  The model keeps orientation + articulation; the depth
               fixes WHERE.

So the model gives the shape/pose (incl. self-occluded limbs, filled by the SMPL
body prior -- exactly the thing NTU can't supervise), and the depth grounds it in
true metric camera coordinates.

    DISPLAY=:0 ~/anaconda3/envs/human-pose/bin/python mesh_live_o3d.py
"""
import os, sys, time, threading
from pathlib import Path
import numpy as np
import cv2
import torch

# Filled by the bound run_model when detailed live profiling is enabled.
MODEL_PROFILE = {}
# Optional: a model that predicts its own camera translation fills this in (see
# mesh_live_o3d.run_tokenhmr).  None means "no such estimate, use depth alone".
MODEL_CAM_T = None
import open3d as o3d
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
# The segmentation weights belong to the parent project, which is where this
# repository normally sits as a submodule; relative to here that is one level
# up.  A standalone checkout, or a machine that keeps them somewhere else,
# overrides both with the environment variables.
#
# A TensorRT engine is tied to one GPU architecture AND one TensorRT version,
# so every machine needs its own; HUMAN_POSE_YOLO / _PT point at the local ones.
YOLO_PT = os.environ.get(
    "HUMAN_POSE_YOLO_PT",
    str(HERE.parent / "yolo26m-seg-custom_20260908.pt"))
YOLO_ENGINE = os.environ.get(
    "HUMAN_POSE_YOLO",
    str(HERE.parent / "yolo26m-seg-custom_20260908_rtx4070.engine"))
YOLO_MODEL = YOLO_ENGINE if Path(YOLO_ENGINE).exists() else YOLO_PT
# The engine is driven through yolo_trt_runtime rather than ultralytics: the
# wrapper costs 0.2 ms on an RTX 4070 but 20-30 ms on a Jetson, where the host
# side is the bottleneck rather than the GPU.  Measured in this pipeline on an
# AGX Orin, `detect` went 36.8 -> 17.5 ms.  Set HUMAN_POSE_YOLO_DIRECT=0 to fall
# back to ultralytics (needed for a .pt model, which has no engine to drive).
YOLO_DIRECT = os.environ.get("HUMAN_POSE_YOLO_DIRECT", "1") == "1"

# --- camera topics -------------------------------------------------------
# Defaults are the Orbbec Gemini driver's.  A bag, or a RealSense, publishes
# under other names -- the depth aligned to colour rather than a separate depth
# frame, and often only a compressed colour stream -- so both demos take
# --camera-ns / --color-topic / --depth-topic / --info-topic (see
# add_camera_args).  Depth still has to be registered to colour: the pipeline
# indexes the depth image with colour pixels and rejects a size mismatch.
COLOR_TOPIC = os.environ.get("HUMAN_POSE_COLOR_TOPIC", "/camera/color/image_raw")
DEPTH_TOPIC = os.environ.get("HUMAN_POSE_DEPTH_TOPIC", "/camera/depth/image_raw")
INFO_TOPIC = os.environ.get("HUMAN_POSE_INFO_TOPIC", "/camera/color/camera_info")


def add_camera_args(ap):
    """Topic overrides, shared by both launchers so they subscribe alike."""
    ap.add_argument("--camera-ns", default=None, metavar="NS",
                    help="shorthand for a driver that follows the usual layout: "
                         "NS/color/image_raw, NS/depth/image_raw, "
                         "NS/color/camera_info (e.g. --camera-ns /camera3)")
    ap.add_argument("--color-topic", default=None,
                    help="sensor_msgs/Image or CompressedImage (auto-detected "
                         "from a /compressed suffix)")
    ap.add_argument("--depth-topic", default=None,
                    help="registered depth, e.g. /camera3/aligned_depth_to_color/image_raw")
    ap.add_argument("--info-topic", default=None,
                    help="CameraInfo for the COLOUR frame (depth is indexed by "
                         "colour pixels, so its own K is the wrong one)")


def apply_camera_args(args):
    """Resolve --camera-ns / --*-topic into the module globals used below."""
    global COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC
    ns = getattr(args, "camera_ns", None)
    if ns:
        ns = "/" + ns.strip("/")
        COLOR_TOPIC = f"{ns}/color/image_raw"
        DEPTH_TOPIC = f"{ns}/depth/image_raw"
        INFO_TOPIC = f"{ns}/color/camera_info"
    COLOR_TOPIC = getattr(args, "color_topic", None) or COLOR_TOPIC
    DEPTH_TOPIC = getattr(args, "depth_topic", None) or DEPTH_TOPIC
    INFO_TOPIC = getattr(args, "info_topic", None) or INFO_TOPIC
    print(f"topics: colour={COLOR_TOPIC}  depth={DEPTH_TOPIC}  info={INFO_TOPIC}",
          flush=True)


def _msg_type(topic):
    """CompressedImage for a /compressed topic, plain Image otherwise."""
    from sensor_msgs.msg import CompressedImage, Image
    return CompressedImage if topic.rstrip("/").endswith("/compressed") else Image


def subscribe_camera(node, qos, on_rgb, on_depth, on_info):
    from sensor_msgs.msg import CameraInfo
    node.create_subscription(_msg_type(COLOR_TOPIC), COLOR_TOPIC, on_rgb, qos)
    node.create_subscription(_msg_type(DEPTH_TOPIC), DEPTH_TOPIC, on_depth, qos)
    node.create_subscription(CameraInfo, INFO_TOPIC, on_info, qos)


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def decode(msg):
    """sensor_msgs Image or CompressedImage -> RGB uint8, or the raw depth grid.

    Depth keeps its wire dtype here; decode_depth converts the units, so that
    16UC1 millimetres and 32FC1 metres cannot be confused at the call site.
    """
    if hasattr(msg, "format"):                            # CompressedImage
        buf = np.frombuffer(msg.data, np.uint8)
        if "compressedDepth" in msg.format:
            # image_transport prepends a header struct to the PNG; its size
            # varies by version, so seek the signature rather than assume 12.
            off = bytes(msg.data).find(PNG_MAGIC)
            buf = buf[off if off >= 0 else 12:]
            im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if im is None:
                raise ValueError(f"cannot decode {msg.format}")
            return im
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"cannot decode {msg.format}")
        return np.ascontiguousarray(im[..., ::-1])        # cv2 gives BGR
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    if msg.encoding in ("rgb8", "bgr8"):
        im = rows[:, :msg.width*3].reshape(msg.height, msg.width, 3)
        return np.ascontiguousarray(im if msg.encoding == "rgb8" else im[..., ::-1])
    if msg.encoding in ("mono8", "8UC1"):
        return rows[:, :msg.width].copy()
    if msg.encoding in ("16UC1", "mono16"):
        return rows[:, :msg.width*2].copy().view(np.uint16).reshape(msg.height, msg.width)
    if msg.encoding == "32FC1":
        return rows[:, :msg.width*4].copy().view(np.float32).reshape(msg.height, msg.width)
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def decode_depth(msg):
    """Depth message -> float32 metres.  Integer depth is millimetres by ROS
    convention (16UC1); float depth is already metres (32FC1)."""
    d = decode(msg)
    return d.astype(np.float32) * (1.0 if d.dtype == np.float32 else 0.001)

FLIP_TEST = False        # 2nd flipped forward pass: +accuracy, ~2x slower. off = live
TEMPORAL_SMOOTHING = True
HOLD_BODY_PARTS = True
FAST_VISIBILITY = False
# Also treat a vertex as hidden when the depth sensor measured a surface well in
# front of it (a chair, a desk).  Without this the two-tone mesh only encodes
# self-occlusion, so a back behind a chair still reads as camera-visible.
#
# OFF by default: it is only meaningful once the mesh actually sits on the
# observed surface.  Measured on a seated person, the grounded mesh is a median
# 0.32 m behind the depth the sensor reports along the same ray (and only 20% of
# its vertices even land inside the person mask), so this test currently calls
# ~64% of the body "occluded" -- it would be reporting the misalignment, not the
# chair.  Fix the placement first, then turn this on.
EXTERNAL_OCCLUSION = False
# Blend the model's own camera translation into the depth-derived root (see
# blend_root).  0.0 = depth only (the old behaviour), 1.0 = trust the model.
MODEL_ROOT_BLEND = True
MODEL_ROOT_ALPHA = 0.5
# How far the model's distance may disagree with the sensor's before the blend
# stops believing it.  The blend is there to absorb a wrong ROOT_OFFSET, which
# is a ~10 cm modelling constant, so a disagreement several times that is not
# about ROOT_OFFSET at all (see blend_root).
MODEL_ROOT_MAX_DISAGREE = 0.25   # m
# What the last root decision was made of, for the caller to report.
LAST_ROOT = {"depth": 0.0, "model": 0.0, "out": 0.0, "why": "never ran"}
# How far the pelvis sits behind the skin the depth camera actually sees.  This
# depends on which way the body faces: head-on the sensor sees the chest/belly
# and the pelvis is ~10 cm behind it, but in profile it sees the side of the
# torso, only ~5 cm out from the centre line.  Holding the frontal value through
# a turn pushes the root to the wrong depth -- visible as the mesh jumping as
# someone rotates.  ORIENT_ROOT_OFFSET interpolates by |cos(angle between the
# body's forward axis and the camera axis)|; set it False for the old constant.
ROOT_OFFSET = 0.10       # facing the camera: pelvis ~10cm behind the front skin
ROOT_OFFSET_PROFILE = 0.05   # side-on: half the torso width, not its depth
ORIENT_ROOT_OFFSET = True
# Reject physically impossible root jumps (see RootStabiliser).  0 disables it.
MAX_ROOT_SPEED = 3.0     # m/s
ROOT_JUMP_GRACE = 3      # consecutive outlier frames before the jump is accepted
ICP_MAX_POINTS = 6000    # observed-cloud cap for the depth z-fit (see refine_to_cloud)
MINZ, MAXZ = 0.3, 5.0
# What is allowed into the registration cloud (see person_cloud).  A depth
# sensor does not measure a surface at a depth step, it interpolates across it,
# so every outline in the image produces a band of points that belong to
# nothing.  0 disables any of these.
CLOUD_ERODE_PX = 2          # silhouette band dropped from the mask, in pixels
CLOUD_SPIKE_M = 0.10        # disagreement with the neighbours' median that is not a surface
CLOUD_DEPTH_SPAN_M = 0.9    # kept either side of the body's own median depth
CLOUD_TARGET_POINTS = 2000  # sampling density chosen per frame; 0 keeps the fixed stride
# How much correspondence the depth fit needs before it will move anything, and
# how wide it may look on the first pass to find the body again (refine_to_cloud).
ICP_ACQUIRE_SCALE = 2.0
ICP_MIN_MATCH_FRAC = 0.10
ICP_MIN_MATCHES = 60
# Why the last depth fit did what it did, for the caller to report.  Written by
# refine_to_cloud; a fit that quietly declines to run is the failure that took
# longest to find, so it is not allowed to be silent any more.
LAST_REFINE = {"cloud": 0, "matched": 0, "floor": 0, "dz": 0.0, "why": "never ran"}
# Cloud vs mesh trade-off (single GPU): the decoupled cloud thread streams point
# uploads that contend with the model's CUDA + steal the GIL. Higher CLOUD_HZ / finer
# DISPLAY_STRIDE = smoother cloud but SLOWER mesh inference. 15 Hz / stride 6 keeps
# the cloud responsive while reserving CPU/upload bandwidth for mesh. Raise to 30/3 if
# you care more about the cloud than the mesh rate.
CLOUD_HZ = 15            # leave CPU/GPU-upload headroom for the latency-critical mesh
DISPLAY_STRIDE = 6       # display-only cloud; a sparser cloud substantially cuts contention
# RGB cloud appearance.  The cloud is the sensor ground truth the mesh is judged
# against, so it is drawn near its true colour; GAIN/LIFT only keep very dark
# pixels from disappearing into the background. Raise POINT_SIZE for a denser
# LOOK without paying the upload cost of a finer DISPLAY_STRIDE.
CLOUD_GAIN, CLOUD_LIFT = 0.95, 0.06
CLOUD_POINT_SIZE = 4.5
# camera optical frame (+y down, +z fwd) -> open3d (+y up): 180 deg about x, DISPLAY only
FLIP = np.array([1.0, -1.0, -1.0])

# SMPL 24-joint kinematic tree (for the overlaid skeleton). pred_xyz_jts_29's
# first 24 entries are the SMPL body joints.
SMPL24_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
                  13, 14, 16, 17, 18, 19, 20, 21]
SMPL_BONES = [(j, p) for j, p in enumerate(SMPL24_PARENTS) if p >= 0]

# The model's per-joint predicted uncertainty (sigma) mapped to a
# blue(confident) -> yellow -> red(uncertain) ramp for the joint spheres. Range
# from measured real frames: ~0.003 (torso core) to ~0.03 (folded/occluded limb).
SIG_LO, SIG_HI = 0.004, 0.018


def sig_color(s):
    t = float(np.clip((s - SIG_LO) / (SIG_HI - SIG_LO), 0.0, 1.0))
    if t < 0.5:
        u = t / 0.5; return [u, u, 1.0 - u]            # blue -> yellow
    u = (t - 0.5) / 0.5; return [1.0, 1.0 - u, 0.0]    # yellow -> red


# mesh two-tone: which SMPL surface the depth camera actually sees (HPR) vs the
# back / self-occluded surface it can't. The green part is what registers to the
# cloud; the navy part is inferred by the SMPL prior only.
MESH_VIS = [0.10, 0.85, 0.55]      # camera-visible front surface (teal-green)
MESH_HID = [0.20, 0.24, 0.42]      # occluded / back-facing (dark navy)
JS = o3d.geometry.TriangleMesh.create_sphere(1.0, 8)   # unit-sphere template, joint markers

# Optional second-model hook used by mesh_live_o3d.py's comparison mode.  The
# callable has the same signature as run_model.  Keeping this opt-in preserves
# the original single-model demo and avoids coupling this reusable pipeline to a
# particular alternative regressor.
COMPARE_RUN = None
COMPARE_FACES = None
COMPARE_NAME = None
COMPARE_OFFSET_X = 1.15       # metres in camera coordinates (displayed side-by-side)
DISPLAY_MODEL_NAME = "Mesh"
RUN_SECONDS = 0.0        # >0: close the window automatically after N seconds
# Side window showing the segmentation actually driving the fit.  The Open3D
# scene shows the result; this shows the evidence, which is what you need when
# the mesh misbehaves (bad mask vs bad model).  Scaled down so it sits beside
# the 3D view rather than covering it.
SEG_VIEW = False
SEG_VIEW_WIDTH = 480
# Click a person in the segmentation window to fit that one instead of whichever
# happens to be largest.  The choice is then held by a PersonTrack (below), not
# re-decided each frame.
SEG_PICK = False
PICK_STATE = {"track": None,     # PersonTrack for the chosen person, or None
              "status": None,    # "locked" / "coasting", for the panel banner
              "click": None,     # pending click from the GUI thread
              "people": None,    # candidates of the last frame, for hit-testing
              "scale": 1.0}
# Association gates.  The source here runs anywhere from 3 to 30 Hz, so these
# are expressed per box-diagonal and per second rather than per frame -- a
# threshold in pixels-per-frame that works at 30 Hz drops every lock at 3 Hz.
PICK_COAST_S = 2.0        # hold a lock this long with no accepted match
PICK_POS_GATE = 0.8       # residual from the PREDICTED centre, in box diagonals
PICK_SCALE_TOL = 0.55     # |log| box-size ratio a real person can show
PICK_APPEAR_W = 1.5       # colour weight when several candidates pass the gates
PICK_REACQ_CORREL = 0.45  # colour agreement required to re-acquire after a miss
PICK_APPEAR_LEARN = 0.1   # how fast the colour model follows lighting and pose
PICK_MAX_EXTRAP = 2.0     # cap the coasting prediction at this many diagonals

# Body-facing indicator.  SMPL's labelled left/right torso joints remove the
# 180-degree ambiguity that a plain person bounding box has: anatomical forward
# is up x right.  The arrow is deliberately built as separate geometry so it can
# be replaced cheaply with each inferred mesh.
FACING_ARROW_COLOR = [1.0, 0.32, 0.05]
FACING_ARROW_LENGTH = 0.42


# The mesh model is injected before main() runs.  Both hooks keep the
# signatures every engine in mesh_live_o3d.py implements:
#     load_model(device)                        -> (model_holder, tf, faces)
#     run_model(model_holder, tf, rgb, bbox_xyxy, device)
#         -> (verts[6890,3] metric root-rel, joints[24,3], pelvis_px[2], sigma|None)
load_model = None
run_model = None
# Optional: a launcher may bind this so losing the person also drops any
# temporally-smoothed body-shape estimate (see mesh_live_o3d.reset_betas_state).
reset_shape_state = None


def metric_root(pelvis_px, depth_m, K, mask, root_offset=None):
    """Sensor-grounded 3D pelvis: depth at the pelvis pixel (median in a window,
    inside the mask), backprojected with the real K, pushed ROOT_OFFSET behind
    the observed front surface. Falls back to whole-mask median depth."""
    H, W = depth_m.shape
    u, v = int(round(pelvis_px[0])), int(round(pelvis_px[1]))
    z = None
    if 0 <= u < W and 0 <= v < H:
        p = depth_m[max(0, v-8):v+9, max(0, u-8):u+9]
        mp = mask[max(0, v-8):v+9, max(0, u-8):u+9]
        d = p[(p > MINZ) & (p < MAXZ) & (mp > 0)]
        if d.size >= 5:
            z = float(np.median(d))
    if z is None:
        d = depth_m[(depth_m > MINZ) & (depth_m < MAXZ) & (mask > 0)]
        if d.size < 20:
            return None
        z = float(np.median(d))
        u, v = pelvis_px                       # keep XY from the projected pelvis
    zr = z + (ROOT_OFFSET if root_offset is None else root_offset)
    xr = (u - K[0, 2]) * zr / K[0, 0]
    yr = (v - K[1, 2]) * zr / K[1, 1]
    return np.array([xr, yr, zr], np.float32)


def weak_persp_root(info):
    """Weak-perspective crop translation -> pelvis position in the camera frame.

    A crop-space model predicts (s, tx, ty) for a square crop, which fixes the
    body's apparent SIZE and so its distance -- but expressed with the focal
    length the network was trained on, over a crop of `box_size` source pixels.
    Rescaling by (box_size / crop_px) and re-projecting through the real
    intrinsics recovers a metric translation, the same quantity Fast SAM's
    worker returns directly.  Returns None when the estimate is unusable.
    """
    if not info:
        return None
    ct = np.asarray(info["cam_t"], np.float64)
    if not np.isfinite(ct).all() or ct[2] <= 1e-3:
        return None
    # The crop covers box_size source pixels in crop_px network pixels, so the
    # focal length implied for the SOURCE image is scaled by that ratio.
    src_focal = info["crop_focal"] * info["box_size"] / info["crop_px"]
    if src_focal <= 1e-6:
        return None
    return ct, src_focal


def blend_root(depth_root, cam_info, K, alpha=None):
    """Combine the depth-derived root with the model's own camera translation.

    Depth is authoritative laterally (the pelvis pixel back-projects exactly)
    but its DEPTH relies on ROOT_OFFSET -- a fixed guess at how far the pelvis
    sits behind the visible skin, which is what goes wrong on a seated, turned
    or occluded person.  The model's weak-perspective translation has no such
    assumption: it reads distance from apparent body size.  Take x, y from
    depth and blend z, so a bad ROOT_OFFSET can no longer strand the mesh.
    """
    if depth_root is None:
        return None
    parsed = weak_persp_root(cam_info)
    if parsed is None or not MODEL_ROOT_BLEND:
        return depth_root
    ct, src_focal = parsed
    # cam_t is expressed for the network's focal length over the crop; rescale
    # its depth to the source camera's focal length.
    z_model = float(ct[2]) * (K[0, 0] / src_focal)
    if not (MINZ < z_model < MAXZ):
        LAST_ROOT.update(depth=float(np.asarray(depth_root)[2]), model=z_model,
                         out=float(np.asarray(depth_root)[2]),
                         why="model z out of range")
        return depth_root
    a = MODEL_ROOT_ALPHA if alpha is None else alpha
    out = np.asarray(depth_root, np.float32).copy()
    z_depth = float(out[2])
    # The blend absorbs a wrong ROOT_OFFSET -- a ~10 cm constant.  When the
    # model's distance disagrees with the sensor's by much more than that, it
    # is not correcting ROOT_OFFSET, it is wrong about something else, and
    # averaging into it does real harm.  Measured live on seated people ~3.2 m
    # away with the lower body behind a desk: the model's z ran 1.3-1.4 m short
    # of the sensor's, all day, and at alpha 0.5 that published the mesh 0.68 m
    # in front of the person -- projecting 309 px tall against their 240, and
    # far enough out that refine_to_cloud could no longer find it.  Why weak
    # perspective is biased this way on a half-occluded body is not established
    # here; that it is, is.  Depth measured the surface it can actually see, so
    # past this band depth wins, and the remaining ROOT_OFFSET error is what
    # refine_to_cloud is for.
    if abs(z_model - z_depth) > MODEL_ROOT_MAX_DISAGREE:
        LAST_ROOT.update(depth=z_depth, model=z_model, out=z_depth,
                         why="model too far from depth")
        return depth_root
    z = (1.0 - a) * z_depth + a * z_model
    LAST_ROOT.update(depth=z_depth, model=z_model, out=z, why="blended")
    # Keep x, y consistent with the new depth: the pelvis pixel is fixed, so
    # moving along the ray means scaling the lateral offsets too.
    if z_depth > 1e-6:
        out[0] *= z / z_depth
        out[1] *= z / z_depth
    out[2] = z
    return out


def ground(verts, j29, root_m):
    pelvis = j29[0]
    return (verts - pelvis + root_m).astype(np.float32), \
           (j29 - pelvis + root_m).astype(np.float32)


# Cloud-refine fit weight per vertex = the model's own per-joint RELIABILITY
# (sigma: smaller = more certain = higher weight). The depth alignment
# then follows whatever the model is currently confident about; at close/seated
# range that is the upper body (legs get occluded/truncated -> high sigma -> auto
# down-weighted), so "upper-body priority" EMERGES from reliability rather than a
# hand-set table. The HEAD is additionally forced low (HEAD_W) regardless of its
# sigma: SMPL is a bare skull but the cloud head is skull + HAIR (bigger/offset),
# so it must not pull the fit even when the model is confident about the head.
SIG_W_LO, SIG_W_HI, W_FLOOR = 0.004, 0.030, 0.05     # sigma -> weight ramp (floored)
HEAD_W = 0.10                                        # SMPL head index 15 cap (hair)
# Shoulders (SMPL joints 16/17) are the most reliable landmark the depth camera
# has on a turning person: they stay visible in profile, they are wide so the
# silhouette pins them well, and unlike the head the surface the sensor sees is
# actually the body (no hair offset).  Weighting them above the rest of the mesh
# makes the depth fit follow them rather than the softer torso surface.
SHOULDER_JOINTS = (16, 17)
SHOULDER_W = 2.5             # multiplier on the base weight; 1.0 = no emphasis
VERT_JOINT = None          # per-vertex dominant SMPL joint (6890,), set by load_model
HEAD_CAP = None            # bool (6890,) head vertices (dominant joint 15)
HEAD_LOCAL_REST = None     # (Nhead,3) canonical head-cap in the REST torso frame
PART_JOINTS = {
    "head": (15,),
    "left_arm": (16, 18, 20, 22),
    "right_arm": (17, 19, 21, 23),
    "left_leg": (1, 4, 7, 10),
    "right_leg": (2, 5, 8, 11),
}
PART_LOCAL_REST = None     # canonical per-part vertices expressed in torso frame
PART_MASKS = None          # bool masks; head includes its weighted neck transition
J_REGRESSOR = None         # (24,6890) SMPL joint regressor (MESH-consistent joints)
UPPER_BODY_4DOF_REFINE = False  # enabled by mesh_live_o3d only for HMR2
UPPER_REFINE_STATE = {"valid": False, "yaw_deg": 0.0,
                      "shift": np.zeros(3, np.float64)}


def _torso_frame_smpl(J):
    """(neck origin, R world<-torso) from SMPL-24 joints: up=pelvis->neck,
    right=Lshoulder->Rshoulder, fwd=right x up. Columns of R are the torso axes."""
    up = J[12] - J[0]; up = up / (np.linalg.norm(up) + 1e-9)
    lr = J[17] - J[16]; lr = lr - up * (lr @ up); lr = lr / (np.linalg.norm(lr) + 1e-9)
    fwd = np.cross(lr, up); fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    return J[12], np.stack([lr, up, fwd], axis=1)


def head_in_fov(head_pt, K, hw, mask):
    """Is the head actually observed? False if the head point projects outside the
    image OR onto no person-mask (both mean the camera can't see the head, so
    the head is a prior hallucination -- typically a tilt)."""
    Himg, Wimg = hw
    p = K @ head_pt
    if p[2] <= 1e-6:
        return False
    u, v = p[0] / p[2], p[1] / p[2]
    if not (0 <= u < Wimg and 0 <= v < Himg):
        return False
    ui, vi = int(u), int(v)
    win = mask[max(0, vi-8):vi+9, max(0, ui-8):ui+9]
    return win.size > 0 and float(win.mean()) > 0.15


class HeadHold:
    """Prevent truncated head/limbs folding back into the visible torso.

    Each SMPL part is cached in torso coordinates while its corresponding image
    boundary is not truncated. When truncation occurs, the last reliable rigid
    part (or canonical rest part) is re-embedded into the current torso frame.
    """
    def __init__(self):
        self.part_local = {name: None for name in PART_JOINTS}
        self.last_held = {name: False for name in PART_JOINTS}

    def __call__(self, verts, joints, K, hw, mask):
        if (HEAD_CAP is None or J_REGRESSOR is None or
                PART_LOCAL_REST is None or PART_MASKS is None):
            return verts, joints, True
        Jm = J_REGRESSOR @ verts                     # mesh-consistent SMPL joints
        neck, R = _torso_frame_smpl(Jm)              # R world<-torso, origin=neck
        H, W = hw
        # Use the mask extent with a safety margin rather than requiring contact
        # with the outermost pixels: segmentation masks are commonly eroded by
        # several pixels exactly where a person is truncated.
        ys, xs = np.nonzero(mask)
        margin_y = max(12, int(0.05 * H))
        margin_x = max(12, int(0.04 * W))
        if len(xs):
            top_cut = int(ys.min()) <= margin_y
            bottom_cut = int(ys.max()) >= H - 1 - margin_y
            left_cut = int(xs.min()) <= margin_x
            right_cut = int(xs.max()) >= W - 1 - margin_x
        else:
            top_cut = bottom_cut = left_cut = right_cut = True
        held = {
            "head": top_cut,
            "left_leg": bottom_cut,
            "right_leg": bottom_cut,
            "left_arm": False,
            "right_arm": False,
        }
        # A plausible head centroid must be clearly above the neck along the
        # current torso-up axis. This catches the failure mode where a hallucinated
        # head folds onto the chest and therefore still projects inside person mask.
        head_idx = PART_MASKS["head"]
        head_local_now = (verts[head_idx].mean(0) - neck) @ R
        head_local_neutral = PART_LOCAL_REST["head"].mean(0)
        if (head_local_now[1] < 0.10 or
                np.linalg.norm(head_local_now - head_local_neutral) > 0.12):
            held["head"] = True
        # At a side boundary, hold the anatomical arm whose current projected
        # centroid is closest to that boundary. Holding only one avoids freezing
        # the fully visible arm.
        arm_u = {}
        for name in ("left_arm", "right_arm"):
            idx = PART_MASKS[name]
            p = K @ verts[idx].mean(0)
            arm_u[name] = p[0] / max(p[2], 1e-6)
        if left_cut:
            held[min(arm_u, key=arm_u.get)] = True
        if right_cut:
            held[max(arm_u, key=arm_u.get)] = True

        # Also catch a genuinely projected-out part when the segmentation does
        # not quite touch the edge because of detector erosion.
        for name in PART_JOINTS:
            idx = PART_MASKS[name]
            pt = verts[idx].mean(0)
            p = K @ pt
            if p[2] <= 1e-6:
                held[name] = True
            else:
                u, vv = p[0] / p[2], p[1] / p[2]
                if not (-2 <= u < W + 2 and -2 <= vv < H + 2):
                    held[name] = True

        v = verts.copy()
        for name in PART_JOINTS:
            idx = PART_MASKS[name]
            if not held[name]:
                self.part_local[name] = (verts[idx] - neck) @ R
                continue
            loc = self.part_local[name]
            if loc is None:
                loc = PART_LOCAL_REST[name]
            if loc is not None and len(loc) == int(idx.sum()):
                v[idx] = neck + loc @ R.T

        self.last_held = held.copy()
        # Keep skeleton geometry consistent with the replaced mesh.
        jm_new = (J_REGRESSOR @ v).astype(np.float32)
        j_out = joints.copy()
        j_out[:min(24, len(j_out))] = jm_new[:min(24, len(j_out))]
        return v.astype(np.float32), j_out, not held["head"]


def vertex_fit_weights(sigma):
    """Per-vertex refine weight in [W_FLOOR, 1] from per-joint sigma (reliable ->
    high). `sigma` is the model's (>=24,) per-joint sigma; head vertices are
    capped low."""
    if VERT_JOINT is None or sigma is None:
        return None
    s = np.asarray(sigma)[:24][VERT_JOINT]                 # per-vertex sigma via dom joint
    w = np.clip((SIG_W_HI - s) / (SIG_W_HI - SIG_W_LO), W_FLOOR, 1.0)
    w[VERT_JOINT == 15] *= HEAD_W
    if SHOULDER_W != 1.0:
        w[np.isin(VERT_JOINT, SHOULDER_JOINTS)] *= SHOULDER_W
    # No clamp here: W_FLOOR already floored the sigma ramp above, and only the
    # ratios matter to refine_to_cloud's weighted median.  Re-flooring would
    # quietly raise the deliberately-tiny head weight back to W_FLOOR.
    return w


def _weighted_median(x, w):
    if w is None or w.sum() <= 0:
        return float(np.median(x))
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def person_cloud(depth_m, mask, K, stride=6, target_points=None):
    """Observed person-surface points (metric camera frame) for the ICP refine.

    Only the front surface of one person should reach the registration, so
    three populations are dropped before they can pull on a median:

      * The silhouette band.  A pixel on the outline straddles the person and
        whatever is behind them, and the sensor returns neither -- it returns a
        blend, which lands somewhere in the empty space between the two.  The
        mask is eroded by CLOUD_ERODE_PX first: that costs the outermost few
        millimetres of real surface and removes the entire population.
      * Flying pixels, which are the same failure away from the outline -- the
        edge of a raised arm against the torso behind it, for instance.  A
        sample that disagrees with the median of its own neighbours by
        CLOUD_SPIKE_M is not a surface, it is the sensor interpolating.
      * Whatever the mask leaked onto.  A silhouette that bleeds onto the wall
        contributes points a metre behind the body, and a body -- arms and all
        -- spans well under CLOUD_DEPTH_SPAN_M either side of its own median.

    The stride adapts unless `target_points` is 0, because a fixed stride
    samples the image while the person lives in the world: the same person gave
    ~1200 points at 1.5 m and ~300 at 3.2 m, and 300 is where the registration
    began running out of correspondences to work with.  It never goes sparser
    than the `stride` asked for.
    """
    target_points = (CLOUD_TARGET_POINTS if target_points is None
                     else target_points)
    m8 = np.ascontiguousarray(mask).astype(np.uint8)
    if CLOUD_ERODE_PX > 0:
        k = 2 * CLOUD_ERODE_PX + 1
        thin = cv2.erode(m8, np.ones((k, k), np.uint8))
        # A limb can be narrower than the kernel.  Losing the outline is the
        # point of this; losing the arm is not, so the erosion is kept only
        # while most of the person survives it.
        if cv2.countNonZero(thin) > 0.35 * cv2.countNonZero(m8):
            m8 = thin
    area = cv2.countNonZero(m8)
    if area < 20:
        return np.zeros((0, 3), np.float64)
    if target_points:
        stride = (1 if area <= target_points else
                  int(min(stride, max(1, round(np.sqrt(area / target_points))))))
    # Only over the person.  A finer stride over the whole frame spent most of
    # its time on pixels the mask had already rejected: at 3.2 m the body is
    # about a tenth of the image.
    x0, y0, bw, bh = cv2.boundingRect(m8)
    ys, xs = np.mgrid[y0:y0 + bh:stride, x0:x0 + bw:stride]
    z = depth_m[ys, xs]
    ok = (z > MINZ) & (z < MAXZ) & (m8[ys, xs] > 0)
    if ok.sum() < 20:
        return np.zeros((0, 3), np.float64)

    if CLOUD_SPIKE_M > 0:
        # Each kept sample against the median of its eight grid neighbours.
        # Deliberately on the sampled grid and not the full image: neighbours a
        # stride apart straddle a depth step, and the pixel beside a flying
        # pixel is usually another flying pixel.  Only the kept samples are
        # gathered, so this is ~2k rows of nine, not the whole frame.
        gh, gw = ok.shape
        pad = np.full((gh + 2, gw + 2), np.nan, np.float32)
        pad[1:-1, 1:-1] = np.where(ok, z, np.nan)
        r, c = np.nonzero(ok)
        nbr = np.stack([pad[r + dy, c + dx]
                        for dy in (0, 1, 2) for dx in (0, 1, 2)], 1)
        # The centre is one of the nine and is never NaN, so no all-NaN rows.
        spike = np.abs(z[r, c] - np.nanmedian(nbr, axis=1)) > CLOUD_SPIKE_M
        ok[r[spike], c[spike]] = False

    if CLOUD_DEPTH_SPAN_M > 0 and ok.sum() >= 20:
        r, c = np.nonzero(ok)
        zs = z[r, c]
        far = np.abs(zs - float(np.median(zs))) >= CLOUD_DEPTH_SPAN_M
        # Only if something is left: a badly cropped person can be all tail.
        if (~far).sum() >= 20:
            ok[r[far], c[far]] = False

    ys, xs, z = ys[ok], xs[ok], z[ok]
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1).astype(np.float64)


def visible_vertices(verts, K=None, mask=None, depth_m=None, candidate_idx=None,
                     occlusion_margin=0.15):
    """Mesh front-surface vertices inside this camera's FOV.

    HPR removes the back/self-occluded mesh surface; projection removes everything
    outside the physical image.  The segmentation/depth belong to the *target*
    person cloud, not this source visibility test: intersecting them here would
    shrink the source whenever the detector erodes a silhouette or before the mesh
    has been registered to depth. `candidate_idx` may reuse the preceding HPR set
    but is always reprojected into the current frame.
    """
    if FAST_VISIBILITY and K is not None and mask is not None:
        # Stable O(N) image-space z-buffer.  SMPL has only 6,890 vertices, so
        # recomputing all candidates is both faster and more responsive than
        # HPR plus reuse of the previous frame's visibility subset.
        h, w = mask.shape
        z = verts[:, 2]
        ids = np.flatnonzero(np.isfinite(verts).all(1) & (z > 1e-5))
        vv = verts[ids]
        u = np.rint(K[0, 0] * vv[:, 0] / vv[:, 2] + K[0, 2]).astype(np.int32)
        v = np.rint(K[1, 1] * vv[:, 1] / vv[:, 2] + K[1, 2]).astype(np.int32)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ids, u, v, zz = ids[inside], u[inside], v[inside], vv[inside, 2]
        if not len(ids):
            return ids
        cell = 4
        gw, gh = (w + cell - 1) // cell, (h + cell - 1) // cell
        bins = (v // cell) * gw + (u // cell)
        nearest = np.full(gw * gh, np.inf, np.float32)
        np.minimum.at(nearest, bins, zz)
        front = zz <= nearest[bins] + 0.035          # mesh self-occlusion
        if EXTERNAL_OCCLUSION and depth_m is not None:
            # Self-occlusion alone only asks "is another part of the BODY in the
            # way", so a back hidden behind a chair still came out green.  Compare
            # each vertex against what the sensor actually measured along that
            # ray: a vertex sitting well behind the measured surface has
            # something real in front of it.
            #
            # Only where the depth is valid.  Holes read 0 (or beyond MAXZ) on
            # dark, glossy and out-of-range surfaces, and treating those as
            # "occluded" would punch spurious holes in the mesh -- which is the
            # failure the original comment warned about.
            d = depth_m[v, u]
            measurable = (d > MINZ) & (d < MAXZ)
            front &= ~(measurable & (zz > d + occlusion_margin))
        return ids[front]
    if candidate_idx is None:
        p = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(verts.astype(np.float64)))
        diameter = float(np.linalg.norm(verts.max(0) - verts.min(0)))
        try:
            _, idx = p.hidden_point_removal(
                [0.0, 0.0, 0.0], diameter * 100.0)
            idx = np.asarray(idx, dtype=np.int64)
        except Exception:
            idx = np.arange(len(verts), dtype=np.int64)
    else:
        idx = np.asarray(candidate_idx, dtype=np.int64)

    if K is None or mask is None:
        return idx
    H, W = mask.shape
    vv = verts[idx]
    z = vv[:, 2]
    good_z = z > 1e-6
    u = np.zeros(len(vv), np.int64)
    v = np.zeros(len(vv), np.int64)
    u[good_z] = np.rint(K[0, 0] * vv[good_z, 0] / z[good_z] + K[0, 2]).astype(np.int64)
    v[good_z] = np.rint(K[1, 1] * vv[good_z, 1] / z[good_z] + K[1, 2]).astype(np.int64)
    keep = good_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return idx[keep]


def refine_to_cloud(verts, joints, cloud, vert_w=None, visible_idx=None,
                    max_corr=0.35, iters=2, max_shift=0.4):
    """DEPTH-ONLY (z) snap of the grounded mesh onto the observed cloud, matching
    ONLY the camera-VISIBLE mesh surface (the depth camera sees just the front).
    LATERAL (x,y) is already set accurately by the model's 2D grounding (the pelvis
    pixel back-projected) -- 2D localisation is these models' strength -- and a
    nearest-neighbour lateral correction just SLIDES spuriously on the smooth torso
    surface (that added a ~2cm frontal left-right bias). Only DEPTH placement
    genuinely drifts: the grounding offset is frontal-calibrated, so in profile the
    mesh sits up to ~25cm off in z. Rotation is also excluded -- a 6-DOF ICP slips
    into a 92deg 'flip' + 2m translation (low surface residual, WRONG pose). So we
    fix z alone; the residual profile error (~yaw) is left, not mis-corrected.

    `vert_w` (6890,) weights the z-fit by per-vertex RELIABILITY (from sigma, see
    `vertex_fit_weights`): the depth locks to the joints the model trusts most."""
    if cloud is None or len(cloud) < 200:
        LAST_REFINE.update(cloud=0 if cloud is None else len(cloud), matched=0,
                           floor=200, dz=0.0, why="cloud too small")
        return verts, joints
    # dz is a WEIGHTED MEDIAN over every match, so decimating the cloud costs
    # precision only as sqrt(n).  Measured against the full ~15k-point cloud, a
    # 6k cap moves dz by 0.48 mm median / 2.4 mm worst -- well inside the depth
    # sensor's own noise -- and saves ~2.4 ms.  Below 6k the curve turns bad
    # value: the fixed costs dominate, so 3k is only 0.3 ms faster but doubles
    # the error.  The cloud comes off an image-space grid, so a plain stride
    # stays spatially uniform.
    if len(cloud) > ICP_MAX_POINTS:
        cloud = cloud[::-(-len(cloud) // ICP_MAX_POINTS)]
    v = verts.astype(np.float64); dz_total = 0.0
    # The caller also needs this set for mesh colouring.  Refinement is a global
    # z-only translation, which cannot change self-occlusion, so compute HPR once
    # and reuse it for both registration and rendering.
    vis = visible_vertices(v) if visible_idx is None else visible_idx
    if len(vis) < 50:
        LAST_REFINE.update(cloud=len(cloud), matched=0, floor=0, dz=0.0,
                           why="mesh barely visible")
        return verts, joints
    # The correction is a pure z-TRANSLATION of the mesh, so shifting the cloud by
    # -dz instead of the mesh by +dz leaves every pairwise distance unchanged.
    # That keeps the source set fixed, so the KD-tree is built ONCE and reused:
    # the correspondences and the resulting dz are identical to rebuilding it per
    # iteration, at roughly half the cost.
    vv = v[vis]
    # Exact nearest neighbours either way; skipping the median-split build is
    # ~2x faster to construct and no slower to query at these sizes.
    tree = cKDTree(vv, balanced_tree=False, compact_nodes=False)
    query = cloud
    # Acquire wide, then refine.  The gate is there to stop the registration
    # pairing the body with whatever else got into the cloud -- but it also
    # decides whether the registration runs at all, and a mesh that has drifted
    # past it cannot muster the correspondences to be pulled back, so it stays
    # drifted.  That is a latch, not a threshold: measured over ten frames of
    # one person at 3.2 m, nine had 25-76 matches inside 0.35 m against a floor
    # of 100, broke out with dz=0, and were published sitting 24-29 cm in front
    # of their own point cloud; the tenth had 292 and landed at 2 cm.  Every one
    # of the nine had 265+ matches at 0.70 m.  So the first pass opens the gate
    # wide enough to find the body again and the rest close it back down, while
    # max_shift stays as the guard against pulling onto something that is not
    # the body.  The floor is a fraction of the cloud because the absolute one
    # silently became a third of it as the person walked away from the camera.
    floor = max(ICP_MIN_MATCHES, int(ICP_MIN_MATCH_FRAC * len(cloud)))
    matched = 0
    why = "ok"
    for gate in [max_corr * ICP_ACQUIRE_SCALE] + [max_corr] * iters:
        d, idx = tree.query(query, workers=-1)
        m = d < gate
        matched = int(m.sum())
        if matched < floor:
            why = "too few correspondences"
            break
        resid = query[m, 2] - vv[idx[m], 2]              # per-match depth residual
        w = vert_w[vis[idx[m]]] if vert_w is not None else None   # reliability weight
        dz = _weighted_median(resid, w)
        dz_total += dz
        if abs(dz) < 2e-3:
            break
        query = cloud.copy()
        query[:, 2] -= dz_total
    LAST_REFINE.update(cloud=len(cloud), matched=matched, floor=floor,
                       dz=float(dz_total), why=why)
    if abs(dz_total) > max_shift:
        LAST_REFINE["why"] = "shift over the cap"
        return verts, joints
    ov, oj = verts.copy(), joints.copy()
    ov[:, 2] += dz_total; oj[:, 2] += dz_total
    return ov.astype(np.float32), oj.astype(np.float32)


def refine_upper_body_4dof(verts, joints, cloud, vert_w, visible_idx, state=None):
    """Temporally warm-started, bounded upper-body XYZ + yaw registration."""
    if cloud is None or len(cloud) < 200 or vert_w is None:
        return verts, joints
    upper_joints = np.array([3, 6, 9, 12, 13, 14, 16, 17, 18, 19,
                             20, 21, 22, 23])
    vis = np.asarray(visible_idx)
    use = vis[np.isin(VERT_JOINT[vis], upper_joints)]
    if len(use) < 100:
        return verts, joints
    if len(use) > 700:
        use = use[::max(1, len(use) // 700)]
    source = verts[use].astype(np.float64)
    source_w = vert_w[use].astype(np.float64)
    pivot = joints[0].astype(np.float64)
    tree = cKDTree(np.asarray(cloud, np.float64))
    state = UPPER_REFINE_STATE if state is None else state
    if state["valid"]:
        prev_yaw = float(state["yaw_deg"])
        prev_shift = np.asarray(state["shift"], np.float64)
        yaw_candidates = np.clip(
            prev_yaw + np.array([-2.0, 0.0, 2.0]), -8.0, 8.0)
    else:
        prev_yaw = 0.0
        prev_shift = np.zeros(3, np.float64)
        yaw_candidates = np.linspace(-8.0, 8.0, 5)

    best = None
    for yaw_deg in np.unique(yaw_candidates):
        a = np.deg2rad(yaw_deg)
        c, s = np.cos(a), np.sin(a)
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0],
                      [-s, 0.0, c]], np.float64)
        rotated = (source - pivot) @ R.T + pivot
        shift = prev_shift.copy()
        d, nn = tree.query(rotated + shift)
        good = d < 0.20
        if good.sum() < 80:
            continue
        delta = cloud[nn[good]] - (rotated[good] + shift)
        w = source_w[good]
        step = np.array([_weighted_median(delta[:, k], w)
                         for k in range(3)])
        shift += step
        shift = np.clip(shift, [-0.08, -0.08, -0.06],
                         [0.08, 0.08, 0.06])
        d, _ = tree.query(rotated + shift)
        keep = d < np.percentile(d, 80)
        if not keep.any():
            continue
        score = np.average(d[keep], weights=source_w[keep])
        # Prefer temporal continuity when geometric scores are nearly tied.
        score += 0.04 * np.linalg.norm(shift - prev_shift)
        score += 0.0005 * abs(float(yaw_deg) - prev_yaw)
        if best is None or score < best[0]:
            best = score, R, shift, float(yaw_deg)
    if best is None:
        return verts, joints
    _, R, shift, yaw_deg = best
    state["valid"] = True
    state["yaw_deg"] = yaw_deg
    state["shift"] = shift.copy()
    out_v = (verts.astype(np.float64) - pivot) @ R.T + pivot + shift
    out_j = (joints.astype(np.float64) - pivot) @ R.T + pivot + shift
    return out_v.astype(np.float32), out_j.astype(np.float32)


def backproject(depth_m, rgb, K, stride=3):
    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride].reshape(2, -1)
    z = depth_m[ys, xs]
    m = (z > MINZ) & (z < MAXZ)
    xs, ys, z = xs[m], ys[m], z[m]
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    col = rgb[ys, xs].astype(np.float32) / 255.0
    return np.stack([x, y, z], 1), col


def largest_person(res, hw):
    """YOLO result -> (bbox_xyxy, mask bool HxW) for the biggest person, or None."""
    if res.masks is None or len(res.masks) == 0:
        return None
    boxes = res.boxes.xyxy.cpu().numpy()
    # Choose the largest detected person in ORIGINAL-image coordinates.  Do not
    # rank or reconstruct from masks.data: TensorRT may leave that tensor in its
    # square letterboxed inference resolution.
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(
        boxes[:, 3] - boxes[:, 1], 0)
    pi = int(areas.argmax())
    box = boxes[pi]

    # `Masks.xy` is Ultralytics' own inverse-letterboxed polygon in original
    # image pixels.  The previous direct cv2.resize(masks.data, W,H) stretched
    # the padded square mask to 16:9, visibly misaligning it with the person and
    # corrupting the RGB-D target cloud.
    mask = np.zeros(hw, np.uint8)
    polygons = res.masks.xy
    poly = polygons[pi] if pi < len(polygons) else None
    if poly is not None and len(poly) >= 3:
        pts = np.rint(np.asarray(poly)).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, hw[1] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, hw[0] - 1)
        cv2.fillPoly(mask, [pts], 1)
    else:
        return None
    # fillPoly wrote exactly 0/1, which is bit-identical to numpy's bool layout,
    # so reinterpret instead of copying the whole frame again.
    mask = mask.view(bool)
    return box, mask


class OneEuro:
    """Jitter filter whose strength depends on how fast the thing is moving.

    RootStabiliser rejects the impossible but passes everything else through
    untouched, which is right for a teleport and useless for a tremor: with a
    person sitting still the published root moved 2.5 cm per frame (p50) and
    its steps were negatively autocorrelated (-0.38 on y, -0.39 on z), the
    signature of noise being over-corrected rather than of anybody moving.

    A fixed low-pass would fix that and lag real motion, which is the trade
    RootStabiliser's docstring rejects.  This is the usual way out of it: the
    cutoff rises with the observed speed, so standing still is smoothed hard
    and walking is barely touched.  `beta` is how quickly it gets out of the
    way; `min_cutoff` is how still "still" looks.
    """

    def __init__(self, min_cutoff=0.8, beta=4.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x, now=None):
        if x is None:
            return None
        if self.min_cutoff <= 0.0:
            return np.asarray(x, np.float32)     # disabled
        x = np.asarray(x, np.float64)
        now = time.perf_counter() if now is None else now
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x.copy(), np.zeros_like(x), now
            return x.astype(np.float32)
        dt = max(now - self.t_prev, 1e-3)
        self.t_prev = now
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx_prev = a_d * dx + (1.0 - a_d) * self.dx_prev
        speed = float(np.linalg.norm(self.dx_prev))
        a = self._alpha(self.min_cutoff + self.beta * speed, dt)
        self.x_prev = a * x + (1.0 - a) * self.x_prev
        return self.x_prev.astype(np.float32)


class RootStabiliser:
    """Rate-limit the metric root, so outliers cannot teleport the mesh.

    The root comes from a depth median under the pelvis pixel, so it is normally
    accurate but occasionally very wrong: the mask clips onto a wall or a chair,
    the pelvis pixel lands on a depth hole, or the person turns and the visible
    skin jumps to a different part of the body.  Those frames move the root by
    far more than a person can actually move, and the mesh teleports.

    A plain EMA would also lag genuine motion, which is the thing this demo has
    to keep responsive.  So leave normal movement completely untouched and only
    act on the impossible: anything past MAX_ROOT_SPEED for the elapsed time is
    clamped back to the reachable sphere.  A jump that persists is accepted
    after ROOT_JUMP_GRACE consecutive frames -- that is real motion the tracker
    briefly lost, not an outlier, and refusing it forever would strand the mesh.
    """

    def __init__(self, max_speed=3.0, grace=3):
        self.max_speed = float(max_speed)   # m/s, sprinting is ~10 but the root is smoother
        self.grace = int(grace)
        self.prev = None
        self.prev_t = None
        self.pending = 0

    def __call__(self, root):
        now = time.perf_counter()
        if root is None:
            return None
        if self.prev is None:
            self.prev, self.prev_t, self.pending = root.copy(), now, 0
            return root
        dt = max(now - self.prev_t, 1e-3)
        step = root - self.prev
        dist = float(np.linalg.norm(step))
        budget = self.max_speed * dt
        if dist <= budget:
            self.prev, self.prev_t, self.pending = root.copy(), now, 0
            return root
        self.pending += 1
        if self.pending >= self.grace:
            # Held across several frames: treat it as real and re-seed there.
            self.prev, self.prev_t, self.pending = root.copy(), now, 0
            return root
        clamped = (self.prev + step * (budget / dist)).astype(np.float32)
        self.prev, self.prev_t = clamped.copy(), now
        return clamped

    def reset(self):
        self.prev = self.prev_t = None
        self.pending = 0


class MeshEMA:
    """Light temporal smoothing on the grounded mesh + joints (the model is per-frame,
    so it jitters). Reset on lost track so it re-seeds cleanly on reappearance."""
    def __init__(self, a=0.6):
        self.a = a; self.v = None; self.j = None

    def __call__(self, verts, joints):
        if self.v is None or self.v.shape != verts.shape:
            self.v, self.j = verts, joints
        else:
            self.v = self.a * verts + (1 - self.a) * self.v
            self.j = self.a * joints + (1 - self.a) * self.j
        return self.v, self.j

    def reset(self):
        self.v = self.j = None


# --- geometry builders run OFF the GUI thread -------------------------------
# The heavy work (compute_vertex_normals ~6ms, two-tone colours, 24 spheres) is
# done in the producer threads; the main-thread update_* callbacks then only
# remove+add these finished geometries. Building geometry is pure CPU (no GL), so
# it is safe off the main thread as long as nothing touches the scene/renderer.

# --- unified latency accounting ---------------------------------------------
# Every live demo fills these SAME keys from the SAME origin -- the RGB message
# arriving in this ROS node -- so two engines can be compared line by line.
#   capture_wait  arrival -> processing start (queueing + image decode)
#   decode        image decode alone (a subset of capture_wait, not of frame_total)
#   detect        YOLO segmentation + person selection/mask prep
#   model_core    the mesh model's own inference (GPU, or a worker's self-report)
#   model_total   what the frame loop actually pays for the model: model_core
#                 plus this process's pre/post and any IPC to a worker process
#   fit           metric root + depth-surface registration (+ hold/EMA if on)
#   geometry      building the viewer geometry
#   frame_total   processing start -> geometry ready
#   end_to_end    arrival -> geometry ready  (= capture_wait + frame_total)
UNIFIED_KEYS = ("sensor_lag", "capture_wait", "decode", "detect", "detect_gpu",
                "dispatch", "model_core", "model_total", "model_exposed",
                "fit", "fit_visible", "fit_cloud", "fit_icp",
                "geometry", "geo_visible", "geo_mesh", "geo_skeleton",
                "cloud_pts", "vis_verts", "frame_total", "end_to_end")
# rows that are NOT summed into frame_total: `sensor_lag` happens before the
# measurement origin, `decode`/`detect_gpu` are sub-items of the row above them.
UNIFIED_PRE_ORIGIN = ("sensor_lag",)
UNIFIED_SUB_ITEMS = ("decode", "detect_gpu", "model_exposed",
                     "fit_visible", "fit_cloud", "fit_icp",
                     "geo_visible", "geo_mesh", "geo_skeleton",
                     "cloud_pts", "vis_verts")


def unified_summary(rows, model_name, warmup=5):
    """Print the shared latency table. `rows` are per-frame dicts of UNIFIED_KEYS."""
    rows = rows[warmup:]
    if not rows:
        print("UNIFIED LATENCY: no warm frames recorded", flush=True)
        return
    print(f"UNIFIED LATENCY [{model_name}] median / p90 ms "
          f"(origin = RGB message arrival; {len(rows)} warm frames):", flush=True)
    print("   [+ before the origin, not in end_to_end | . sub-item of the row "
          "above | * totals]", flush=True)
    for k in UNIFIED_KEYS:
        a = np.array([r[k] for r in rows if k in r and r[k] is not None], np.float64)
        if not len(a):
            continue
        mark = ("+" if k in UNIFIED_PRE_ORIGIN else
                "." if k in UNIFIED_SUB_ITEMS else
                "*" if k in ("frame_total", "end_to_end") else " ")
        print(f" {mark}{k:<14}{np.median(a):8.2f} /{np.percentile(a, 90):8.2f}"
              f"   n={len(a)}", flush=True)
    a = np.array([r["frame_total"] for r in rows if "frame_total" in r], np.float64)
    print(f"  -> mesh rate {1000/np.median(a):.1f} Hz from frame_total", flush=True)


def _appearance(rgb, box, mask):
    """Hue-saturation histogram over the person's mask pixels, L1-normalised.

    Cropped to the box first: the masks are full-frame, so indexing one whole
    would copy the entire image for every candidate on every frame.  Hue and
    saturation only -- value is what changes when the person walks into shade.
    """
    if rgb is None or mask is None:
        return None
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(int(x2), w), min(int(y2), h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    m = np.ascontiguousarray(mask[y1:y2, x1:x2]).astype(np.uint8)
    if int(m.sum()) < 64:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[y1:y2, x1:x2]), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], m, [16, 8], [0, 180, 0, 256])
    total = float(hist.sum())
    return hist / total if total > 0 else None


def _correl(a, b):
    """Histogram agreement in [0, 1]; None when either side has no colour."""
    if a is None or b is None:
        return None
    return max(0.0, float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL)))


class PersonTrack:
    """The clicked person, carried across frames.

    Nearest-centre association is not enough: between two frames at 3 Hz a
    walking person moves further than the gap to a bystander standing still, so
    nearest-centre hands the lock to the bystander.  A match here has to agree
    with a PREDICTED position, with the box size, and with the colour the person
    was wearing, and the position gate is measured in box diagonals so it means
    the same thing at 3 Hz and at 30 Hz.

    A frame with no acceptable match does NOT release the lock.  The track
    coasts on its last velocity for PICK_COAST_S and pick_person returns None,
    so the pipeline fits nobody for those frames rather than silently
    transferring the fit to a stranger -- a wrong fit looks like a working one,
    a missing fit does not.
    """

    def __init__(self, centre, size, hist, t):
        self.c = np.asarray(centre, np.float64)
        self.v = np.zeros(2)                  # px/s, so a variable frame rate
        self.size = np.asarray(size, np.float64)   # cannot corrupt it
        self.hist = hist
        self.t = t
        self.missing_since = None

    @property
    def diag(self):
        return max(float(np.hypot(*self.size)), 1.0)

    def coast_s(self, t):
        return 0.0 if self.missing_since is None else max(0.0, t - self.missing_since)

    def predict(self, t):
        """Where the person should be now, capped so a long coast cannot fling
        the prediction off the far side of the image."""
        step = self.v * max(0.0, t - self.t)
        far = float(np.linalg.norm(step))
        limit = PICK_MAX_EXTRAP * self.diag
        if far > limit:
            step *= limit / far
        return self.c + step

    def match(self, boxes, centres, hist_of, t):
        """(index, correl) of the candidate that is this person, or (None, None).

        `hist_of(i)` is evaluated lazily -- typically only one candidate clears
        the position and size gates, so only that one costs a histogram.
        """
        pred = self.predict(t)
        # The prediction decays as it ages; widen the gate with it rather than
        # dropping a lock that only a long gap made uncertain.
        gate = PICK_POS_GATE * (1.0 + self.coast_s(t) / PICK_COAST_S)
        dn = np.linalg.norm(centres - pred, axis=1) / self.diag
        cand_diag = np.hypot(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        sn = np.abs(np.log(np.maximum(cand_diag, 1.0) / self.diag))
        ok = np.flatnonzero((dn <= gate) & (sn <= PICK_SCALE_TOL))
        best, best_cost, best_correl = None, None, None
        for i in ok:
            i = int(i)
            correl = _correl(self.hist, hist_of(i))
            # After a miss the position evidence is stale, so colour has to
            # carry the re-acquisition; without this a passer-by inherits the
            # lock exactly when the real target is hardest to see.
            if self.missing_since is not None and correl is not None \
                    and correl < PICK_REACQ_CORREL:
                continue
            cost = dn[i] / gate
            if correl is not None:
                cost += PICK_APPEAR_W * (1.0 - correl)
            if best_cost is None or cost < best_cost:
                best, best_cost, best_correl = i, cost, correl
        return best, best_correl

    def update(self, centre, size, hist, correl, t):
        centre = np.asarray(centre, np.float64)
        dt = t - self.t
        if dt > 1e-3:
            self.v = 0.5 * self.v + 0.5 * (centre - self.c) / dt
        self.c = centre
        self.size = 0.7 * self.size + 0.3 * np.asarray(size, np.float64)
        # Learn the colour only from a confident match: blending a frame where
        # something is half-covering the person teaches the model the occluder,
        # and then the occluder is what gets re-acquired.
        if hist is not None and (correl is None or correl >= PICK_REACQ_CORREL):
            self.hist = hist if self.hist is None else \
                (1.0 - PICK_APPEAR_LEARN) * self.hist + PICK_APPEAR_LEARN * hist
        self.t = t
        self.missing_since = None

    def miss(self, t):
        """Record a frame with no match. False once the coast window is spent."""
        if self.missing_since is None:
            self.missing_since = t
        return (t - self.missing_since) <= PICK_COAST_S


def pick_person(people, rgb=None):
    """Choose which candidate to fit: the tracked one, else the largest.

    `people` is [(box, mask, conf), ...] largest first.  A pending click starts
    a PersonTrack on whichever box contains it; from then on that track decides.
    Returns None while the track is coasting, so the caller fits nobody rather
    than the wrong person.  Without a track the behaviour is the old one: the
    largest box.
    """
    now = time.monotonic()

    boxes = centres = None
    if people:
        boxes = np.array([p[0] for p in people], np.float64)
        centres = np.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5,
                            (boxes[:, 1] + boxes[:, 3]) * 0.5], axis=1)
    sizes = None if boxes is None else np.stack(
        [boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], axis=1)

    cache = {}
    def hist_of(i):
        if i not in cache:
            cache[i] = _appearance(rgb, people[i][0], people[i][1])
        return cache[i]

    click = PICK_STATE.get("click")
    if click is not None and people:
        PICK_STATE["click"] = None
        cx, cy = click
        inside = ((boxes[:, 0] <= cx) & (cx <= boxes[:, 2]) &
                  (boxes[:, 1] <= cy) & (cy <= boxes[:, 3]))
        hit = np.flatnonzero(inside)
        if len(hit):
            # Smallest containing box: clicking a person standing in front of a
            # larger one should pick the person, not the one behind.
            areas = ((boxes[hit, 2] - boxes[hit, 0]) *
                     (boxes[hit, 3] - boxes[hit, 1]))
            i = int(hit[int(areas.argmin())])
            PICK_STATE["track"] = PersonTrack(centres[i], sizes[i], hist_of(i), now)
            PICK_STATE["status"] = "locked"
            return i
        PICK_STATE["track"] = None         # clicked empty space -> release
        PICK_STATE["status"] = None

    track = PICK_STATE.get("track")
    if track is None:
        return 0 if people else None       # no pick: largest, the old behaviour

    if not people:
        if not track.miss(now):
            PICK_STATE["track"] = None
            PICK_STATE["status"] = None
        else:
            PICK_STATE["status"] = "coasting"
        return None

    i, correl = track.match(boxes, centres, hist_of, now)
    if i is None:
        if not track.miss(now):
            PICK_STATE["track"] = None
            PICK_STATE["status"] = None
        else:
            PICK_STATE["status"] = "coasting"
        return None
    track.update(centres[i], sizes[i], hist_of(i), correl, now)
    PICK_STATE["status"] = "locked"
    return i


def build_seg_view_multi(rgb, people, chosen, ms=None):
    """Panel showing every detected person, with the fitted one highlighted."""
    small_w = SEG_VIEW_WIDTH
    scale = small_w / rgb.shape[1]
    PICK_STATE["scale"] = scale
    small_h = int(round(rgb.shape[0] * scale))
    out = cv2.cvtColor(cv2.resize(rgb, (small_w, small_h)), cv2.COLOR_RGB2BGR)
    if not people:
        cv2.putText(out, "NO PERSON", (16, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2, cv2.LINE_AA)
        track = PICK_STATE.get("track")
        if track is not None:
            cv2.putText(out, f"COASTING {track.coast_s(time.monotonic()):.1f}s",
                        (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 165, 255), 2, cv2.LINE_AA)
        return out
    for i, (box, mask, conf) in enumerate(people):
        picked = (i == chosen)
        m = cv2.resize(np.asarray(mask, np.uint8), (small_w, small_h),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
        colour = (0, 255, 90) if picked else (150, 150, 150)
        tint = np.zeros_like(out); tint[:] = colour
        a = 0.45 if picked else 0.22
        out[m] = ((1 - a) * out[m] + a * tint[m]).astype(np.uint8)
        x1, y1, x2, y2 = (np.rint(np.asarray(box, np.float64) * scale)).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2),
                      (0, 255, 255) if picked else (120, 120, 120),
                      2 if picked else 1)
        tag = f"{'FIT' if picked else str(i)} {conf:.2f}"
        cv2.putText(out, tag, (x1 + 3, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255) if picked else (170, 170, 170), 1, cv2.LINE_AA)
    status = PICK_STATE.get("status")
    track = PICK_STATE.get("track")
    if status == "coasting" and track is not None:
        # Draw where the track believes the person is.  Without this a coasting
        # lock is indistinguishable from a lost one: nothing is highlighted
        # either way, and the natural reaction is to click again -- which throws
        # away a track that was about to re-acquire.
        c = track.predict(time.monotonic()) * scale
        hw, hh = track.size * scale * 0.5
        p1 = (int(round(c[0] - hw)), int(round(c[1] - hh)))
        p2 = (int(round(c[0] + hw)), int(round(c[1] + hh)))
        cv2.rectangle(out, p1, p2, (0, 165, 255), 2)
        cv2.putText(out, "COASTING", (p1[0] + 3, max(p1[1] - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    if status == "coasting" and track is not None:
        msg = f"COASTING {track.coast_s(time.monotonic()):.1f}s / {PICK_COAST_S:.0f}s"
        colour = (0, 165, 255)
    elif status == "locked":
        msg, colour = "LOCKED - click elsewhere to release", (0, 255, 255)
    else:
        msg, colour = "click a person to fit them", (220, 220, 220)
    bar = f"{len(people)} person(s)   {msg}"
    if ms is not None:
        bar += f"   detect {ms:.0f} ms"
    cv2.rectangle(out, (0, 0), (small_w, 24), (0, 0, 0), -1)
    cv2.putText(out, bar, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                colour, 1, cv2.LINE_AA)
    return out


def _on_seg_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        sc = PICK_STATE.get("scale", 1.0) or 1.0
        PICK_STATE["click"] = (x / sc, y / sc)      # back to source pixels


def build_seg_view(rgb, det, ms=None):
    """BGR panel of the person mask + box that this frame's fit is using."""
    small_w = SEG_VIEW_WIDTH
    scale = small_w / rgb.shape[1]
    small_h = int(round(rgb.shape[0] * scale))
    out = cv2.cvtColor(cv2.resize(rgb, (small_w, small_h)), cv2.COLOR_RGB2BGR)
    if det is None:
        cv2.putText(out, "NO PERSON", (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2, cv2.LINE_AA)
        return out
    box, mask = det
    m = np.asarray(mask, dtype=bool)
    if m.shape != rgb.shape[:2]:
        return out
    ms_small = cv2.resize(m.astype(np.uint8), (small_w, small_h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    tint = np.zeros_like(out); tint[..., 1] = 255
    out[ms_small] = (0.55 * out[ms_small] + 0.45 * tint[ms_small]).astype(np.uint8)
    x1, y1, x2, y2 = (np.rint(np.asarray(box, np.float64) * scale)).astype(int)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
    label = "PERSON MASK" if ms is None else f"PERSON MASK  detect {ms:.0f} ms"
    cv2.rectangle(out, (0, 0), (small_w, 26), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1, cv2.LINE_AA)
    return out


# Emphasis for the person being fitted: the measured surface is the ground truth
# the predicted mesh is judged against, so it is sampled finer and kept at full
# colour while the room behind it is dimmed to context.
PERSON_CLOUD_STRIDE = 3
PERSON_CLOUD_GAIN, PERSON_CLOUD_LIFT = 1.05, 0.05
ROOM_CLOUD_GAIN, ROOM_CLOUD_LIFT = 0.55, 0.02


def build_scene_cloud(depth_m, rgb, K, person_mask=None, stride=None,
                      person_stride=None, centre=None, half=None):
    """Room cloud with the selected person's surface emphasised.

    Shared by both engines so the two demos look identical.  Without a mask this
    is the plain single-rate cloud; with one, the person is sampled at
    `person_stride` and left bright while everything else is dimmed, so rotating
    the view shows the mesh against exactly the points it has to explain.
    """
    stride = DISPLAY_STRIDE if stride is None else stride
    person_stride = PERSON_CLOUD_STRIDE if person_stride is None else person_stride
    h, w = depth_m.shape
    parts = []
    if person_mask is None or person_stride >= stride:
        rates = ((stride, None),)
    else:
        rates = ((person_stride, True), (stride, False))
    for st, want_person in rates:
        ys, xs = np.mgrid[0:h:st, 0:w:st].reshape(2, -1)
        z = depth_m[ys, xs]
        keep = (z > MINZ) & (z < MAXZ)
        if want_person is not None:
            on = person_mask[ys, xs]
            keep &= on if want_person else ~on
        if not keep.any():
            continue
        xs_k, ys_k, z_k = xs[keep], ys[keep], z[keep]
        x = (xs_k - K[0, 2]) * z_k / K[0, 0]
        y = (ys_k - K[1, 2]) * z_k / K[1, 1]
        pts = np.stack([x, y, z_k], axis=1).astype(np.float32)
        col = rgb[ys_k, xs_k].astype(np.float32) / 255.0
        if want_person is True:
            col = col * PERSON_CLOUD_GAIN + PERSON_CLOUD_LIFT
        elif want_person is False:
            col = col * ROOM_CLOUD_GAIN + ROOM_CLOUD_LIFT
        else:
            col = col * CLOUD_GAIN + CLOUD_LIFT
        parts.append((pts, np.clip(col, 0.0, 1.0)))
    if not parts:
        return o3d.geometry.PointCloud(), np.zeros((0, 3), np.float32)
    cloud = np.concatenate([p[0] for p in parts])
    col = np.concatenate([p[1] for p in parts])
    if centre is not None and half is not None and len(cloud):
        keep = np.all(np.abs(cloud - centre) < half, axis=1)
        cloud, col = cloud[keep], col[keep]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.ascontiguousarray(cloud * FLIP, np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(col, np.float64))
    return pcd, cloud


def build_cloud_geometry(cloud, col):
    # Vector3dVector copies float64 straight into its buffer but converts anything
    # else element-by-element (~35x slower); cloud/col are float32, so cast first.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(cloud * FLIP, np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(
        np.clip(col * CLOUD_GAIN + CLOUD_LIFT, 0.0, 1.0), np.float64))
    return pcd


def build_mesh_geometry(o3d_faces, verts_m, joints_m, sig_s, vis_idx, head_infov):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_m * FLIP)
    mesh.triangles = o3d_faces
    mesh.compute_vertex_normals()
    # two-tone: green = surface the depth camera sees, navy = occluded/back
    mcol = np.tile(MESH_HID, (len(verts_m), 1))
    if vis_idx is not None:
        mcol[vis_idx] = MESH_VIS
    if not head_infov and HEAD_CAP is not None:
        mcol[HEAD_CAP] = MESH_HID                 # held/neutral head = inferred, not observed
    mesh.vertex_colors = o3d.utility.Vector3dVector(mcol)
    Jf = joints_m * FLIP
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(Jf[:24])
    ls.lines = o3d.utility.Vector2iVector(np.array(SMPL_BONES, np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.tile([0.55, 0.55, 0.55], (len(SMPL_BONES), 1)))
    # joint spheres COLOURED + SIZED by the model's per-joint uncertainty
    joints = o3d.geometry.TriangleMesh()
    for j in range(24):
        sj = float(sig_s[j]) if sig_s is not None else SIG_LO
        t = np.clip((sj - SIG_LO) / (SIG_HI - SIG_LO), 0.0, 1.0)
        m = o3d.geometry.TriangleMesh(JS); m.scale(0.018 + 0.032*t, (0, 0, 0))
        m.translate(Jf[j]); m.paint_uniform_color(sig_color(sj)); joints += m
    joints.compute_vertex_normals()
    return mesh, ls, joints


def body_forward(joints_m):
    """Unit forward axis of the body in the camera frame, or None.

    SMPL indices: pelvis=0, left/right hip=1/2, neck=12, left/right shoulder=16/17.
    Combining hips and shoulders keeps this stable when one arm is articulated or
    partly occluded.
    """
    J = np.asarray(joints_m, dtype=np.float64)
    if J.shape[0] < 18 or not np.isfinite(J[:18]).all():
        return None
    right = (J[17] - J[16]) + 0.65 * (J[2] - J[1])
    up = J[12] - J[0]
    right -= up * np.dot(right, up) / max(np.dot(up, up), 1e-12)
    rn, un = np.linalg.norm(right), np.linalg.norm(up)
    if rn < 1e-5 or un < 1e-5:
        return None
    forward = np.cross(up / un, right / rn)
    fn = np.linalg.norm(forward)
    return forward / fn if fn > 1e-5 else None


def root_offset_for(forward):
    """Torso half-thickness along the camera axis, from the body's facing.

    |cos| of the angle between the forward axis and the camera's +z: 1 when the
    person faces the camera (or away), 0 in profile.
    """
    if not ORIENT_ROOT_OFFSET or forward is None:
        return ROOT_OFFSET
    frontal = abs(float(forward[2]))          # camera looks along +z
    return ROOT_OFFSET_PROFILE + (ROOT_OFFSET - ROOT_OFFSET_PROFILE) * frontal


def build_facing_arrow(joints_m):
    """Build an orange arrow from the pelvis in the SMPL body's forward direction.

    SMPL indices are pelvis=0, left/right hip=1/2, neck=12 and left/right
    shoulder=16/17.  Combining hips and shoulders makes the estimate less
    sensitive to an articulated or partly occluded arm.
    """
    J = np.asarray(joints_m, dtype=np.float64)
    forward = body_forward(joints_m)
    if forward is None:
        return None

    # create_arrow points along local +Z.  Transform only for display after the
    # direction was estimated in the camera frame.
    origin = J[0] * FLIP
    direction = forward * FLIP
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, direction)
    axis_n = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(z, direction), -1.0, 1.0))
    if axis_n < 1e-8:
        R = np.eye(3) if dot > 0 else o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([np.pi, 0.0, 0.0]))
    else:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            axis / axis_n * np.arccos(dot))
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.018, cone_radius=0.045,
        cylinder_height=FACING_ARROW_LENGTH * 0.72,
        cone_height=FACING_ARROW_LENGTH * 0.28,
        resolution=16, cylinder_split=1, cone_split=1)
    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(origin)
    arrow.paint_uniform_color(FACING_ARROW_COLOR)
    arrow.compute_vertex_normals()
    return arrow


def main():
    # model + YOLO run on fixed-size inputs, so let cuDNN autotune the conv
    # kernels once and reuse them -- a few ms off the forward pass.
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if load_model is None or run_model is None:
        raise RuntimeError(
            "no mesh model bound: set live_pipeline.load_model / .run_model "
            "before calling main() (mesh_live_o3d.py does this)")
    model, tf, faces = load_model(device)
    from ultralytics import YOLO
    yolo = YOLO(YOLO_MODEL)
    person_cls = next((i for i, n in yolo.names.items()
                       if str(n).lower() == "person"), 0)
    direct_yolo = None
    if YOLO_DIRECT and str(YOLO_MODEL).endswith(".engine"):
        from yolo_trt_runtime import YoloSegTRT
        direct_yolo = YoloSegTRT(YOLO_MODEL, conf=0.4, person_class=person_cls)
        print("YOLO: direct TensorRT runner (ultralytics wrapper bypassed)",
              flush=True)
    ema = MeshEMA(a=0.6)
    root_stab = RootStabiliser(MAX_ROOT_SPEED, ROOT_JUMP_GRACE)
    head_hold = HeadHold()
    o3d_faces = o3d.utility.Vector3iVector(faces)

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

    rclpy.init(); node = rclpy.create_node("mesh_live_o3d")
    # cloud + mesh are produced by SEPARATE threads (see cloud_worker/infer_worker):
    #   "cloud" -> fast depth back-projection at ~sensor rate (low latency)
    #   "mesh"  -> slow YOLO+model fit at ~10 Hz
    # "pelvis" is the last fitted root, used only to centre the fast cloud's crop.
    st = {"K": None, "count": 0, "cloud_geo": None,
          "compare_cloud_geo": None, "mesh_geo": None,
          "facing_arrow_geo": None,
          "compare_geo": None, "pelvis": None,
          "rgb_msg": None, "depth_msg": None, "proc_ms": 0.0, "cloud_hz": 0.0,
          "cam_set": False, "running": True, "rgb_recv_t": None,
          "rgb_sensor_lag": None, "last_infer_recv_t": None,
          "seg_view": None, "seg_cb": False, "fit_mask": None,
          "mesh_profile": None, "profile_rows": [], "mesh_update_pending": False,
          "last_vis_idx": None}
    HALF = 1.2
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=1)

    def on_info(m):
        if st["K"] is None:
            st["K"] = np.array(m.k, np.float32).reshape(3, 3)
    def on_depth(m):
        st["depth_msg"] = m
    def on_rgb(m):
        st["rgb_msg"] = m
        st["rgb_recv_t"] = time.perf_counter()
        # Sensor exposure -> arrival here (camera driver + transport).  Every
        # other bucket starts at arrival, so this is the one part of the visible
        # latency that none of them can see.  Needs the publisher on this clock.
        hs = m.header.stamp
        st["rgb_sensor_lag"] = (time.time_ns() * 1e-6
                                - (hs.sec * 1e3 + hs.nanosec * 1e-6))
    subscribe_camera(node, qos, on_rgb, on_depth, on_info)
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    def grab_frames():
        """Latest synchronised RGB + metric depth + K, or None. Shared by both
        producer threads; QoS depth=1 means this is always the freshest frame."""
        # Snapshot the references used below so a callback cannot make the image
        # and its receipt timestamp refer to different ROS messages.
        rgb_msg, depth_msg = st["rgb_msg"], st["depth_msg"]
        K, recv_t = st["K"], st["rgb_recv_t"]
        sensor_lag = st["rgb_sensor_lag"]
        if rgb_msg is None or depth_msg is None or K is None or recv_t is None:
            return None
        t0 = time.perf_counter()
        rgb = np.ascontiguousarray(decode(rgb_msg))
        depth = decode_depth(depth_msg)
        if depth.shape[:2] != rgb.shape[:2]:
            return None
        out = rgb, depth, K
        return out + (recv_t, (time.perf_counter()-t0)*1e3, sensor_lag)

    def cloud_latest():
        """FAST path: depth back-projection only (<2 ms). Runs at ~sensor rate,
        decoupled from YOLO+model, so the displayed cloud is ~1 frame old instead
        of trapped behind the ~100 ms inference loop. Cropped around the last
        fitted pelvis (or the depth median before the first fit)."""
        fr = grab_frames()
        if fr is None:
            return
        rgb, depth_m, K = fr[:3]
        centre = st["pelvis"]
        if centre is None:
            probe, _ = backproject(depth_m, rgb, K, stride=DISPLAY_STRIDE * 3)
            centre = np.median(probe, 0) if len(probe) else np.zeros(3)
        # The mask of whoever is being fitted, so their surface is emphasised.
        pcd, cloud = build_scene_cloud(depth_m, rgb, K,
                                       person_mask=st.get("fit_mask"),
                                       centre=centre, half=HALF)
        st["cloud_geo"] = pcd                                # built off the GUI thread
        if COMPARE_RUN is not None:
            # Same points, shifted sideways so the comparison model has its own
            # cloud to sit in.  Colours come straight off the built geometry.
            shifted = o3d.geometry.PointCloud(pcd)
            shifted.translate((COMPARE_OFFSET_X * FLIP[0], 0.0, 0.0))
            st["compare_cloud_geo"] = shifted
        else:
            st["compare_cloud_geo"] = None

    def infer_latest():
        """SLOW path: YOLO segmentation + model fit + cloud refine (~10 Hz).
        Publishes the mesh/skeleton and the pelvis crop-centre for the fast cloud."""
        # Edge-trigger.  Grabbing "whatever is latest" handed the model a frame of
        # uniformly random age -- half a camera period on average, ~16 ms at 30 fps
        # -- for nothing.  Waiting for an unseen frame costs throughput only when
        # the loop is faster than the camera, and buys that staleness back.
        recv_t = st["rgb_recv_t"]
        if recv_t is None or recv_t == st["last_infer_recv_t"]:
            return False
        fr = grab_frames()
        if fr is None:
            return False
        rgb, depth_m, K, recv_t, decode_ms, sensor_lag = fr
        st["last_infer_recv_t"] = recv_t
        t0 = time.perf_counter(); st["count"] += 1
        p = {"_recv_t": recv_t, "topic_wait": (t0-recv_t)*1e3,
             "decode_copy": decode_ms, "sensor_lag": sensor_lag}
        q = time.perf_counter()
        people = None
        if direct_yolo is not None and SEG_PICK:
            # Decode every person so the panel can show them all and the user
            # can click one; costs one extra mask decode per extra person.
            people = direct_yolo(rgb, all_people=True)
            chosen = pick_person(people, rgb)
            det = (people[chosen][0], people[chosen][1]) if chosen is not None else None
            PICK_STATE["people"] = people
            p["yolo"] = (time.perf_counter()-q)*1e3
            p["detection_post"] = 0.0
        elif direct_yolo is not None:
            # Straight to TensorRT: takes RGB as-is (no full-frame colour
            # conversion) and returns the same (box, mask) largest_person does.
            det = direct_yolo(rgb)
            p["yolo"] = (time.perf_counter()-q)*1e3
            p["detection_post"] = 0.0
        else:
            # Tracking is pure overhead here: `largest_person` ranks by box area
            # and nothing downstream reads a track id.
            res = yolo.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               classes=[person_cls], conf=0.4, verbose=False)[0]
            p["yolo"] = (time.perf_counter()-q)*1e3
            sp = getattr(res, "speed", None) or {}
            p["yolo_pre"] = sp.get("preprocess")
            p["yolo_gpu"] = sp.get("inference")
            p["yolo_post"] = sp.get("postprocess")
        q = time.perf_counter()
        verts_m = joints_m = sig_s = vis_idx = None; mask = None; head_infov = True
        compare_result = None
        if direct_yolo is None:
            det = largest_person(res, rgb.shape[:2])
            p["detection_post"] = (time.perf_counter()-q)*1e3
        if det is not None:
            box, mask = det
            q = time.perf_counter()
            verts, j29, pelvis_px, sigma = run_model(model, tf, rgb, box, device)
            p.update(MODEL_PROFILE)
            # Run the comparison model on exactly the same RGB frame and person
            # box.  Its result is grounded independently below, using its own
            # predicted pelvis pixel and the same metric depth image.
            if COMPARE_RUN is not None:
                cq = time.perf_counter()
                compare_result = COMPARE_RUN(rgb, box, device)
                p["compare_model"] = (time.perf_counter()-cq)*1e3
            q = time.perf_counter()
            # The regressor's joints are root-relative here, but a direction is
            # translation-invariant, so the facing axis is already valid.
            fwd = body_forward(j29)
            root_m = metric_root(pelvis_px, depth_m, K, mask,
                                 root_offset=root_offset_for(fwd))
            root_m = blend_root(root_m, MODEL_CAM_T, K)
            if MAX_ROOT_SPEED > 0:
                root_m = root_stab(root_m)
            p["root_offset"] = (root_offset_for(fwd) if fwd is not None
                                else ROOT_OFFSET)
            p["metric_root"] = (time.perf_counter()-q)*1e3
            if root_m is not None:
                q = time.perf_counter()
                st["sig"] = sigma if st.get("sig") is None else 0.5*sigma + 0.5*st["sig"]
                sig_s = st["sig"]
                vg, jg = ground(verts, j29, root_m)
                vw = vertex_fit_weights(sig_s)           # per-vertex reliability (1/sigma)
                # Vertex topology is fixed and adjacent video frames are close:
                # use the previous rendered visibility set for registration, then
                # compute the exact current set once after temporal/head updates.
                vis_for_refine = st["last_vis_idx"]
                if vis_for_refine is None:
                    vis_for_refine = visible_vertices(vg, K, mask, depth_m)
                else:
                    vis_for_refine = visible_vertices(
                        vg, K, mask, depth_m, candidate_idx=vis_for_refine)
                # sub-items of cloud_refine; they are reported, not re-summed
                r = time.perf_counter(); p["cr_visible"] = (r-q)*1e3
                observed = person_cloud(depth_m, mask, K)
                r2 = time.perf_counter(); p["cr_cloud"] = (r2-r)*1e3
                vg, jg = refine_to_cloud(vg, jg, observed, vw,
                                         visible_idx=vis_for_refine)
                p["cr_icp"] = (time.perf_counter()-r2)*1e3
                p["cloud_refine"] = (time.perf_counter()-q)*1e3
                if UPPER_BODY_4DOF_REFINE:
                    q = time.perf_counter()
                    vg, jg = refine_upper_body_4dof(
                        vg, jg, observed, vw, vis_for_refine)
                    p["upper_4dof_refine"] = (time.perf_counter()-q)*1e3
                q = time.perf_counter()
                verts_m, joints_m = (ema(vg, jg) if TEMPORAL_SMOOTHING
                                      else (vg, jg))
                # head out of FOV -> the model hallucinates a tilted head; replace it
                # with the last-seen torso-relative head pose (or neutral upright).
                if HOLD_BODY_PARTS:
                    verts_m, joints_m, head_infov = head_hold(
                        verts_m, joints_m, K, depth_m.shape, mask)
                p["temporal_head"] = (time.perf_counter()-q)*1e3
                q = time.perf_counter()
                vis_idx = visible_vertices(verts_m, K, mask, depth_m)
                st["last_vis_idx"] = vis_idx
                p["visible_surface"] = (time.perf_counter()-q)*1e3
                st["pelvis"] = joints_m[0].copy()        # crop centre for the fast cloud
        else:
            ema.reset(); root_stab.reset()
            st["sig"] = None; st["pelvis"] = None
            # A new person means a new build: drop any smoothed shape estimate.
            if callable(globals().get("reset_shape_state")):
                reset_shape_state()
            st["last_vis_idx"] = None
            UPPER_REFINE_STATE["valid"] = False
        # The comparison geometry gets the same metric grounding and depth-only
        # surface alignment, but no temporal state shared with the primary model.
        # A fixed horizontal offset prevents coincident surfaces from hiding one
        # another in the same Open3D scene.
        st["compare_geo"] = None
        if compare_result is not None and mask is not None:
            cv, cj, cpp, cs = compare_result
            cr = metric_root(cpp, depth_m, K, mask)
            if cr is not None:
                cv, cj = ground(cv, cj, cr)
                cvis = visible_vertices(cv, K, mask, depth_m)
                cobserved = person_cloud(depth_m, mask, K)
                cv, cj = refine_to_cloud(
                    cv, cj, cobserved, vertex_fit_weights(cs),
                    visible_idx=cvis)
                cvis = visible_vertices(cv, K, mask, depth_m)
                cv = cv.copy(); cj = cj.copy()
                cv[:, 0] += COMPARE_OFFSET_X
                cj[:, 0] += COMPARE_OFFSET_X
                cfaces = (o3d.utility.Vector3iVector(COMPARE_FACES)
                          if COMPARE_FACES is not None else o3d_faces)
                cgeo = build_mesh_geometry(cfaces, cv, cj, cs, cvis, True)
                # Orange makes the second model immediately distinguishable from
                # the primary green/navy mesh, including its joint markers.
                cmesh, cbones, cjoints = cgeo
                cmesh.paint_uniform_color([0.95, 0.38, 0.08])
                cjoints.paint_uniform_color([1.0, 0.72, 0.12])
                st["compare_geo"] = (cmesh, cbones, cjoints)
        # build the (heavy) mesh geometry HERE, off the GUI thread; update_mesh only
        # swaps it in. None -> no person this frame, so update_mesh clears the mesh.
        q = time.perf_counter()
        st["mesh_geo"] = (build_mesh_geometry(o3d_faces, verts_m, joints_m, sig_s,
                                              vis_idx, head_infov)
                          if verts_m is not None else None)
        st["facing_arrow_geo"] = (build_facing_arrow(joints_m)
                                  if joints_m is not None else None)
        p["geometry_build"] = (time.perf_counter()-q)*1e3
        p["worker_total"] = (time.perf_counter()-t0)*1e3
        p["topic_to_geometry"] = (time.perf_counter()-recv_t)*1e3
        # same buckets, same names as every other engine (see UNIFIED_KEYS)
        p["capture_wait"] = p["topic_wait"]
        p["decode"] = p["decode_copy"]
        p["detect"] = p["yolo"] + p["detection_post"]
        p["detect_gpu"] = p.get("yolo_gpu")
        p["model_core"] = p.get("model_gpu")
        p["model_total"] = sum(p[k] for k in ("model_pre", "model_gpu", "model_post")
                               if k in p) or None
        p["fit"] = sum(p[k] for k in ("metric_root", "cloud_refine",
                                      "upper_4dof_refine", "temporal_head",
                                      "visible_surface") if k in p) or None
        p["geometry"] = p["geometry_build"]
        p["frame_total"] = p["worker_total"]
        p["end_to_end"] = p["topic_to_geometry"]
        st["fit_mask"] = mask if det is not None else None
        if SEG_VIEW:
            st["seg_view"] = (build_seg_view_multi(rgb, people, chosen, p.get("yolo"))
                              if people is not None
                              else build_seg_view(rgb, det, p.get("yolo")))
        st["mesh_profile"] = p
        st["proc_ms"] = p["worker_total"]
        if st["count"] % 30 == 0:
            hi = f" max-sigma {sig_s[:24].max():.3f}" if sig_s is not None else ""
            print(f"  infer {1000/max(st['proc_ms'],1):.1f} fps {st['proc_ms']:.0f} ms"
                  f"  cloud {st['cloud_hz']:.0f} Hz"
                  f"  {'person' if verts_m is not None else 'no-fit'}{hi}", flush=True)
        return True

    print("waiting for first frame...", flush=True)
    while grab_frames() is None:
        time.sleep(0.02)
    cloud_latest(); infer_latest()

    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
    gui.Application.instance.initialize()
    win = gui.Application.instance.create_window(
        (("HMR2 (green/navy) vs " + str(COMPARE_NAME) + " (orange), side-by-side")
         if COMPARE_RUN is not None else
         DISPLAY_MODEL_NAME + " SMPL mesh -- green=camera-visible surface / navy=occluded; "
         "orange arrow=body facing; joint colour=uncertainty"), 1280, 860)
    sw = gui.SceneWidget(); sw.scene = rendering.Open3DScene(win.renderer)
    sw.scene.set_background([1, 1, 1, 1]); win.add_child(sw)

    m_pts = rendering.MaterialRecord(); m_pts.shader = "defaultUnlit"
    m_pts.point_size = CLOUD_POINT_SIZE
    m_mesh = rendering.MaterialRecord(); m_mesh.shader = "defaultLitTransparency"
    m_mesh.base_color = [1, 1, 1, 0.7]                         # white -> per-vertex visible/hidden colours show
    m_bone = rendering.MaterialRecord(); m_bone.shader = "unlitLine"; m_bone.line_width = 4.0
    m_joint = rendering.MaterialRecord(); m_joint.shader = "defaultLit"
    m_joint.base_color = [1, 1, 1, 1]                          # white -> per-sphere vertex colours show
    m_arrow = rendering.MaterialRecord(); m_arrow.shader = "defaultLit"
    m_arrow.base_color = [1, 1, 1, 1]
    def pump_seg_view():
        """Show the segmentation panel from the GUI thread.

        Driven off the cloud update rather than the mesh one: the mesh callback
        only fires when a fit produced geometry, so with nobody in frame the
        panel would freeze on the last person instead of showing NO PERSON.
        """
        if SEG_VIEW and st["seg_view"] is not None:
            title = "segmentation (input to the fit)"
            cv2.imshow(title, st["seg_view"])
            if SEG_PICK and not st.get("seg_cb"):
                cv2.setMouseCallback(title, _on_seg_click)
                st["seg_cb"] = True
            cv2.waitKey(1)

    def update_cloud():
        """THIN main-thread swap: geometry was already built in cloud_worker, so this
        only removes the old cloud and adds the new one (a GPU upload, no CPU build).
        Touches only the "cloud" geometry -> never blocks on the mesh; runs ~30 Hz."""
        pump_seg_view()
        pcd = st["cloud_geo"]
        if pcd is None:
            return
        s = sw.scene
        if s.has_geometry("cloud"):
            s.remove_geometry("cloud")
        s.add_geometry("cloud", pcd, m_pts)
        compare_pcd = st["compare_cloud_geo"]
        if s.has_geometry("compare_cloud"):
            s.remove_geometry("compare_cloud")
        if compare_pcd is not None:
            s.add_geometry("compare_cloud", compare_pcd, m_pts)
        if not st["cam_set"] and pcd.has_points():
            bb = s.bounding_box; sw.setup_camera(60, bb, bb.get_center())
            st["cam_set"] = True

    def update_mesh():
        """THIN main-thread swap (posted by infer_worker, ~10 Hz): the mesh/skeleton/
        sphere geometry was already built off-thread by build_mesh_geometry, so the
        heavy compute_vertex_normals no longer stalls rendering. None -> clear."""
        q = time.perf_counter()
        try:
            # Read at execution time, not post time: if several inferences finish
            # while the GUI is busy, this callback renders only the newest mesh.
            geo = st["mesh_geo"]
            p = st["mesh_profile"]
            s = sw.scene
            for n in ("mesh", "bones", "joints", "compare_mesh",
                      "compare_bones", "compare_joints", "facing_arrow"):
                if s.has_geometry(n):
                    s.remove_geometry(n)
            if geo is not None:
                mesh, ls, joints = geo
                s.add_geometry("mesh", mesh, m_mesh)
                s.add_geometry("bones", ls, m_bone)
                s.add_geometry("joints", joints, m_joint)
            facing_arrow = st["facing_arrow_geo"]
            if facing_arrow is not None:
                s.add_geometry("facing_arrow", facing_arrow, m_arrow)
            cgeo = st["compare_geo"]
            if cgeo is not None:
                cmesh, cbones, cjoints = cgeo
                s.add_geometry("compare_mesh", cmesh, m_mesh)
                s.add_geometry("compare_bones", cbones, m_bone)
                s.add_geometry("compare_joints", cjoints, m_joint)
            if p is not None:
                row = dict(p)
                row["render_submit"] = (time.perf_counter()-q)*1e3
                row["topic_to_render"] = (time.perf_counter() - p["_recv_t"])*1e3
                st["profile_rows"].append(row)
        finally:
            st["mesh_update_pending"] = False

    def infer_worker():
        while st["running"]:
            # Only post a GUI swap when a NEW mesh exists.  Under the edge trigger
            # this loop spins ~200x/s while the camera runs at 30 fps, and posting
            # unconditionally would re-add the same geometry (and re-append its
            # profile row) on every idle poll.
            fitted = infer_latest()
            if fitted and st["running"] and not st["mesh_update_pending"]:
                st["mesh_update_pending"] = True
                gui.Application.instance.post_to_main_thread(win, update_mesh)
            time.sleep(0.005)

    def cloud_worker():
        period = 1.0 / CLOUD_HZ
        last = time.time()
        while st["running"]:
            t = time.time()
            cloud_latest()
            if st["running"]:
                gui.Application.instance.post_to_main_thread(win, update_cloud)
            now = time.time()
            st["cloud_hz"] = 0.8 * st["cloud_hz"] + 0.2 / max(now - last, 1e-3)  # true loop rate
            last = now
            time.sleep(max(0.0, period - (now - t)))

    threading.Thread(target=infer_worker, daemon=True).start()
    threading.Thread(target=cloud_worker, daemon=True).start()
    print(f"live {DISPLAY_MODEL_NAME} 3D running (cloud@~{CLOUD_HZ}Hz / mesh@~10Hz); "
          "close the window to stop", flush=True)
    if RUN_SECONDS > 0:
        # Close the window on a timer so a latency run is reproducible without
        # hand timing; the shutdown path (and its PROFILE table) is unchanged.
        def _autostop():
            time.sleep(RUN_SECONDS)
            if st["running"]:
                print(f"--run-seconds {RUN_SECONDS:.0f} elapsed; closing", flush=True)
                gui.Application.instance.post_to_main_thread(win, win.close)
        threading.Thread(target=_autostop, daemon=True).start()
    gui.Application.instance.run()               # blocks until the window is closed
    st["running"] = False                        # stop workers posting to a dead window
    time.sleep(1.5 / CLOUD_HZ)                    # let in-flight worker iterations drain
    try:                                         # spin thread races shutdown; harmless
        node.destroy_node(); rclpy.shutdown()
    except Exception:
        pass
    rows = st["profile_rows"][5:]  # discard cold-start frames
    if rows:
        keys = ["sensor_lag", "topic_wait", "decode_copy",
                "yolo", "yolo_pre", "yolo_gpu", "yolo_post", "detection_post",
                "model_pre", "model_pre_crop", "model_pre_h2d",
                "model_gpu", "model_post", "metric_root",
                "cloud_refine", "cr_visible", "cr_cloud", "cr_icp",
                "upper_4dof_refine", "temporal_head",
                "visible_surface",
                "geometry_build", "worker_total", "render_submit",
                "topic_to_render"]
        unified_summary(rows, DISPLAY_MODEL_NAME, warmup=0)
        print("PROFILE median / p90 ms (warm frames):", flush=True)
        for k in keys:
            a = np.array([r[k] for r in rows if r.get(k) is not None], np.float64)
            if len(a):
                print(f"  {k:20s} {np.median(a):8.2f} / {np.percentile(a,90):8.2f}"
                      f"  n={len(a)}", flush=True)
    print(f"stopped after {st['count']} frames")


if __name__ == "__main__":
    main()
