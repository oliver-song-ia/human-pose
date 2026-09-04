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
import open3d as o3d
from scipy.spatial import cKDTree

HP = "/media/oliver/9a72b131-ff4a-4fc1-a6d5-53ef9c8524e1/code/human-pose"
YOLO_PT = "/home/oliver/Documents/semantic_perception/yolo26m-seg-custom_20260903.pt"
YOLO_ENGINE = "/home/oliver/Documents/semantic_perception/yolo26m-seg-custom_20260903_rtx4070.engine"
YOLO_MODEL = YOLO_ENGINE if Path(YOLO_ENGINE).exists() else YOLO_PT

FLIP_TEST = False        # 2nd flipped forward pass: +accuracy, ~2x slower. off = live
TEMPORAL_SMOOTHING = True
HOLD_BODY_PARTS = True
FAST_VISIBILITY = False
ROOT_OFFSET = 0.10       # pelvis sits ~10cm behind the front torso skin the depth sees
ICP_MAX_POINTS = 6000    # observed-cloud cap for the depth z-fit (see refine_to_cloud)
MINZ, MAXZ = 0.3, 5.0
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


def metric_root(pelvis_px, depth_m, K, mask):
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
    zr = z + ROOT_OFFSET
    xr = (u - K[0, 2]) * zr / K[0, 0]
    yr = (v - K[1, 2]) * zr / K[1, 1]
    return np.array([xr, yr, zr], np.float32)


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
    return w


def _weighted_median(x, w):
    if w is None or w.sum() <= 0:
        return float(np.median(x))
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[np.searchsorted(c, 0.5 * c[-1])])


def person_cloud(depth_m, mask, K, stride=6):
    """Observed person-surface points (metric camera frame) for the ICP refine."""
    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride].reshape(2, -1)
    z = depth_m[ys, xs]
    m = (z > MINZ) & (z < MAXZ) & mask[ys, xs]
    xs, ys, z = xs[m], ys[m], z[m]
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
        return ids[zz <= nearest[bins] + 0.035]
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
    for _ in range(iters):
        d, idx = tree.query(query, workers=-1)
        m = d < max_corr
        if m.sum() < 100:
            break
        resid = query[m, 2] - vv[idx[m], 2]              # per-match depth residual
        w = vert_w[vis[idx[m]]] if vert_w is not None else None   # reliability weight
        dz = _weighted_median(resid, w)
        dz_total += dz
        if abs(dz) < 2e-3:
            break
        query = cloud.copy()
        query[:, 2] -= dz_total
    if abs(dz_total) > max_shift:
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


def build_facing_arrow(joints_m):
    """Build an orange arrow from the pelvis in the SMPL body's forward direction.

    SMPL indices are pelvis=0, left/right hip=1/2, neck=12 and left/right
    shoulder=16/17.  Combining hips and shoulders makes the estimate less
    sensitive to an articulated or partly occluded arm.
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
    right /= rn; up /= un
    forward = np.cross(up, right)
    fn = np.linalg.norm(forward)
    if fn < 1e-5:
        return None
    forward /= fn

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
    ema = MeshEMA(a=0.6)
    head_hold = HeadHold()
    o3d_faces = o3d.utility.Vector3iVector(faces)

    import rclpy
    from sensor_msgs.msg import CameraInfo, Image
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

    def decode(msg):
        rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        if msg.encoding in ("rgb8", "bgr8"):
            im = rows[:, :msg.width*3].reshape(msg.height, msg.width, 3)
            return im if msg.encoding == "rgb8" else im[..., ::-1]
        if msg.encoding == "16UC1":
            return rows[:, :msg.width*2].copy().view(np.uint16).reshape(msg.height, msg.width)
        raise ValueError(msg.encoding)

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
          "mesh_profile": None, "profile_rows": [], "mesh_update_pending": False,
          "last_vis_idx": None}
    HALF = 1.2
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST, depth=1)

    def on_info(m):
        if st["K"] is None:
            st["K"] = np.array(m.k, np.float32).reshape(3, 3)
    node.create_subscription(CameraInfo, "/camera/color/camera_info", on_info, qos)
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
    node.create_subscription(Image, "/camera/depth/image_raw", on_depth, qos)
    node.create_subscription(Image, "/camera/color/image_raw", on_rgb, qos)
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
        depth = decode(depth_msg)
        if depth.shape[:2] != rgb.shape[:2]:
            return None
        out = rgb, depth.astype(np.float32) * 0.001, K
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
        cloud, col = backproject(depth_m, rgb, K, stride=DISPLAY_STRIDE)
        centre = st["pelvis"]
        if centre is None:
            centre = np.median(cloud, 0) if len(cloud) else np.zeros(3)
        if len(cloud):
            keep = np.all(np.abs(cloud - centre) < HALF, axis=1)
            cloud, col = cloud[keep], col[keep]
        st["cloud_geo"] = build_cloud_geometry(cloud, col)   # built here, off the GUI thread
        if COMPARE_RUN is not None:
            compare_cloud = cloud.copy()
            compare_cloud[:, 0] += COMPARE_OFFSET_X
            st["compare_cloud_geo"] = build_cloud_geometry(compare_cloud, col)
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
        # Tracking is pure overhead here: `largest_person` ranks by box area and
        # nothing downstream reads a track id.
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
            root_m = metric_root(pelvis_px, depth_m, K, mask)
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
            ema.reset(); st["sig"] = None; st["pelvis"] = None
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
    def update_cloud():
        """THIN main-thread swap: geometry was already built in cloud_worker, so this
        only removes the old cloud and adds the new one (a GPU upload, no CPU build).
        Touches only the "cloud" geometry -> never blocks on the mesh; runs ~30 Hz."""
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
