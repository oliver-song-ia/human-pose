# Occluded-Joint Recovery + Realtime Demo (2026-07-24)

Recover self-occluded joints (arms) from an RGB-D stream by holding their
pre-occlusion pose relative to the torso and only deviating under hard
constraints. Layered on the **open-source MediaPipe** pose model (not the
project's own weak checkpoint) + Gemini 335L depth, running live at ~15-25 fps.

This directory is a snapshot pulled out of scratchpad so the work survives a
crash. Paths inside the `.py` files are absolute to this folder.

## Files

**Core solver** (`ntu_temporal.py`, `ntu_occlusion.py`)
- `solve_occluded(sh, L1, L2, anchor_x, Jb, side, depth, K, torso, ...)`:
  hold the anchor pose; deviate only for self-overlap / observation-conflict /
  torso non-penetration. Local Nelder-Mead search from the anchor.
- `observation_conflict(pts, depth, K)`: penalise a hidden-limb surface sample
  that would be VISIBLE (off-frame, on background, or in front of the observed
  surface) -- the "hold unless the mesh contradicts the point cloud" test.
- `arm_self_collision` / `resolve_self_collision`: capsule-capsule
  (`seg_seg_dist`, Ericson), joint-sharing links exempt; torso is the flat slab.
- `torso_frame` / `mirror_shoulder`: torso pose + occluded-shoulder anchor,
  both profile-view robust.

**MediaPipe RGB-D pipeline** (`mediapipe_rgbd.py`)
- `make_landmarker` (Tasks API, `models/pose_landmarker_full.task`),
  `mediapipe_15` (33 landmarks -> our 15, detect on a 384px downscale),
  `lift_to_3d` (depth backproject + per-joint tracked flag), `build_torso_slab`.

**Live demos**
- `mediapipe_live_o3d.py` -- MAIN demo. open3d gui window: RGB cloud + semi-
  transparent capsule body (orange=visible, red=occluded arm) + green/yellow
  joint spheres, latency stages printed. One-Euro joint de-jitter, torso-
  relative frozen hold. Run: `python mediapipe_live_o3d.py` (start the camera
  first; close window to stop). `ArmRecoverer` lives in `mediapipe_live_ros.py`.
- `mediapipe_live_ros.py` -- 2D-overlay live version + `ArmRecoverer` class.
- `mediapipe_recover.py` -- single-frame offline validation on lab data.
- `ablation_live_o3d.py` -- same 3D demo, pose source = the project's own
  RGBD-Pose15 NTU-only ablation model (`checkpoints/ntu_only_T2_best.pt`);
  occlusion judged by a depth-surface test (its vis head is unsupervised).
- `hybrik_live_o3d.py` -- **HybrIK (SMPL body-mesh recovery)** demo, the
  parametric-model comparison point. YOLO box -> HybrIK -> full 6890-vert SMPL
  mesh + 29 joints (metric scale, monocular). Depth grounds it: HybrIK's mesh is
  root-relative, so its pelvis is placed at the sensor distance (median mask
  depth + torso half-thickness, `ROOT_OFFSET`) back-projected with the real K --
  HybrIK owns shape/pose (self-occluded limbs filled by the SMPL prior), depth
  owns WHERE. `refine_to_cloud` then does a DEPTH-ONLY (z) snap of the mesh onto
  the observed cloud, registering ONLY the camera-VISIBLE mesh surface: the depth
  camera sees just the front, so `visible_vertices` (hidden-point removal from the
  origin) drops back-facing + self-occluded verts first -- matching the full closed
  mesh biases the fit and under-reports error. ONLY z is corrected: lateral (x,y)
  is already accurate from HybrIK's 2D grounding (pelvis pixel), and a 3-DOF
  nearest-neighbour shift SLID spuriously on the smooth torso -> a ~2cm frontal
  left-right bias; rotation is excluded too (a 6-DOF ICP slips into a 92deg 'flip'
  + 2m translation, low residual / WRONG pose). Depth is the only placement that
  genuinely drifts: frontal-calibrated grounding sits up to ~25cm off in z in
  PROFILE (measured visible-surface median: frontal 56->43mm no lateral move;
  profiles 81->62, 92->50; worst profile 306->227mm). The residual profile error
  (~yaw) is left, not mis-corrected. The z-fit is a WEIGHTED median where each
  vertex's weight = HybrIK's own per-joint RELIABILITY, from its `pred_sigma`
  (`vertex_fit_weights`: smaller sigma -> higher weight, ramp SIG_W_LO/HI
  0.004/0.030, floor 0.05; per-vertex via the SMPL skinning-argmax joint). The
  depth locks to whatever the model currently trusts, so "upper-body priority"
  EMERGES from reliability -- at close/seated range legs occlude/truncate -> high
  sigma -> auto down-weighted (measured: frontal most-reliable = pelvis/hips/spine
  sigma ~0.002, least = knees/ankles/feet ~0.006; profile least = the occluded far
  arm sigma ~0.1). The HEAD is additionally CAPPED low (`HEAD_W` 0.1) regardless of
  its sigma: SMPL is a bare skull but the cloud head is skull + HAIR (bigger/
  offset), so it must not pull the fit even when the model is confident about it. Light temporal EMA (`MeshEMA`).
  When the HEAD leaves the FOV (`head_in_fov`: head joint projects outside the
  image OR onto no person-mask), HybrIK hallucinates a tilted head from the prior;
  `HeadHold` replaces it by RE-EMBEDDING a known head-cap into the current torso
  frame (NOT rotating the bad head -- a large correction is degenerate and folds/
  inverts it): the head-cap as LAST SEEN (stored in torso coords, re-carried), or
  the canonical REST head-cap if never seen (neutral upright). Rigid re-embed, so
  the head shape is preserved and can never fold. CRUCIAL: this is all in MESH
  space -- the neck/torso come from `J_REGRESSOR @ verts` (mesh-consistent), NOT
  `pred_xyz_jts_29`, which is a differently-scaled joint prediction whose neck sits
  ~25cm off the mesh's real neck (using it placed the neutral head sunk in the
  chest). The held head is drawn navy (inferred); in-FOV heads are trusted. Renders the
  translucent SMPL mesh TWO-TONE -- **green = camera-VISIBLE surface (the HPR
  `visible_vertices` front, what registers to the cloud), navy = occluded/back
  (inferred by the SMPL prior only)**, per-vertex colours (`MESH_VIS`/`MESH_HID`,
  material base white so they show) -- + SMPL skeleton over the RGB cloud,
  **joint spheres coloured + sized by HybrIK's
  own per-joint uncertainty** (`pred_sigma`, RLE; blue=confident → red=uncertain,
  `sig_color`/`SIG_LO,SIG_HI`). That sigma is a weak occlusion proxy — it rises on
  occluded/distal/OOD joints (measured 0.003 core → 0.03 folded limb, live up to
  0.08) but conflates occlusion with pose-difficulty; not a true occlusion flag.
  Runs live ~13 fps (HRNet-W48, flip_test off).
  Validated offline on single-person NTU frames: 37-114mm cloud->mesh surface
  overlap, 41-170mm pelvis-vs-GT. Code + weights in `third_party/HybrIK`
  (`model_files/` has the bundled SMPL neutral pkl; `pretrained_models/
  hybrik_hrnet.pth`, HRNet-W48 w/ 3DPW+3DHP). NOT installed via setup.py (its
  `opencv==4.1.2.30` pin would break the env) -- imported via sys.path + chdir.
- `dual_live_o3d.py` -- side-by-side in one window: LEFT MediaPipe / RIGHT
  RGBD-Pose15 ablation, same camera, x-offset, rotate together.
- `dual_hybrik_mediapipe_o3d.py` -- side-by-side LEFT MediaPipe (sparse joints +
  explicit occluded-arm recovery, blue capsules) / RIGHT HybrIK (full SMPL mesh,
  teal, hidden limbs implied by the body prior). The paradigm contrast. ~7-8 fps
  (MediaPipe + YOLO + HybrIK all per frame). 3D labels over each panel.

**NTU offline tools** `find_ntu_seq/find_ntu_side/find_occluded/find_visible_arm`
(scan for occluded frames), `ntu_player.py` (open3d GUI play/pause player),
`ntu_animate.py` (GIF), `viz_solve.py` (elbow-solve viz), `profile_solve.py` /
`time_solve.py` (timing), `ntu_smpl_body.py` / `ntu_smpl_recover.py` (the
abandoned SMPL-body attempt, kept for reference).

## The recovery algorithm (final, user-directed)

1. **Hold the pre-occlusion pose, RELATIVE TO THE TORSO.** Arm directions are
   stored in the torso frame (`R.T @ world_dir`) and re-carried each frame by the
   current torso pose (`R @ dir_local`), so the occluded arm follows torso
   translation AND rotation. (Storing them in world space was a bug -- the
   shoulder rotated but the arm stayed pinned, detaching it.)
2. **Deviate only when forced**: self-overlap (never allowed) or the held mesh
   contradicting the observed depth. The current live build FREEZES the pose
   entirely while occluded (per user request) -- no solve, just torso-carry --
   which is jump-proof by construction; the constrained-solve path
   (`solve_occluded`) is available for when re-fitting to observations is wanted.
3. **Bone lengths are a locked biometric**: slow EMA over sane measurements
   (only when the shoulder is also tracked; clamp 0.12-0.45 m). VISIBLE arms are
   ALSO rendered length-normalised to this estimate, so nothing stretches.

## Today's stability fixes (each was a real, measured bug)

- **Handoff jump (visible -> occluded).** Measured with a debug print: the seed
  anchor was correct to ~10 mm, but `solve_occluded` (conflict/self-collision)
  threw the arm 100-300 mm. Fix: onset frame trusts the seed (skip solve);
  later frames rate-limit the correction to <=4 cm/frame. Then made the live
  build freeze entirely, removing the class of bug.
- **Occluded arm becomes GIANT.** `arm_visible` didn't check the shoulder; when
  SH was untracked, `J[SH]=(0,0,0)` made `L1=|elbow-0|~2 m`. Fix: measure bone
  length only when SH tracked + clamp to a human range.
- **Visible arm length jitter.** MediaPipe per-frame length noise -> render
  visible arm with the EMA-locked lengths, not the raw joints.
- **Joint jitter from mask-edge noise.** Added a **One-Euro filter** per 3-D
  joint (strong smoothing when slow, light when fast; the standard hand/pose
  de-jitter). Reset a joint's filter when it goes untracked.
- **Side-view breakdown.** (a) mark occluded joints in 3D only where their
  MediaPipe pixel has valid depth (don't guess onto a plane); (b) `torso_frame`
  guards the degenerate lateral axis (shoulder near the spine line) with a
  camera-derived fallback.
- **Latency 200-700 ms -> ~50-90 ms.** It was FRAME QUEUE BACKLOG (processing
  slower than 30 fps). Fix: callbacks stash only the newest raw msg; all heavy
  work runs in the render loop on the freshest frame; QoS BEST_EFFORT depth=1.
- **Recovery solve 94.5 -> 17.7 ms/call (5.3x)** via cProfile: pure-float
  `seg_seg_dist` (numpy per-op overhead on 3-vecs dominated), hand-written
  `_cross3` (np.cross moveaxis is huge on 3-vecs), precomputed capsule ring
  (`fast_capsule`), Nelder-Mead maxiter 90->45.
- **open3d stutter.** Was per-frame mesh add/remove in the legacy Visualizer.
  Options used: point-cloud-only in-place updates (legacy), OR the gui
  `SceneWidget` + `defaultLitTransparency` for true semi-transparent capsules
  (current main demo).

## Known limits (honest)

- Fully-occluded pose is fundamentally under-constrained: hold + non-penetration
  only NARROW it; the true arm pose needs a data-driven motion prior or a second
  camera. The freeze build makes no claim beyond "plausible, stable, attached".
- MediaPipe drives everything; its 2D on the far side of a profile view is
  unreliable, so those joints are shown as depth-valid markers only.
- Timing is on an i9 desktop / RTX 4070; Jetson Orin will be slower and needs
  the Gauss-Newton solver instead of Nelder-Mead for a tight budget.

See also memory: `mediapipe-rgbd-occlusion-demo.md`,
`occluded-joint-optimization.md`.
