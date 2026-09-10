# human-pose

Live RGB-D human mesh recovery on an Orbbec Gemini 335L, visualised in Open3D.

Two interchangeable real-time demos share one pipeline (YOLO-seg person mask -> mesh model -> depth grounding -> camera-visible
surface refine -> Open3D):

- `mesh_live_o3d.py --engine tokenhmr`  TokenHMR, SMPL topology, TensorRT FP16
- `mesh_live_o3d.py --engine fastsam3d` Fast SAM 3D Body, MHR topology, run in a
  separate-environment ZMQ worker

`live_pipeline.py` is the shared pipeline; a launcher binds `load_model` /
`run_model` and calls `main()`. See `PIPELINE.md` for details.

## Getting the dependencies

Fast SAM 3D Body lives in `third_party/Fast-SAM-3D-Body`, tracked here in full
rather than fetched: it is not the upstream tree, because it carries our own ZMQ
mesh worker and the fixes needed to run on a Jetson.  A plain clone gets it:

    git clone git@github.com:oliver-song-ia/human-pose.git

(Its multi-GB checkpoints stay out via that directory's own `.gitignore`; fetch
them from the upstream project.)

TokenHMR is unmodified upstream and is not tracked, because its checkpoints run
to several GB:

    git clone https://github.com/saidwivedi/TokenHMR.git third_party/TokenHMR
    # then follow its README for data/checkpoints and data/body_models

Model weights, TensorRT engines and upstream checkpoints are not tracked here
(see `.gitignore`).  Engines in particular have to be built on the machine that
will run them -- a TensorRT engine is tied to both the GPU architecture and the
TensorRT version:

    python export_tokenhmr_trt.py            # add --conv1d-as-conv2d on TensorRT 8.x
    python third_party/Fast-SAM-3D-Body/convert_backbone_tensorrt.py --all

## Running

Both demos take the same options and share one launcher; `--engine` defaults to
`tokenhmr`:

    python mesh_live_o3d.py                      # TokenHMR
    python mesh_live_o3d.py --engine fastsam3d   # Fast SAM 3D Body

The engines are built per machine and the YOLO weights live outside the tree, so
point at them first (put these in your shell rc):

    export HUMAN_POSE_YOLO=/path/to/yolo-seg.engine
    export HUMAN_POSE_YOLO_PT=/path/to/yolo-seg.pt      # fallback if no engine

### Choosing the camera topics

The defaults are the Orbbec Gemini driver's (`/camera/color/image_raw`,
`/camera/depth/image_raw`, `/camera/color/camera_info`).  Anything else -- a bag,
a RealSense -- is selected per run:

    python mesh_live_o3d.py \
      --color-topic /camera3/color/image_raw/compressed \
      --depth-topic /camera3/aligned_depth_to_color/image_raw \
      --info-topic  /camera3/color/camera_info

`--camera-ns /camera3` is the shorthand when the driver uses the usual layout.
A colour topic ending in `/compressed` is subscribed as `CompressedImage` and
JPEG-decoded (costs ~8-10 ms a frame, so prefer the raw topic when there is one).
The same three can be set with `HUMAN_POSE_COLOR_TOPIC` / `_DEPTH_TOPIC` /
`_INFO_TOPIC`, which is the tidier way to keep a bag and a live camera side by
side.

Depth must be registered to colour -- the pipeline indexes the depth image with
colour pixels and refuses a size mismatch.  On the Orbbec driver that is
`depth_registration:=true`; on a RealSense it is the
`aligned_depth_to_color` stream.

### Picking a person

    python mesh_live_o3d.py --seg-pick

opens a segmentation panel showing every detected person.  Click one to fit that
person instead of the largest; click empty space to release.  The selection is
held by a tracker (see `PersonTrack`), not re-decided each frame: it predicts
through fast motion at a low frame rate, refuses a bystander who is merely
closer, and coasts for `PICK_COAST_S` through missed detections -- fitting
nobody, and showing an orange predicted box, rather than silently moving the fit
onto a stranger.

### Useful flags

    --run-seconds N        exit after N seconds and print the latency table
    --no-gui               Fast SAM only: no window, for a clean benchmark
    --cloud-hz / --render-hz   display rates; lower frees the mesh thread
    --seg-view             the panel without click-to-pick
