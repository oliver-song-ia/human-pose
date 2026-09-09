# human-pose

Live RGB-D human mesh recovery on an Orbbec Gemini 335L, visualised in Open3D.

`occlusion_recovery/` holds two interchangeable real-time demos that share one
pipeline (YOLO-seg person mask -> mesh model -> depth grounding -> camera-visible
surface refine -> Open3D):

- `mesh_live_o3d.py --engine tokenhmr`  TokenHMR, SMPL topology, TensorRT FP16
- `mesh_live_o3d.py --engine fastsam3d` Fast SAM 3D Body, MHR topology, run in a
  separate-environment ZMQ worker

`live_pipeline.py` is the shared pipeline; a launcher binds `load_model` /
`run_model` and calls `main()`. See `occlusion_recovery/README.md` for details
and `LIVE_MESH_OPTIMIZATION_NOTES.md` for the latency work.

## Getting the dependencies

Fast SAM 3D Body is a submodule -- it carries our own ZMQ mesh worker and the
fixes needed to run on a Jetson, so it is not the upstream tree:

    git clone --recurse-submodules git@github.com:oliver-song-ia/human-pose.git
    # or, in an existing clone:
    git submodule update --init --recursive

TokenHMR is unmodified upstream and is not tracked, because its checkpoints run
to several GB:

    git clone https://github.com/saidwivedi/TokenHMR.git third_party/TokenHMR
    # then follow its README for data/checkpoints and data/body_models

Model weights, TensorRT engines and upstream checkpoints are not tracked here
(see `.gitignore`).  Engines in particular have to be built on the machine that
will run them -- a TensorRT engine is tied to both the GPU architecture and the
TensorRT version:

    python occlusion_recovery/export_tokenhmr_trt.py            # add --conv1d-as-conv2d on TensorRT 8.x
    python third_party/Fast-SAM-3D-Body/convert_backbone_tensorrt.py --all
