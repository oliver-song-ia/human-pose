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

Model weights, TensorRT engines and upstream checkpoints are not tracked here
(see `.gitignore`); build them locally.
