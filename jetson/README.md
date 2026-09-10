# Jetson AGX Orin

What this pipeline needs from the machine, rather than from the checkout.  All
of it survives a reboot; none of it survives a reflash or a cleared cache, and
nothing announces its absence — the demos still start, they just run slower, or
fail three gigabytes into an export.

    sudo ./jetson/setup.sh        # install
    ./jetson/setup.sh --check     # report what is missing, change nothing

## jetson-clocks.service

`jetson_clocks` pins every clock to the ceiling of the active nvpmodel profile.
Without it the `schedutil` governor never ramps up under this workload: the GPU
waits between short CPU bursts read as idle time, so the CPU sits at ~70% of its
maximum while the pipeline is launch-bound on exactly that CPU.

Measured on the Fast SAM demo: `frame_total` 344 → 286 ms, at 48 °C, i.e. not
thermal.  The unit waits 5 s for `nvpmodel` first, because pinning to a profile
that has not been applied yet pins the wrong ceiling.

## patch_dinov3_rope.py

`Tensor.tile(2)` with a single int is rank-dependent — PyTorch left-pads the
repeats to the tensor's rank — so the ONNX exporter cannot fold it and emits an
`If` branching on that rank.  TensorRT 8.6, which is what JetPack 6.0 ships,
refuses the result:

    IIfConditionalOutputLayer inputs must have the same shape

`angles` is `[HW, D//2]` there, so `tile(1, 2)` is the identical operation with
no branch.  The edit lands in torch.hub's **download cache**, which is why this
is a script and not a patch file: it matches on the call rather than a line
number, re-running it is a no-op, and it refuses to touch a file whose upstream
has changed shape rather than patching blind.

Only needed when rebuilding the Fast SAM backbone engine.

## fold_onnx_if.py

The `If` that `patch_dinov3_rope.py` removes is not the only one — the rest come
from ops the export cannot reach.  Their conditions do not depend on the input:
they are decided by shapes that are fixed once the input size is, so each `If`
has one reachable branch.

This evaluates every condition once with onnxruntime, repoints the `If` at a
constant, and lets onnxruntime's basic optimiser delete the dead branch.

    python jetson/fold_onnx_if.py backbone_dinov3.onnx backbone_dinov3_noif.onnx

On the real backbone: 4644 nodes with 2 `If`, both conditions False, folding to
2377 nodes with 0 `If` — which TensorRT then accepts.

Two things worth knowing:

* It repoints rather than renames.  An initialiser sharing a node output's name
  breaks SSA, and the graph is then rejected outright.
* It uses onnxruntime's optimiser at `BASIC`, not onnxsim.  onnxsim v0.7.3
  segfaults on graphs of this shape, and `EXTENDED` adds operator fusions — the
  point of this file is to hand TensorRT a graph it can parse, not one another
  runtime has already rearranged.

It checks its own work: a mis-read condition usually leaves a branch that is
invalid for the input shape and the fold fails outright, and where it stays
valid the tool compares the original and folded outputs and requires a maximum
absolute difference of exactly 0.  Both nets were checked by deliberately
inverting the conditions.

## Engines are not portable

A TensorRT engine is tied to a GPU architecture **and** a TensorRT version, so
every machine builds its own; they are gitignored.  On the Orin:

    python export_tokenhmr_trt.py --conv1d-as-conv2d
    python third_party/Fast-SAM-3D-Body/convert_backbone_tensorrt.py --all

`--conv1d-as-conv2d` rewrites each `Conv1d` as an equivalent height-1 `Conv2d`.
TensorRT 8.6's `Conv1DOptimization` pass asserts on this graph
(`out.size() == mNode.outputs.size()`); the rewrite is numerically identical
(verified at 0.0 max difference on all four outputs) and sidesteps the pass.

Build the Fast SAM backbone with fp16 **bindings**, not just `--fp16`:

    trtexec --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw ...

`--fp16` alone leaves the bindings FLOAT while the wrapper feeds half — which
does not error, it produces all-NaN vertices.

## Still outside version control

* The Orbbec SDK 2.7.6 upgrade in `ia_bot_ws` (the vendored 2.7.2 arm64 build
  crashes with `double free or corruption` on any depth configuration).
* The camera driver does not start on boot; it needs
  `ros2 launch orbbec_camera gemini_330_series.launch.py depth_registration:=true`
  after every reboot.  `depth_registration` is not optional: the pipeline
  indexes the depth image with colour pixels and rejects a size mismatch.
