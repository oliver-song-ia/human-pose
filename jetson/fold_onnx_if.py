#!/usr/bin/env python3
"""Remove `If` nodes from an ONNX graph whose conditions are actually constant.

TensorRT 8.6 (JetPack 6.0) rejects the DINOv3 backbone with

    IIfConditionalOutputLayer inputs must have the same shape

The `If` nodes come from rank-dependent PyTorch ops that the exporter could not
fold.  Their conditions do not depend on the input at all -- they are decided by
shapes that are fixed once the input size is -- so each one has exactly one
reachable branch.  This evaluates every `If` condition once with onnxruntime,
writes the answers back into the graph as constant initialisers, and lets a
constant folder delete the dead branch.

Measured on backbone_dinov3.onnx: 4644 nodes with 2 `If`, both conditions
False, folding to 2377 nodes with 0 `If` -- which TensorRT then accepts.

This is a graph rewrite, so it verifies rather than assumes, twice.  Pinning a
condition to the wrong value usually makes the surviving branch invalid for the
input shape, and the fold fails outright; where it stays valid, the tool runs
the original and the folded graph on the same input and requires a maximum
absolute difference of exactly 0.  Both nets have been checked by deliberately
inverting the conditions.

    python jetson/fold_onnx_if.py backbone_dinov3.onnx backbone_dinov3_noif.onnx

Prefer fixing the export where you can -- see patch_dinov3_rope.py, which
removes one `If` at the source.  This is for the ones you cannot reach.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def if_nodes(model):
    return [n for n in model.graph.node if n.op_type == "If"]


def random_inputs(model, rng):
    """One random tensor per graph input, from its declared type and shape."""
    TYPES = {onnx.TensorProto.FLOAT: np.float32,
             onnx.TensorProto.FLOAT16: np.float16,
             onnx.TensorProto.DOUBLE: np.float64,
             onnx.TensorProto.INT64: np.int64,
             onnx.TensorProto.INT32: np.int32,
             onnx.TensorProto.BOOL: np.bool_}
    initialised = {i.name for i in model.graph.initializer}
    feeds = {}
    for vi in model.graph.input:
        if vi.name in initialised:
            continue
        tt = vi.type.tensor_type
        shape = [d.dim_value if d.dim_value > 0 else 1 for d in tt.shape.dim]
        dtype = TYPES.get(tt.elem_type, np.float32)
        if dtype in (np.int64, np.int32):
            feeds[vi.name] = rng.integers(0, 2, size=shape).astype(dtype)
        elif dtype is np.bool_:
            feeds[vi.name] = rng.integers(0, 2, size=shape).astype(np.bool_)
        else:
            feeds[vi.name] = rng.standard_normal(shape).astype(dtype)
    return feeds


def evaluate(model_path, names, feeds):
    """Values of `names` for one forward pass, by exposing them as outputs."""
    import onnxruntime as ort
    model = onnx.load(str(model_path))
    known = {vi.name for vi in model.graph.output}
    for name in names:
        if name not in known:
            model.graph.output.append(helper.ValueInfoProto(name=name))
    tmp = Path(str(model_path) + ".probe.onnx")
    # save_as_external_data keeps this under the 2 GiB protobuf limit; the
    # weights stay in the original file's directory and are not duplicated.
    onnx.save(model, str(tmp), save_as_external_data=True,
              location=tmp.name + ".data", all_tensors_to_one_file=True)
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(str(tmp), so,
                                    providers=["CPUExecutionProvider"])
        outs = sess.run(list(names), feeds)
        return dict(zip(names, outs))
    finally:
        tmp.unlink(missing_ok=True)
        Path(str(tmp) + ".data").unlink(missing_ok=True)
        Path(tmp.parent / (tmp.name + ".data")).unlink(missing_ok=True)


def fold(src, dst, verify=True, seed=0):
    model = onnx.load(str(src))
    ifs = if_nodes(model)
    print(f"{src}: {len(model.graph.node)} nodes, {len(ifs)} If")
    if not ifs:
        print("nothing to fold")
        return 0

    rng = np.random.default_rng(seed)
    feeds = random_inputs(model, rng)
    conds = [n.input[0] for n in ifs]
    values = evaluate(src, conds, feeds)
    for name, v in values.items():
        print(f"  condition {name} = {bool(np.asarray(v).reshape(-1)[0])}")

    # Repoint each If at a NEW constant rather than renaming the condition in
    # place: an initialiser sharing a node output's name breaks SSA and onnxsim
    # refuses the graph outright.  The original producer is left alone and dead
    # -code elimination collects it along with the branch it fed.
    for node in ifs:
        cond = node.input[0]
        const = f"{cond}__folded"
        b = bool(np.asarray(values[cond]).reshape(-1)[0])
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(b, np.bool_), const))
        node.input[0] = const

    # onnxruntime's own basic optimiser does the folding: it is already a
    # dependency here, and onnxsim v0.7.3 segfaults on graphs of this shape.
    # BASIC and not EXTENDED deliberately -- extended adds operator fusions,
    # and the point of this file is to hand TensorRT a graph it can parse, not
    # one another runtime has already rearranged.
    import onnxruntime as ort
    pinned = Path(str(dst) + ".pinned.onnx")
    onnx.save(model, str(pinned), save_as_external_data=True,
              location=pinned.name + ".data", all_tensors_to_one_file=True)
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath = str(dst)
        # Without this a graph over 2 GiB cannot be written back at all.
        so.add_session_config_entry(
            "session.optimized_model_external_initializers_file_name",
            Path(dst).name + ".weights")
        so.add_session_config_entry(
            "session.optimized_model_external_initializers_min_size_in_bytes",
            "1024")
        try:
            ort.InferenceSession(str(pinned), so,
                                 providers=["CPUExecutionProvider"])
        except Exception as exc:
            # A branch that is dead for this input is often invalid for it too,
            # so a mis-read condition usually surfaces here, in shape inference,
            # rather than as a wrong number later.  Say so plainly.
            print(f"folding the pinned graph failed: {exc}\n"
                  f"  The condition values above are what the branches were "
                  f"pinned to; if they look wrong, the probe pass read them "
                  f"from a different input shape than the graph expects.",
                  file=sys.stderr)
            return 2
    finally:
        pinned.unlink(missing_ok=True)
        Path(str(pinned) + ".data").unlink(missing_ok=True)

    folded = onnx.load(str(dst), load_external_data=False)
    left = len(if_nodes(folded))
    print(f"folded to {len(folded.graph.node)} nodes, {left} If")
    if left:
        print("If nodes remain; TensorRT 8.6 will still refuse this graph",
              file=sys.stderr)
        return 2
    print(f"wrote {dst}")

    if verify:
        outs = [o.name for o in model.graph.output]
        a = evaluate(src, outs, feeds)
        b = evaluate(dst, outs, feeds)
        worst = 0.0
        for name in outs:
            d = float(np.max(np.abs(a[name].astype(np.float64)
                                    - b[name].astype(np.float64))))
            worst = max(worst, d)
            print(f"  {name}: max |diff| = {d:g}")
        if worst != 0.0:
            print(f"VERIFY FAILED: folding changed the result by {worst:g}",
                  file=sys.stderr)
            return 3
        print("verified: identical outputs")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the before/after comparison (it needs two more "
                         "forward passes and the memory for both graphs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    return fold(args.src, args.dst, verify=not args.no_verify, seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
