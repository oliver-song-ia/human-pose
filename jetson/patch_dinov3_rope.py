#!/usr/bin/env python3
"""Make DINOv3's rope encoding export an ONNX graph TensorRT 8.6 can parse.

`Tensor.tile(2)` with a single int is rank-dependent -- PyTorch left-pads the
repeats to the tensor's rank -- so the ONNX exporter cannot fold it and emits an
`If` that branches on the rank.  TensorRT 8.6 (JetPack 6.0) refuses that graph:

    IIfConditionalOutputLayer inputs must have the same shape

`angles` is [HW, D//2] at that point, so spelling the repeats out as
`tile(1, 2)` is the identical operation with no branch.

This edits torch.hub's CACHE, which is re-downloaded whenever the cache is
cleared or the upstream ref moves -- hence a script rather than a patch file:
it matches on the call, not on a line number, and re-running it is a no-op.

    python jetson/patch_dinov3_rope.py            # patch the default cache
    python jetson/patch_dinov3_rope.py --check    # report only, exit 1 if unpatched
"""
import argparse
import sys
from pathlib import Path

REL = "dinov3/layers/rope_position_encoding.py"
OLD = "angles = angles.tile(2)"
NEW = "angles = angles.tile(1, 2)"
NOTE = """\
        # tile() with a single int is rank-dependent (PyTorch left-pads the
        # repeats to the tensor's rank), so the ONNX exporter emits an `If`
        # to branch on that rank -- and TensorRT 8.6 refuses to parse it
        # ("IIfConditionalOutputLayer inputs must have the same shape").
        # angles is [HW, D//2] here, so spell the repeats out: identical
        # result, no branch in the graph.
"""


def find_targets(hub_dir):
    return sorted(hub_dir.glob(f"facebookresearch_dinov3_*/{REL}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub-dir", type=Path,
                    default=Path.home() / ".cache/torch/hub",
                    help="torch.hub cache root")
    ap.add_argument("--check", action="store_true",
                    help="report only; exit 1 if any copy is unpatched")
    args = ap.parse_args()

    targets = find_targets(args.hub_dir)
    if not targets:
        print(f"no DINOv3 checkout under {args.hub_dir} -- nothing to patch. "
              f"Run the model once to populate the cache, then re-run this.")
        return 1 if args.check else 0

    unpatched = 0
    for path in targets:
        src = path.read_text()
        if NEW in src:
            print(f"already patched: {path}")
            continue
        if OLD not in src:
            # Upstream changed this line; patching blind would be worse than
            # stopping, because the resulting graph would be silently wrong.
            print(f"UNRECOGNISED: {path}\n"
                  f"  expected to find {OLD!r}; upstream may have changed. "
                  f"Check the rope encoding by hand.", file=sys.stderr)
            unpatched += 1
            continue
        unpatched += 1
        if args.check:
            print(f"needs patching: {path}")
            continue
        line = next(l for l in src.splitlines() if OLD in l)
        indent = line[:len(line) - len(line.lstrip())]
        note = "".join(indent + l[8:] if l.strip() else l
                       for l in NOTE.splitlines(keepends=True))
        path.write_text(src.replace(line, note + line.replace(OLD, NEW), 1))
        print(f"patched: {path}")
        unpatched -= 1

    if args.check and unpatched:
        return 1
    return 1 if unpatched else 0


if __name__ == "__main__":
    sys.exit(main())
