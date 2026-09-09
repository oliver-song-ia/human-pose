"""Export TokenHMR's ViT + token pose regressor to a fixed FP16 TensorRT engine.

Run this once per machine: a TensorRT engine is tied to both the GPU
architecture and the TensorRT version, so an engine built on one box is
unusable on another.

    python export_tokenhmr_trt.py [--engine PATH] [--fp32]

This deliberately does NOT go through mesh_live_o3d.load_tokenhmr_engine.
That loader exists to serve the live demo: it skips the 2.6 GB training
checkpoint (TensorRT already owns every learned weight) and builds only the
SMPL layer, and it raises if the engine file is missing.  Both properties are
fatal here -- we need the learned ViT/token head that the checkpoint holds, and
we are running precisely because no engine exists yet.  So build the full
PyTorch model with TokenHMR's own loader instead.
"""
import argparse
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TOKENHMR_ROOT = HERE.parent / "third_party" / "TokenHMR"
CHECKPOINT = "data/checkpoints/tokenhmr_model_latest.ckpt"   # relative to TOKENHMR_ROOT
MODEL_CONFIG = "data/checkpoints/model_config.yaml"


class Conv1dAsConv2d(torch.nn.Module):
    """A Conv1d rewritten as an equivalent Conv2d over a height-1 image.

    TensorRT 8.x aborts while building this graph -- its Conv1DOptimization pass
    trips an internal assertion on the tokenizer decoder's 1-D convolutions
    ("Assertion out.size() == mNode.outputs.size()", formatCombinationsImpl).
    A Conv1d over [N, C, L] is by definition a Conv2d over [N, C, 1, L] with a
    (1, k) kernel, so the weights carry over unchanged (just unsqueezed) and the
    numerics are identical; TensorRT builds the 2-D path without complaint.
    """

    def __init__(self, conv1d):
        super().__init__()
        c = torch.nn.Conv2d(
            conv1d.in_channels, conv1d.out_channels,
            kernel_size=(1, conv1d.kernel_size[0]),
            stride=(1, conv1d.stride[0]),
            padding=(0, conv1d.padding[0]) if isinstance(conv1d.padding, tuple)
            else (0, conv1d.padding),
            dilation=(1, conv1d.dilation[0]),
            groups=conv1d.groups,
            bias=conv1d.bias is not None,
        )
        with torch.no_grad():
            c.weight.copy_(conv1d.weight.unsqueeze(2))
            if conv1d.bias is not None:
                c.bias.copy_(conv1d.bias)
        self.conv = c.to(conv1d.weight.device).to(conv1d.weight.dtype)

    def forward(self, x):
        return self.conv(x.unsqueeze(2)).squeeze(2)


def convert_conv1d(module):
    """Recursively swap every nn.Conv1d for the equivalent Conv2d wrapper."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Conv1d):
            setattr(module, name, Conv1dAsConv2d(child))
            n += 1
        else:
            n += convert_conv1d(child)
    return n


class TokenHMRRegressor(torch.nn.Module):
    """Neural regressor only; SMPL and camera projection stay in PyTorch."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.head = model.smpl_head
        # TokenHMR hides this module behind a plain Python Proxy and moves it
        # inside forward(), which makes ONNX treat trainable tensors as constants.
        # Register the exact decoder normally and call it without the Proxy.
        proxy = self.head.decpose.tokenize.__self__
        device = next(self.head.parameters()).device
        self.tokenizer = proxy.tokenizer.to(device).eval()
        self.head.decpose.tokenize = self.tokenizer.forward

    def forward(self, image):
        features = self.backbone(image)
        params, camera, _ = self.head(features)
        return (params["global_orient"], params["body_pose"],
                params["betas"], camera)


def load_full_model(device):
    """Full TokenHMR module (backbone + smpl_head) with checkpoint weights."""
    sys.path.insert(0, str(TOKENHMR_ROOT))
    old_cwd = os.getcwd()
    try:
        # TokenHMR's config resolves its asset paths relative to the repo root.
        os.chdir(TOKENHMR_ROOT)
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        ckpt = Path(CHECKPOINT)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"TokenHMR checkpoint missing: {TOKENHMR_ROOT / CHECKPOINT}")
        from tokenhmr.lib.models import load_tokenhmr
        # init_renderer=False keeps pyrender/EGL out of this: the export needs
        # no visualisation and a headless box may have no GL context at all.
        model, cfg = load_tokenhmr(checkpoint_path=str(ckpt),
                                   model_cfg=MODEL_CONFIG,
                                   is_train_state=False, is_demo=True,
                                   init_renderer=False)
    finally:
        os.chdir(old_cwd)
    return model.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=None,
                    help="output engine path (default engines/tokenhmr_vit_token_256_fp16.engine)")
    ap.add_argument("--fp32", action="store_true", help="build FP32 instead of FP16")
    ap.add_argument("--workspace", type=int, default=4, help="builder workspace, GiB")
    # opset 17 introduced a native LayerNormalization op.  TensorRT 8.6 mishandles
    # it in this graph (Assertion out.size() == mNode.outputs.size() in
    # formatCombinationsImpl); exporting at 16 decomposes LayerNorm into
    # ReduceMean/Sub/Pow/Sqrt/Div, which every TRT version builds cleanly.
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset (use 16 for TensorRT 8.x)")
    ap.add_argument("--conv1d-as-conv2d", action="store_true",
                    help="rewrite Conv1d as an equivalent Conv2d (needed on "
                         "TensorRT 8.x, whose Conv1DOptimization pass asserts)")
    args = ap.parse_args()

    device = torch.device("cuda")
    print("loading TokenHMR checkpoint (2.6 GB, this takes a minute)...", flush=True)
    model, cfg = load_full_model(device)
    wrapper = TokenHMRRegressor(model).eval()
    if args.conv1d_as_conv2d:
        n = convert_conv1d(wrapper)
        print(f"rewrote {n} Conv1d layers as Conv2d", flush=True)
    size = int(cfg.MODEL.IMAGE_SIZE)
    dummy = torch.zeros(1, 3, size, size, device=device)

    out_dir = HERE / "engines"
    out_dir.mkdir(exist_ok=True)
    onnx_path = out_dir / f"tokenhmr_vit_token_{size}_op{args.opset}.onnx"
    engine_path = Path(args.engine) if args.engine else (
        out_dir / f"tokenhmr_vit_token_{size}_fp16.engine")

    print(f"exporting ONNX (opset {args.opset}) -> {onnx_path}", flush=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, dummy, str(onnx_path),
            input_names=["image"],
            output_names=["global_orient", "body_pose", "betas", "camera"],
            opset_version=args.opset, do_constant_folding=True, dynamic_axes=None)

    print(f"building TensorRT engine -> {engine_path}", flush=True)
    from ultralytics.utils.export import onnx2engine
    onnx2engine(str(onnx_path), output_file=engine_path, workspace=args.workspace,
                half=not args.fp32, dynamic=False, shape=tuple(dummy.shape),
                verbose=False, prefix="TokenHMR TensorRT:")
    print(f"built {engine_path}", flush=True)


if __name__ == "__main__":
    main()
