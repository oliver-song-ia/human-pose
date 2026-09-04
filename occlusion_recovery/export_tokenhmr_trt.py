"""Export TokenHMR's ViT + token pose regressor to a fixed FP16 TensorRT engine."""
from pathlib import Path

import torch

import mesh_live_o3d as ML


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


def main():
    device = torch.device("cuda")
    eng, _, _ = ML.load_tokenhmr_engine(device)
    wrapper = TokenHMRRegressor(eng["model"]).eval()
    dummy = torch.zeros(1, 3, 256, 256, device=device)
    out_dir = Path(__file__).resolve().parent / "engines"
    out_dir.mkdir(exist_ok=True)
    onnx_path = out_dir / "tokenhmr_vit_token_256.onnx"
    engine_path = out_dir / "tokenhmr_vit_token_256_fp16.engine"
    print(f"exporting ONNX -> {onnx_path}", flush=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, dummy, str(onnx_path),
            input_names=["image"],
            output_names=["global_orient", "body_pose", "betas", "camera"],
            opset_version=17, do_constant_folding=True, dynamic_axes=None)
    from ultralytics.utils.export import onnx2engine
    onnx2engine(str(onnx_path), output_file=engine_path, workspace=4,
                half=True, dynamic=False, shape=tuple(dummy.shape),
                verbose=False, prefix="TokenHMR TensorRT:")
    print(f"built {engine_path}", flush=True)


if __name__ == "__main__":
    main()
