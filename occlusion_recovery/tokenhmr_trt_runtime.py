"""Zero-copy TensorRT 10 runner for the fixed-shape TokenHMR regressor."""
from pathlib import Path

import torch


ENGINE_PATH = Path(__file__).resolve().parent / "engines" / "tokenhmr_vit_token_256_fp16.engine"


class TokenHMRTensorRT:
    OUTPUTS = ("global_orient", "body_pose", "betas", "camera")

    def __init__(self, path=ENGINE_PATH):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()
        self.outputs = {}
        dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.bool: torch.bool,
        }
        for name in self.OUTPUTS:
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = dtype_map[self.engine.get_tensor_dtype(name)]
            self.outputs[name] = torch.empty(shape, device="cuda", dtype=dtype)
            self.context.set_tensor_address(name, self.outputs[name].data_ptr())
        # Pose is latency-sensitive while the BEV furniture detector is a
        # throughput/background task. Use CUDA's greatest-priority stream so
        # queued BEV kernels cannot add hundreds of milliseconds before a pose
        # inference. Stream priority changes scheduling only, never numerics.
        try:
            # CUDA/PyTorch convention: negative values request higher
            # priority. Older PyTorch builds expose Stream(priority=...) but
            # not get_stream_priority_range().
            self.stream = torch.cuda.Stream(priority=-1)
        except (TypeError, RuntimeError):
            # Priority is an optimization only. Never disable pose inference
            # merely because this PyTorch build cannot request it.
            self.stream = torch.cuda.Stream()

    def __call__(self, image):
        if tuple(image.shape) != (1, 3, 256, 256):
            raise ValueError(
                f"fixed TensorRT input expects 1x3x256x256, got {tuple(image.shape)}")
        image = image.contiguous().float()
        self.context.set_tensor_address("image", image.data_ptr())
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        if not self.context.execute_async_v3(stream_handle=self.stream.cuda_stream):
            raise RuntimeError("TokenHMR TensorRT execution failed")
        caller.wait_stream(self.stream)
        return tuple(self.outputs[name] for name in self.OUTPUTS)
