"""直接用 TensorRT Python API 测裸引擎耗时,绕开 ultralytics 包装"""
import sys, time, numpy as np, torch, tensorrt as trt
path = sys.argv[1]
with open(path, "rb") as f:
    head = f.read(4)
    n = int.from_bytes(head, "little")
    if 0 < n < 100000:            # ultralytics 元数据头
        f.read(n); blob = f.read()
    else:
        f.seek(0); blob = f.read()
logger = trt.Logger(trt.Logger.ERROR)
rt = trt.Runtime(logger)
eng = rt.deserialize_cuda_engine(blob)
ctx = eng.create_execution_context()
# TRT 8.6 没有 int64 等成员,按存在与否构建映射
dtype_map = {}
for tname, ttype in (("float32", torch.float32), ("float16", torch.float16),
                     ("int32", torch.int32), ("int64", torch.int64),
                     ("int8", torch.int8), ("bool", torch.bool)):
    if hasattr(trt, tname):
        dtype_map[getattr(trt, tname)] = ttype
bufs = {}
for i in range(eng.num_io_tensors):
    nm = eng.get_tensor_name(i)
    shp = tuple(eng.get_tensor_shape(nm))
    bufs[nm] = torch.zeros(shp, device="cuda", dtype=dtype_map[eng.get_tensor_dtype(nm)])
    ctx.set_tensor_address(nm, bufs[nm].data_ptr())
    mode = "in " if eng.get_tensor_mode(nm) == trt.TensorIOMode.INPUT else "out"
    print(f"  {mode} {nm} {shp}")
s = torch.cuda.Stream()
for _ in range(20):
    ctx.execute_async_v3(stream_handle=s.cuda_stream)
torch.cuda.synchronize()
N = 100
t = time.perf_counter()
for _ in range(N):
    ctx.execute_async_v3(stream_handle=s.cuda_stream)
torch.cuda.synchronize()
per = (time.perf_counter() - t) / N * 1000
print(f"  裸引擎 execute: {per:.2f} ms  ({1000/per:.0f} qps)  <- {path.split('/')[-1]}")
