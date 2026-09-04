#!/usr/bin/env python3
"""Benchmark crop-to-SMPL-mesh latency for TokenHMR (batch 1)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mesh_live_o3d as ML


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("tokenhmr",), default="tokenhmr")
    ap.add_argument("--session", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=50)
    args = ap.parse_args()

    session = Path(args.session)
    frame = sorted((session / "frames").glob("*.npz"))[0]
    det = session / "detections" / frame.name
    with np.load(frame) as z:
        rgb = z["rgb"]
    with np.load(det) as z:
        if not bool(z["valid"]):
            raise RuntimeError(f"no valid person detection in {det}")
        bbox = z["box"].astype(np.float32)

    device = torch.device("cuda")
    loader, runner = {
        "tokenhmr": (ML.load_tokenhmr_engine, ML.run_tokenhmr),
    }[args.engine]
    eng, tf, _ = loader(device)
    backend = "TensorRT FP16" if eng.get("trt") is not None else "PyTorch AMP"
    for _ in range(args.warmup):
        runner(eng, tf, rgb, bbox, device)
    samples = []
    profiles = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        runner(eng, tf, rgb, bbox, device)
        samples.append((time.perf_counter() - t0) * 1000.0)
        profiles.append(dict(ML.HL.MODEL_PROFILE))
    a = np.asarray(samples)
    print(f"RESULT engine={args.engine} backend={backend} n={len(a)} "
          f"mean_ms={a.mean():.3f} median_ms={np.median(a):.3f} "
          f"p95_ms={np.percentile(a, 95):.3f} fps={1000.0/a.mean():.2f}")
    for key in ("model_pre", "model_gpu", "model_post"):
        values = np.asarray([p[key] for p in profiles])
        print(f"  {key}: mean={values.mean():.3f} ms, "
              f"median={np.median(values):.3f} ms")


if __name__ == "__main__":
    main()
