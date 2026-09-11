"""下载官方 YOLO26 n/s/m-seg,导出 TensorRT FP16,用相机真实帧对比速度。"""
import os, sys, time, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODELS = ["yolo26n-seg", "yolo26s-seg", "yolo26m-seg"]
CUSTOM = str(Path(__file__).resolve().parent.parent.parent
              / "yolo26m-seg-custom_20260908_rtx4070.engine")

# ---- 固定测试帧:两台机器用同一张图,保证可比 ----
FRAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_frame.png")
bgr = cv2.imread(FRAME)
if bgr is None:
    print(f"读不到 {FRAME}"); sys.exit(1)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
print(f"测试帧 {bgr.shape[1]}x{bgr.shape[0]}  ({FRAME})", flush=True)

from ultralytics import YOLO
import live_pipeline as PIPE
from pathlib import Path

def bench(engine_path, label):
    y = YOLO(engine_path)
    pc = next((i for i, nm in y.names.items() if str(nm).lower() == "person"), 0)
    for _ in range(6):
        y.predict(bgr, classes=[pc], conf=0.4, verbose=False)
    N = 25
    sp = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    tot = 0.0; det_ms = 0.0; ndet = 0
    for _ in range(N):
        t = time.perf_counter()
        r = y.predict(bgr, classes=[pc], conf=0.4, verbose=False)[0]
        tot += time.perf_counter() - t
        for k in sp: sp[k] += r.speed[k]
        d = time.perf_counter()
        det = PIPE.largest_person(r, rgb.shape[:2])
        det_ms += time.perf_counter() - d
        ndet += det is not None
    return dict(label=label, predict=tot/N*1000, pre=sp["preprocess"]/N,
                inf=sp["inference"]/N, post=sp["postprocess"]/N,
                mask=det_ms/N*1000, det=f"{ndet}/{N}")

results = []
for name in MODELS:
    eng = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{name}.engine")
    if not os.path.exists(eng):
        print(f"[{name}] 下载 + 导出 TensorRT ...", flush=True)
        t = time.time()
        YOLO(f"{name}.pt").export(format="engine", half=True, imgsz=640,
                                  device=0, simplify=True)
        print(f"[{name}] 导出完成 {time.time()-t:.0f}s", flush=True)
    try:
        results.append(bench(eng, name)); print(f"[{name}] 测完", flush=True)
    except Exception as e:
        print(f"[{name}] 失败: {e}", flush=True)

if os.path.exists(CUSTOM):
    results.append(bench(CUSTOM, "custom-m(你的)"))

print("\n=== 1280x720 固定帧  FP16/imgsz640 ===")
print(f"{'模型':<16}{'推理':>8}{'前处理':>8}{'后处理':>8}{'掩码':>8}{'合计':>9}  检出")
for r in results:
    total = r["predict"] + r["mask"]
    print(f"{r['label']:<16}{r['inf']:>7.2f} {r['pre']:>7.2f} {r['post']:>7.2f} "
          f"{r['mask']:>7.2f} {total:>8.2f}  {r['det']}")
