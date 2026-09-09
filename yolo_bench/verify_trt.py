"""对拍 + 计时:直连 TRT runner vs ultralytics 包装。

先证明两者结果一致(框的偏差、掩码 IoU),再比速度 —— 快但算错了没有意义。
"""
import sys, time, argparse
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True)
ap.add_argument("--frame", default=None, help="图片路径;不给就从相机取一帧")
ap.add_argument("--repo", default="/home/ia/human-pose")
a = ap.parse_args()
sys.path.insert(0, a.repo)

# ---- 输入帧 ----
if a.frame:
    bgr = cv2.imread(a.frame)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
else:
    import rclpy
    from sensor_msgs.msg import Image
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    rclpy.init(); n = rclpy.create_node("v")
    q = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                   history=HistoryPolicy.KEEP_LAST)
    st = {}
    n.create_subscription(Image, "/camera/color/image_raw",
                          lambda m: st.update(m=m), q)
    t0 = time.time()
    while "m" not in st and time.time() - t0 < 15:
        rclpy.spin_once(n, timeout_sec=0.2)
    if "m" not in st:
        print("拿不到相机帧"); sys.exit(1)
    m = st["m"]
    rows = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)
    rgb = np.ascontiguousarray(rows[:, :m.width * 3].reshape(m.height, m.width, 3))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
print(f"帧 {rgb.shape[1]}x{rgb.shape[0]}")

# ---- 两条路径 ----
from ultralytics import YOLO
import live_pipeline as PIPE
from yolo_trt_runtime import YoloSegTRT

y = YOLO(a.engine)
pc = next((i for i, nm in y.names.items() if str(nm).lower() == "person"), 0)
runner = YoloSegTRT(a.engine, conf=0.4, person_class=pc)

r = y.predict(bgr, classes=[pc], conf=0.4, verbose=False)[0]
ref = PIPE.largest_person(r, rgb.shape[:2])
got = runner(rgb)

if ref is None or got is None:
    print(f"  ultralytics 检出={ref is not None}  直连检出={got is not None}")
    sys.exit(0 if ref is None and got is None else 1)

rbox, rmask = ref
gbox, gmask = got
inter = np.logical_and(rmask, gmask).sum()
union = np.logical_or(rmask, gmask).sum()
print("=== 正确性 ===")
print(f"  框 ultralytics {np.round(rbox, 1)}")
print(f"  框 直连        {np.round(gbox, 1)}")
print(f"  框最大偏差     {np.abs(np.asarray(rbox) - np.asarray(gbox)).max():.2f} px")
print(f"  掩码 IoU       {inter / max(union, 1):.4f}")
print(f"  掩码像素数     ultralytics {int(rmask.sum())}  直连 {int(gmask.sum())}")

# ---- 计时 ----
def bench(fn, N=30, warm=8):
    for _ in range(warm): fn()
    import torch; torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(N): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / N * 1000

def ultra():
    rr = y.predict(bgr, classes=[pc], conf=0.4, verbose=False)[0]
    return PIPE.largest_person(rr, rgb.shape[:2])

print("=== 速度 ===")
u = bench(ultra)
g = bench(lambda: runner(rgb))
print(f"  ultralytics 全路径  {u:6.2f} ms")
print(f"  直连 TRT 全路径     {g:6.2f} ms   ({u/g:.1f}x 快, 省 {u-g:.1f} ms)")
