"""多帧对拍:两条路径在实时视频上选出来的框/掩码是否一致。

conf 语义存疑(同一检测 ultralytics 报 0.800、引擎原始输出 0.918),
所以在把直连设为默认之前,用真实视频量化分歧率。
"""
import sys, time, argparse
import numpy as np, cv2
ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True)
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--repo",
                default=str(Path(__file__).resolve().parent.parent))
a = ap.parse_args()
sys.path.insert(0, a.repo)

import rclpy
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
rclpy.init(); nd = rclpy.create_node("agree")
q = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
st = {"m": None, "s": 0}
nd.create_subscription(Image, "/camera/color/image_raw",
                       lambda m: st.update(m=m, s=time.time_ns()), q)
import threading
threading.Thread(target=lambda: rclpy.spin(nd), daemon=True).start()

from ultralytics import YOLO
import live_pipeline as PIPE
from yolo_trt_runtime import YoloSegTRT
from pathlib import Path
y = YOLO(a.engine)
pc = next((i for i, n in y.names.items() if str(n).lower() == "person"), 0)
runner = YoloSegTRT(a.engine, conf=0.4, person_class=pc)

def decode(m):
    rows = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)
    im = rows[:, :m.width*3].reshape(m.height, m.width, 3)
    return np.ascontiguousarray(im if m.encoding == "rgb8" else im[..., ::-1])

box_err, ious, both, only_u, only_d, neither, last = [], [], 0, 0, 0, 0, 0
while len(box_err) + only_u + only_d + neither < a.frames:
    if st["m"] is None or st["s"] == last:
        time.sleep(0.003); continue
    last = st["s"]
    rgb = decode(st["m"])
    r = y.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), classes=[pc], conf=0.4, verbose=False)[0]
    ref = PIPE.largest_person(r, rgb.shape[:2])
    got = runner(rgb)
    if ref is None and got is None: neither += 1; continue
    if ref is None: only_d += 1; continue
    if got is None: only_u += 1; continue
    both += 1
    box_err.append(np.abs(np.asarray(ref[0], float) - np.asarray(got[0], float)).max())
    i = np.logical_and(ref[1], got[1]).sum(); u = np.logical_or(ref[1], got[1]).sum()
    ious.append(i / max(u, 1))

e, io = np.array(box_err), np.array(ious)
print(f"\n=== {both + only_u + only_d + neither} 帧对拍 ===")
print(f"  两边都检出      {both}")
print(f"  只有 ultralytics {only_u}    只有直连 {only_d}    都没有 {neither}")
if len(e):
    print(f"  框最大偏差      median {np.median(e):6.2f}  p90 {np.percentile(e,90):6.2f}  max {e.max():6.2f} px")
    print(f"  掩码 IoU        median {np.median(io):6.4f}  p10 {np.percentile(io,10):6.4f}  min {io.min():6.4f}")
    bad = int((e > 20).sum())
    print(f"  框偏差 >20px 的帧: {bad} / {len(e)}  ({100*bad/len(e):.1f}%)")
