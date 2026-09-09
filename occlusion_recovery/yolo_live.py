"""YOLO-seg only: live person mask from the RGB stream, nothing else.

No mesh model, no depth, no Open3D -- just the detector, so its own latency and
mask quality are visible on their own.

    HUMAN_POSE_YOLO=/path/to.engine python yolo_live.py [--ultralytics] [--run-seconds N]
"""
import argparse
import os
import time
from collections import deque

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=os.environ.get(
        "HUMAN_POSE_YOLO", "/home/ia/assets/yolo26m-seg-custom_20260908.engine"))
    ap.add_argument("--ultralytics", action="store_true",
                    help="use the ultralytics wrapper instead of the direct runner")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--run-seconds", type=float, default=0.0)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--person-only", action="store_true",
                    help="只画人(管线里的行为);默认画出模型认识的全部类别")
    args = ap.parse_args()

    import rclpy
    from sensor_msgs.msg import Image
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

    rclpy.init()
    node = rclpy.create_node("yolo_live")
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                     history=HistoryPolicy.KEEP_LAST)
    st = {"msg": None, "stamp": 0}

    def on_rgb(m):
        st["msg"] = m
        st["stamp"] = time.time_ns()

    node.create_subscription(Image, "/camera/color/image_raw", on_rgb, qos)
    import threading
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()

    from ultralytics import YOLO
    probe = YOLO(args.engine)
    person_cls = next((i for i, n in probe.names.items()
                       if str(n).lower() == "person"), 0)
    if not args.person_only:
        # 全类别模式:直连 runner 只解人,所以这里走 ultralytics,
        # 目的是在屏幕上看清模型认识的每一类,而不是测管线速度。
        detect_label = "ultralytics (全类别)"
        names = probe.names

        def detect_all(rgb):
            r = probe.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                              conf=args.conf, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                return []
            b = r.boxes.xyxy.cpu().numpy()
            c = r.boxes.conf.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            polys = r.masks.xy if r.masks is not None else [None] * len(b)
            return [(b[i], c[i], cl[i], polys[i] if i < len(polys) else None)
                    for i in range(len(b))]
        detect = None
    elif args.ultralytics:
        import live_pipeline as PIPE
        detect_label = "ultralytics"

        def detect(rgb):
            r = probe.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                              classes=[person_cls], conf=args.conf, verbose=False)[0]
            return PIPE.largest_person(r, rgb.shape[:2])
    else:
        from yolo_trt_runtime import YoloSegTRT
        runner = YoloSegTRT(args.engine, conf=args.conf, person_class=person_cls)
        detect_label = "direct TensorRT"

        def detect(rgb):
            return runner(rgb)

    print(f"engine  {os.path.basename(args.engine)}")
    print(f"path    {detect_label}")
    print("waiting for frames...", flush=True)

    def decode(m):
        rows = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)
        im = rows[:, :m.width * 3].reshape(m.height, m.width, 3)
        return np.ascontiguousarray(im if m.encoding == "rgb8" else im[..., ::-1])

    lat, det_ms, hits, n = deque(maxlen=120), deque(maxlen=120), 0, 0
    last_stamp, t_start = 0, time.time()
    green = np.array([0, 200, 90], np.uint8)

    while True:
        if args.run_seconds and time.time() - t_start > args.run_seconds:
            break
        m, stamp = st["msg"], st["stamp"]
        if m is None or stamp == last_stamp:
            time.sleep(0.003)
            continue
        last_stamp = stamp
        rgb = decode(m)
        t0 = time.perf_counter()
        out = detect(rgb) if detect is not None else detect_all(rgb)
        det_ms.append((time.perf_counter() - t0) * 1000)
        lat.append((time.time_ns() - stamp) * 1e-6)
        n += 1
        hits += bool(out) if isinstance(out, list) else (out is not None)

        if not args.no_window:
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if isinstance(out, list):
                # 每类一个颜色,标签写清楚,便于确认加载的是哪个模型
                palette = [(0, 200, 90), (0, 165, 255), (255, 120, 0),
                           (200, 0, 200), (0, 220, 220), (120, 60, 220)]
                for box, conf, cls, poly in out:
                    col = palette[int(cls) % len(palette)]
                    if poly is not None and len(poly) >= 3:
                        ov = vis.copy()
                        cv2.fillPoly(ov, [np.rint(poly).astype(np.int32)], col)
                        vis = cv2.addWeighted(ov, 0.45, vis, 0.55, 0)
                    x0, y0, x1, y1 = [int(v) for v in box]
                    cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
                    cv2.putText(vis, f"{names[int(cls)]} {conf:.2f}",
                                (x0, max(y0 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, col, 2, cv2.LINE_AA)
            elif out is not None:
                box, mask = out
                vis[mask] = (0.45 * vis[mask] + 0.55 * green).astype(np.uint8)
                x0, y0, x1, y1 = [int(v) for v in box]
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 255), 2)
            txt = (f"{detect_label}  detect {np.median(det_ms):5.1f} ms  "
                   f"arrival->done {np.median(lat):5.1f} ms  "
                   f"{1000/max(np.median(det_ms), 1e-6):4.1f} fps  "
                   f"hit {hits}/{n}")
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(vis, txt, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("YOLO-seg person mask", vis)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

        if n % 60 == 0:
            print(f"  detect {np.median(det_ms):6.2f} ms (p90 "
                  f"{np.percentile(det_ms, 90):6.2f})  arrival->done "
                  f"{np.median(lat):6.2f} ms  hit {hits}/{n}", flush=True)

    if det_ms:
        d, l = np.array(det_ms), np.array(lat)
        print(f"\n=== {detect_label} | {os.path.basename(args.engine)} ===")
        print(f"  帧数            {n}   检出 {hits}/{n}")
        print(f"  detect          {np.median(d):6.2f} / p90 {np.percentile(d,90):6.2f} ms")
        print(f"  到达->完成      {np.median(l):6.2f} / p90 {np.percentile(l,90):6.2f} ms")
        print(f"  吞吐            {1000/np.median(d):.1f} fps")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
