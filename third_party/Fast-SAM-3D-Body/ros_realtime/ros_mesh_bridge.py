#!/usr/bin/env python3
"""ROS 2 bridge for real-time Fast SAM 3D Body visualization.

Runs on the system python (the one rclpy was built for). Forwards camera frames
to mesh_worker.py over ZMQ and republishes the returned meshes as RViz markers.

  in  : sensor_msgs/Image            (default /camera/color/image_raw)
  out : visualization_msgs/MarkerArray  /fastsam3d/mesh
        sensor_msgs/CompressedImage     /fastsam3d/overlay/compressed
"""
import argparse, json, sys
import numpy as np
import cv2
import zmq

sys.path.insert(0, __file__.rsplit('/', 2)[0])
from ros_realtime.wire import pack, unpack

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import Imu
import tf2_ros

PALETTE = [(0.65, 0.74, 0.86), (0.86, 0.70, 0.65), (0.68, 0.85, 0.70), (0.86, 0.82, 0.65)]


class MeshBridge(Node):
    def __init__(self, args):
        super().__init__("fast_sam3d_bridge")
        self.args = args
        self.last_sent = 0.0
        self.seen_ids = set()

        ctx = zmq.Context.instance()
        self.frames = ctx.socket(zmq.PUB)
        self.frames.setsockopt(zmq.SNDHWM, 2)
        self.frames.bind(args.frames_endpoint)
        self.results = ctx.socket(zmq.SUB)
        self.results.setsockopt(zmq.CONFLATE, 1)
        self.results.setsockopt(zmq.SUBSCRIBE, b"")
        self.results.connect(args.results_endpoint)

        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, args.image_topic, self.on_image, sensor_qos)
        self.gravity = None          # EMA of the accelerometer vector, in its own frame
        self.gravity_frame = ""
        self.floor_z = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        if args.imu_topic:
            self.create_subscription(Imu, args.imu_topic, self.on_imu, sensor_qos)
            self.create_timer(0.1, self.publish_world_tf)

        self.pub_mesh = self.create_publisher(MarkerArray, "/fastsam3d/mesh", 1)
        self.pub_overlay = self.create_publisher(CompressedImage, "/fastsam3d/overlay/compressed", 1)
        self.create_timer(0.01, self.poll_results)

        self.get_logger().info(f"subscribed to {args.image_topic}; "
                               f"publishing /fastsam3d/mesh at up to {args.send_hz} Hz")

    # ---------- gravity-aligned world frame ----------
    def on_imu(self, msg: Imu):
        a = np.array([msg.linear_acceleration.x,
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.z], dtype=np.float64)
        n = np.linalg.norm(a)
        if n < 5.0 or n > 15.0:          # not a plausible gravity reading
            return
        a /= n
        self.gravity_frame = msg.header.frame_id
        self.gravity = a if self.gravity is None else 0.98 * self.gravity + 0.02 * a

    def publish_world_tf(self):
        """Publish world -> root_frame so that +Z is up, using measured gravity."""
        if self.gravity is None:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.args.root_frame, self.gravity_frame, rclpy.time.Time())
        except Exception:
            return
        R = quat_to_mat(tf.transform.rotation)
        up = R @ self.gravity                       # gravity vector in root_frame, points up
        up /= np.linalg.norm(up)

        fwd = np.array([1.0, 0.0, 0.0])             # root_frame x is the camera's forward axis
        fwd = fwd - up * float(fwd @ up)
        if np.linalg.norm(fwd) < 1e-3:
            fwd = np.array([0.0, 0.0, 1.0]) - up * float(np.array([0.0, 0.0, 1.0]) @ up)
        fwd /= np.linalg.norm(fwd)
        left = np.cross(up, fwd)

        A = np.stack([fwd, left, up], axis=1)       # world -> root_frame
        q = mat_to_quat(A.T)                        # TF needs child(root_frame) -> parent(world)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.args.world_frame
        t.child_frame_id = self.args.root_frame
        t.transform.translation.z = float(-self.floor_z) if self.floor_z is not None else 0.0
        t.transform.rotation.x, t.transform.rotation.y = float(q[0]), float(q[1])
        t.transform.rotation.z, t.transform.rotation.w = float(q[2]), float(q[3])
        self.tf_broadcaster.sendTransform(t)

    # ---------- camera -> worker ----------
    def on_image(self, msg: Image):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_sent < 1.0 / self.args.send_hz:
            return
        self.last_sent = now

        buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        if msg.encoding == "rgb8":
            bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        elif msg.encoding == "bgr8":
            bgr = buf
        else:
            self.get_logger().warn(f"unsupported encoding {msg.encoding}", once=True)
            return
        if self.args.width and msg.width != self.args.width:
            h = int(round(msg.height * self.args.width / msg.width))
            bgr = cv2.resize(bgr, (self.args.width, h), interpolation=cv2.INTER_AREA)

        ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return
        meta = json.dumps({"stamp_sec": msg.header.stamp.sec,
                           "stamp_nsec": msg.header.stamp.nanosec,
                           "frame_id": msg.header.frame_id}).encode()
        self.frames.send(pack([meta, jpeg.tobytes()]), zmq.NOBLOCK)

    # ---------- worker -> RViz ----------
    def poll_results(self):
        try:
            header, faces_b, verts_b, overlay_b = unpack(self.results.recv(zmq.NOBLOCK))
        except zmq.Again:
            return
        h = json.loads(header)
        faces = np.frombuffer(faces_b, np.int32).reshape(-1, 3)
        verts = np.frombuffer(verts_b, np.float32).reshape(-1, 3)

        stamp = rclpy.time.Time(seconds=h["stamp_sec"], nanoseconds=h["stamp_nsec"]).to_msg()
        frame_id = h["frame_id"] or self.args.frame_id
        n = h["n_persons"]
        per = h["verts_per_person"]

        arr = MarkerArray()
        live = set()
        for i in range(n):
            v = verts[i * per:(i + 1) * per]
            tri = v[faces.reshape(-1)]                    # (3F, 3) triangle soup
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = frame_id
            m.ns, m.id = "person", i
            m.type, m.action = Marker.TRIANGLE_LIST, Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 1.0
            c = PALETTE[i % len(PALETTE)]
            m.color = ColorRGBA(r=c[0], g=c[1], b=c[2], a=1.0)
            m.lifetime.sec = 1
            m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in tri]
            arr.markers.append(m)
            live.add(i)

        if n and self.gravity is not None:
            self.update_floor(verts, frame_id)

        for stale in self.seen_ids - live:              # clear people who left the frame
            m = Marker()
            m.header.frame_id = frame_id
            m.ns, m.id, m.action = "person", stale, Marker.DELETE
            arr.markers.append(m)
        self.seen_ids = live

        if arr.markers:
            self.pub_mesh.publish(arr)
        if overlay_b:
            ci = CompressedImage()
            ci.header.stamp = stamp
            ci.header.frame_id = frame_id
            ci.format = "jpeg"
            ci.data = overlay_b
            self.pub_overlay.publish(ci)


    def update_floor(self, verts, frame_id):
        """Drop the world origin to the lowest mesh point so the grid reads as the floor."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.args.world_frame, frame_id, rclpy.time.Time())
        except Exception:
            return
        R = quat_to_mat(tf.transform.rotation)       # rotation only: height w.r.t. the camera
        z = (verts.astype(np.float64) @ R.T)[:, 2]
        low = float(np.percentile(z, 1.0))
        self.floor_z = low if self.floor_z is None else 0.9 * self.floor_z + 0.1 * low


def quat_to_mat(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x, y, z = (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.array([x, y, z, w])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-topic", default="/camera/color/image_raw")
    ap.add_argument("--frames-endpoint", default="tcp://127.0.0.1:5599")
    ap.add_argument("--results-endpoint", default="tcp://127.0.0.1:5600")
    ap.add_argument("--send-hz", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=0, help="downscale frames before inference")
    ap.add_argument("--frame-id", default="camera_color_optical_frame")
    ap.add_argument("--imu-topic", default="/camera/gyro_accel/sample",
                    help="accelerometer topic used to level the world frame ('' to disable)")
    ap.add_argument("--world-frame", default="fastsam3d_world")
    ap.add_argument("--root-frame", default="camera_link",
                    help="root of the camera TF tree; the world frame is attached above it")
    args = ap.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = MeshBridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
