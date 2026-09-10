#!/bin/bash
# Real-time pose-mesh visualization from a live ROS 2 image topic.
#   worker  (conda fast_sam_3d_body)  <-- ZMQ -->  bridge (system python + rclpy)  --> RViz2
# Usage: bash ros_realtime/run_realtime.sh [--topic /camera/color/image_raw] [--no-rviz]
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TOPIC=/camera/color/image_raw
IMG_SIZE=${IMG_SIZE:-448}
START_RVIZ=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --topic) TOPIC="$2"; shift 2;;
    --no-rviz) START_RVIZ=0; shift;;
    *) echo "unknown arg $1"; exit 1;;
  esac
done

LOGDIR=${LOGDIR:-/tmp/fastsam3d_ros}
mkdir -p "$LOGDIR"

# ---- 1. inference worker (conda env) ----
(
  source /home/oliver/anaconda3/etc/profile.d/conda.sh
  conda activate fast_sam_3d_body
  NVLIBS=$(ls -d "$CONDA_PREFIX"/lib/python3.11/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
  export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
  export GPU_HAND_PREP=1 LAYER_DTYPE=fp32 SKIP_KEYPOINT_PROMPT=1 IMG_SIZE=$IMG_SIZE
  export USE_COMPILE=0 MHR_USE_CUDA_GRAPH=0 KEYPOINT_PROMPT_INTERM_INTERVAL=999
  export BODY_INTERM_PRED_LAYERS=0,1,2 HAND_INTERM_PRED_LAYERS=0,1 MHR_NO_CORRECTIVES=1
  export FOV_TRT=0 FOV_FAST=1 FOV_MODEL=s FOV_LEVEL=0
  exec python ros_realtime/mesh_worker.py
) > "$LOGDIR/worker.log" 2>&1 &
WORKER=$!

# ---- 2. ROS bridge (system python, rclpy) ----
(
  set +u; source /opt/ros/humble/setup.bash
  exec /usr/bin/python3 ros_realtime/ros_mesh_bridge.py --image-topic "$TOPIC"
) > "$LOGDIR/bridge.log" 2>&1 &
BRIDGE=$!

# ---- 3. RViz ----
RVIZ=""
if [[ $START_RVIZ -eq 1 ]]; then
  ( set +u; source /opt/ros/humble/setup.bash
    exec rviz2 -d ros_realtime/fastsam3d.rviz ) > "$LOGDIR/rviz.log" 2>&1 &
  RVIZ=$!
fi

echo "worker pid $WORKER | bridge pid $BRIDGE | rviz pid ${RVIZ:-none}"
echo "logs in $LOGDIR   (model load takes ~15 s before the first mesh appears)"
trap 'kill $WORKER $BRIDGE ${RVIZ:-} 2>/dev/null' INT TERM
wait $WORKER
