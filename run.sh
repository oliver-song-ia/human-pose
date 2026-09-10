#!/usr/bin/env bash
# Launch a demo without retyping the environment.  Everything below is a
# default that can be overridden from the caller's environment, so
#   SOURCE=live ./run.sh --engine fastsam3d
# and
#   ./run.sh --no-gui --run-seconds 30
# both work, and any other flag is passed straight through to the launcher.
# No `set -u`: conda's own activate.d hooks reference unset variables
# (geotiff-activate.sh, among others) and abort the shell under it.
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Which camera: "bag" (the /camera3 recording) or "live" (Orbbec defaults).
SOURCE="${SOURCE:-bag}"

case "$(uname -m)" in
  aarch64)                                     # Jetson: venv, not conda
    source /opt/ros/humble/setup.bash
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
    source "$HOME/hp-venv/bin/activate"
    export DISPLAY="${DISPLAY:-:1}"
    : "${HUMAN_POSE_YOLO:=/home/ia/assets/yolo26m-seg-custom_20260908.engine}"
    : "${HUMAN_POSE_YOLO_PT:=/home/ia/assets/yolo26m-seg-custom_20260908.pt}"
    ;;
  *)
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate human-pose
    source /opt/ros/humble/setup.bash
    : "${HUMAN_POSE_YOLO:=/home/oliver/Documents/semantic_perception/yolo26m-seg-custom_20260908_rtx4070.engine}"
    : "${HUMAN_POSE_YOLO_PT:=/home/oliver/Documents/semantic_perception/yolo26m-seg-custom_20260908.pt}"
    ;;
esac
export HUMAN_POSE_YOLO HUMAN_POSE_YOLO_PT

if [ "$SOURCE" = "bag" ]; then
  # The recording publishes colour compressed only, and its depth is the
  # RealSense-style stream already registered to colour.
  export HUMAN_POSE_COLOR_TOPIC="${HUMAN_POSE_COLOR_TOPIC:-/camera3/color/image_raw/compressed}"
  export HUMAN_POSE_DEPTH_TOPIC="${HUMAN_POSE_DEPTH_TOPIC:-/camera3/aligned_depth_to_color/image_raw}"
  export HUMAN_POSE_INFO_TOPIC="${HUMAN_POSE_INFO_TOPIC:-/camera3/color/camera_info}"
fi
# SOURCE=live leaves the topics alone: the built-in defaults are the Orbbec
# driver's, which is what a live camera publishes.

exec python mesh_live_o3d.py --seg-pick "$@"
