#!/usr/bin/env bash
# Apply the machine-level fixes an AGX Orin needs for this pipeline.
#
# These are the things that live OUTSIDE the Python packages and therefore
# outside a git checkout: a systemd unit, and an edit to torch.hub's download
# cache.  A reflash, or clearing the hub cache, loses them silently -- the
# demos still start, they just run slower or fail to build an engine.
#
#   sudo ./jetson/setup.sh          install everything
#   ./jetson/setup.sh --check       report what is missing, change nothing
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

if [ "$(uname -m)" != "aarch64" ]; then
  echo "not a Jetson ($(uname -m)); nothing here applies" >&2
  exit 1
fi

fail=0

# ── clocks ────────────────────────────────────────────────────────────
# Without this the schedutil governor never ramps up under this workload: the
# GPU waits between short CPU bursts read as idle, and the CPU sits at ~70% of
# its ceiling.  Measured on the Fast SAM demo: frame_total 344 -> 286 ms.
UNIT=/etc/systemd/system/jetson-clocks.service
if [ "$CHECK" = 1 ]; then
  if systemctl is-enabled --quiet jetson-clocks.service 2>/dev/null; then
    echo "OK      jetson-clocks.service enabled"
  else
    echo "MISSING jetson-clocks.service"; fail=1
  fi
else
  install -m 644 jetson/jetson-clocks.service "$UNIT"
  systemctl daemon-reload
  systemctl enable --now jetson-clocks.service
  echo "installed $UNIT"
fi

# Whatever the unit's state, report the clocks themselves: an enabled unit that
# ran before nvpmodel settled leaves the ceiling where it was.
MIN=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq)
MAX=$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq)
if [ "$MIN" = "$MAX" ]; then
  echo "OK      CPU pinned at ${MAX} kHz"
else
  echo "WARN    CPU floor ${MIN} of ${MAX} kHz -- clocks are not pinned"
  fail=1
fi

# ── DINOv3 export fix ─────────────────────────────────────────────────
# Only matters when (re)building the Fast SAM backbone engine, but it is
# invisible until then, so check it here rather than at 3 GB into an export.
if [ "$CHECK" = 1 ]; then
  python3 jetson/patch_dinov3_rope.py --check && \
    echo "OK      DINOv3 rope patch" || { echo "MISSING DINOv3 rope patch"; fail=1; }
else
  python3 jetson/patch_dinov3_rope.py || true
fi

exit $fail
