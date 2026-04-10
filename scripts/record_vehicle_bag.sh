#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <output_path>"
  exit 1
fi

output_path="$1"

ros2 bag record \
  /ins/odometry \
  /ins/imu \
  /ins/nav_sat_fix \
  /ins/nav_sat_ref \
  /command \
  /enable \
  /controller_record \
  /tf \
  /tf_static \
  --output "$output_path"
