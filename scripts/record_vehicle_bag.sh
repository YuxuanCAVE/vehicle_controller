#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <output_path>"
  exit 1
fi

output_path="$1"
timestamp="$(date +%Y%m%d_%H%M%S)"
resolved_output_path="${output_path}_${timestamp}"
suffix=1

while [[ -e "$resolved_output_path" ]]; do
  resolved_output_path="${output_path}_${timestamp}_${suffix}"
  suffix=$((suffix + 1))
done

mkdir -p "$(dirname "$resolved_output_path")"

echo "Recording bag to: $resolved_output_path"

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
  --output "$resolved_output_path"
