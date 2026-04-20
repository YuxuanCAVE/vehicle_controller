#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from typing import Sequence

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message


CONTROLLER_RECORD_TOPIC = "/controller_record"
CONTROLLER_RECORD_COLUMNS = [
    "stamp_sec",
    "actual_loop_dt",
    "control_update_dt",
    "ref_idx",
    "x",
    "y",
    "yaw",
    "vx",
    "vy",
    "yaw_rate",
    "ax",
    "xr",
    "yr",
    "psi_ref",
    "kappa_ref",
    "v_ref",
    "e_longitudinal",
    "e_lateral",
    "e_heading",
    "steering_rad",
    "steering_command",
    "accel_cmd",
    "throttle",
    "brake",
    "f_resist",
    "f_required",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export /controller_record messages from a ROS 2 bag to CSV."
    )
    parser.add_argument("bag_path", help="Path to a ROS 2 bag directory.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV path. Defaults to <bag_path>/controller_record.csv.",
    )
    parser.add_argument(
        "--topic",
        default=CONTROLLER_RECORD_TOPIC,
        help=f"Topic to export. Defaults to {CONTROLLER_RECORD_TOPIC}.",
    )
    return parser.parse_args()


def guess_storage_id(bag_path: Path) -> str:
    if bag_path.is_file():
        if bag_path.suffix == ".mcap":
            return "mcap"
        if bag_path.suffix == ".db3":
            return "sqlite3"
    for candidate in bag_path.iterdir():
        if candidate.suffix == ".mcap":
            return "mcap"
        if candidate.suffix == ".db3":
            return "sqlite3"
    raise FileNotFoundError(f"Could not detect rosbag storage under {bag_path}")


def build_output_path(bag_path: Path, output: str | None) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    if bag_path.is_dir():
        return (bag_path / "controller_record.csv").resolve()
    return bag_path.with_name(f"{bag_path.stem}_controller_record.csv").resolve()


def validate_row(values: Sequence[float], topic: str) -> list[float]:
    if len(values) != len(CONTROLLER_RECORD_COLUMNS):
        raise ValueError(
            f"Topic {topic} produced {len(values)} values, "
            f"expected {len(CONTROLLER_RECORD_COLUMNS)}."
        )
    return [float(value) for value in values]


def export_topic_to_csv(bag_path: Path, output_path: Path, topic: str) -> int:
    storage_id = guess_storage_id(bag_path)
    storage_options = StorageOptions(uri=str(bag_path), storage_id=storage_id)
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if topic not in topic_types:
        available_topics = ", ".join(sorted(topic_types))
        raise ValueError(f"Topic {topic} not found in bag. Available topics: {available_topics}")

    message_type = get_message(topic_types[topic])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CONTROLLER_RECORD_COLUMNS)

        while reader.has_next():
            current_topic, data, _timestamp = reader.read_next()
            if current_topic != topic:
                continue

            msg = deserialize_message(data, message_type)
            writer.writerow(validate_row(msg.data, topic))
            rows_written += 1

    return rows_written


def main() -> int:
    args = parse_args()
    bag_path = Path(args.bag_path).expanduser().resolve()
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    output_path = build_output_path(bag_path, args.output)
    rows_written = export_topic_to_csv(bag_path, output_path, args.topic)
    print(f"Exported {rows_written} messages from {args.topic} to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
