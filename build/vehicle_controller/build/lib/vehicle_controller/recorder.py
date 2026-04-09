import csv
from datetime import datetime
from pathlib import Path
from typing import Any


class VehicleRecorder:
    FIELDNAMES = [
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
        "steering_norm",
        "accel_cmd",
        "throttle",
        "brake",
        "f_resist",
        "f_required",
    ]

    def __init__(self, output_dir: str, file_prefix: str = "vehicle_record") -> None:
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = out_dir / f"{file_prefix}_{timestamp}.csv"
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()
