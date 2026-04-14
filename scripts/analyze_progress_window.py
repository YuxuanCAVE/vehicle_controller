#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze path spacing and suggest a reasonable sim.progress_window range."
    )
    parser.add_argument(
        "path_file",
        nargs="?",
        default="data/path_ref.mat",
        help="Reference path .mat file. Defaults to data/path_ref.mat.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.10,
        help="Nominal controller period in seconds. Defaults to 0.10.",
    )
    parser.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 5.0, 8.0],
        help="Representative vehicle speeds in m/s used for recommendations.",
    )
    parser.add_argument(
        "--margin-factors",
        type=float,
        nargs="+",
        default=[3.0, 5.0],
        help="Window multipliers relative to points advanced per control step.",
    )
    return parser.parse_args()


def load_path(path_file: Path) -> tuple[np.ndarray, np.ndarray]:
    mat = loadmat(path_file)
    if "x_opt" not in mat or "y_opt" not in mat:
        raise KeyError(f"Reference file must contain x_opt and y_opt: {path_file}")

    x = np.asarray(mat["x_opt"], dtype=float).reshape(-1)
    y = np.asarray(mat["y_opt"], dtype=float).reshape(-1)
    if len(x) < 2:
        raise ValueError("Reference path must contain at least 2 points.")

    # Mirror the runtime behavior in ReferenceManager.
    if np.hypot(x[-1] - x[0], y[-1] - y[0]) < 0.50:
        x = x[:-1]
        y = y[:-1]

    return x, y


def summarize_spacing(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    ds = np.hypot(np.diff(x), np.diff(y))
    return {
        "count": float(len(ds)),
        "min": float(np.min(ds)),
        "p10": float(np.percentile(ds, 10)),
        "mean": float(np.mean(ds)),
        "median": float(np.median(ds)),
        "p90": float(np.percentile(ds, 90)),
        "max": float(np.max(ds)),
    }


def recommend_windows(
    spacing_stats: dict[str, float],
    dt: float,
    speeds: list[float],
    margin_factors: list[float],
) -> list[dict[str, float]]:
    mean_ds = max(spacing_stats["mean"], 1e-6)
    p10_ds = max(spacing_stats["p10"], 1e-6)
    rows = []
    for speed in speeds:
        points_per_step_mean = speed * dt / mean_ds
        points_per_step_dense = speed * dt / p10_ds
        row = {
            "speed": float(speed),
            "points_per_step_mean": float(points_per_step_mean),
            "points_per_step_dense": float(points_per_step_dense),
        }
        for factor in margin_factors:
            row[f"window_mean_x{factor:g}"] = float(np.ceil(max(1.0, factor * points_per_step_mean)))
            row[f"window_dense_x{factor:g}"] = float(
                np.ceil(max(1.0, factor * points_per_step_dense))
            )
        rows.append(row)
    return rows


def main() -> int:
    args = parse_args()
    path_file = Path(args.path_file).expanduser().resolve()
    x, y = load_path(path_file)
    stats = summarize_spacing(x, y)
    recommendations = recommend_windows(
        spacing_stats=stats,
        dt=float(args.dt),
        speeds=[float(speed) for speed in args.speeds],
        margin_factors=[float(factor) for factor in args.margin_factors],
    )

    print(f"path_file: {path_file}")
    print(f"path_points: {len(x)}")
    print("spacing_m:")
    print(f"  min:    {stats['min']:.4f}")
    print(f"  p10:    {stats['p10']:.4f}")
    print(f"  mean:   {stats['mean']:.4f}")
    print(f"  median: {stats['median']:.4f}")
    print(f"  p90:    {stats['p90']:.4f}")
    print(f"  max:    {stats['max']:.4f}")
    print(f"controller_dt_s: {args.dt:.3f}")
    print()
    print("suggested_progress_window:")
    print("  Notes:")
    print("  - window_mean uses average path spacing.")
    print("  - window_dense uses p10 spacing and is more conservative for dense local segments.")
    print("  - Pick a value big enough to survive localization jumps, but not so big that it can hop to nearby loops.")
    for row in recommendations:
        print(
            f"  speed={row['speed']:.2f} m/s | "
            f"pts/step(mean)={row['points_per_step_mean']:.2f} | "
            f"pts/step(dense)={row['points_per_step_dense']:.2f}"
        )
        for factor in args.margin_factors:
            print(
                f"    x{factor:g}: window_mean={int(row[f'window_mean_x{factor:g}'])}, "
                f"window_dense={int(row[f'window_dense_x{factor:g}'])}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
