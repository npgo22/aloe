import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

from aloe.sim import RocketParams, SensorConfig, simulate_rocket, add_sensor_data
from aloe.params import get_param_ranges

try:
    from aloe.filter import FilterConfig, run_filter_pipeline, compute_error_report, write_error_report_xlsx

    _HAS_FILTER = True
except Exception:
    _HAS_FILTER = False

# ── Sweep ranges (matching GUI slider bounds) ────────────────────────
# Each entry: (param_name, start, stop, step)
DEFAULT_SWEEP = [
    ("dry_mass", 5, 100, 20),
    ("propellant_mass", 10, 300, 60),
    ("thrust", 500, 25000, 5000),
    ("burn_time", 2, 45, 10),
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aloe",
        description="Hobby-rocket flight simulator — CLI batch mode",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for output files (default: ./output)",
    )
    p.add_argument(
        "-f",
        "--format",
        choices=["parquet", "csv", "xlsx"],
        default="parquet",
        help="Output file format (default: parquet)",
    )

    # ── Single-run mode ──────────────────────────────────────────
    p.add_argument(
        "--single", action="store_true", help="Run a single simulation with the given parameters instead of a sweep"
    )
    p.add_argument("--dry-mass", type=float, default=50.0)
    p.add_argument("--propellant-mass", type=float, default=150.0)
    p.add_argument("--thrust", type=float, default=15000.0)
    p.add_argument("--burn-time", type=float, default=25.0)
    p.add_argument("--drag-coeff", type=float, default=0.40)
    p.add_argument("--ref-area", type=float, default=0.03)
    p.add_argument("--gravity", type=float, default=9.81)
    p.add_argument("--wind-speed", type=float, default=3.0)
    p.add_argument("--wind-speed-z", type=float, default=0.0)
    p.add_argument("--air-density", type=float, default=1.225)
    p.add_argument("--launch-delay", type=float, default=0.0, help="Pre-launch idle time on the pad (s, default: 0)")
    p.add_argument(
        "--spin-rate", type=float, default=0.0, help="Rocket roll rate around longitudinal axis (°/s, default: 0)"
    )
    p.add_argument(
        "--thrust-cant", type=float, default=0.0, help="Thrust vector cant angle from body axis (°, default: 0)"
    )

    # ── Sweep control ────────────────────────────────────────────
    p.add_argument(
        "--sweep-params",
        nargs="*",
        default=None,
        help="Parameters to sweep (default: dry_mass propellant_mass thrust burn_time)",
    )
    p.add_argument("--sweep-steps", type=int, default=5, help="Number of steps per swept parameter (default: 5)")

    # ── Sensor options ───────────────────────────────────────────
    p.add_argument("--no-sensors", action="store_true", help="Skip adding artificial sensor data")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sensor noise (default: 42)")
    p.add_argument("--noise-scale", type=float, default=1.0, help="Global noise multiplier (default: 1.0)")
    p.add_argument(
        "--disable-sensor",
        nargs="*",
        default=[],
        help="Sensor groups to disable: bmi088_accel bmi088_gyro adxl375 ms5611 lis3mdl lc29h",
    )

    # ── Filter / Quantization ────────────────────────────────────
    p.add_argument(
        "--filter",
        action="store_true",
        help="Run ES-EKF sensor fusion + telemetry quantization (requires aloe_core native lib)",
    )
    p.add_argument(
        "--filter-report",
        action="store_true",
        help="Generate XLSX error report (implies --filter)",
    )
    p.add_argument(
        "--mag-declination",
        type=float,
        default=0.0,
        help="Magnetic declination at launch site (degrees, default: 0)",
    )
    p.add_argument(
        "--home-lat",
        type=float,
        default=35.0,
        help="Home latitude (deg, default: 35.0)",
    )
    p.add_argument(
        "--home-lon",
        type=float,
        default=-106.0,
        help="Home longitude (deg, default: -106.0)",
    )
    p.add_argument(
        "--home-alt",
        type=float,
        default=1500.0,
        help="Home altitude MSL (m, default: 1500)",
    )
    return p


def _write(df, path: Path, fmt: str) -> None:
    """Write a DataFrame to disk in the requested format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.write_parquet(path.with_suffix(".parquet"))
    elif fmt == "csv":
        df.write_csv(path.with_suffix(".csv"))
    elif fmt == "xlsx":
        df.write_excel(path.with_suffix(".xlsx"), worksheet="Flight Data")


# ── Sweep range definitions (matching GUI sliders) ───────────────────
PARAM_RANGES = get_param_ranges()


def run_cli(argv: list[str] | None = None) -> None:
    """Entry point for CLI batch mode."""
    args = _build_parser().parse_args(argv)

    # Build sensor config
    sensor_cfg = SensorConfig(
        noise_scale=args.noise_scale,
        seed=args.seed,
    )
    for s in args.disable_sensor:
        sensor_cfg.enabled[s] = False

    if args.single:
        # ── Single run ───────────────────────────────────────────
        params = RocketParams(
            dry_mass=args.dry_mass,
            propellant_mass=args.propellant_mass,
            thrust=args.thrust,
            burn_time=args.burn_time,
            drag_coeff=args.drag_coeff,
            ref_area=args.ref_area,
            gravity=args.gravity,
            wind_speed=args.wind_speed,
            wind_speed_z=args.wind_speed_z,
            air_density=args.air_density,
            launch_delay=args.launch_delay,
            spin_rate=args.spin_rate,
            thrust_cant=args.thrust_cant,
        )
        df = simulate_rocket(params)
        if not args.no_sensors:
            df = add_sensor_data(df, sensor_cfg)

        # ── Filter pipeline ──────────────────────────────────────
        run_filter = args.filter or args.filter_report
        if run_filter:
            if not _HAS_FILTER:
                print("✗ aloe_core native lib not available. Build with: maturin develop --release", file=sys.stderr)
                sys.exit(1)
            fcfg = FilterConfig(
                mag_declination_deg=args.mag_declination,
                home_lat_deg=args.home_lat,
                home_lon_deg=args.home_lon,
                home_alt_m=args.home_alt,
            )
            df = run_filter_pipeline(df, fcfg)
            err_df = compute_error_report(df)
            print("\n── Error Statistics ──")
            print(err_df)

            if args.filter_report:
                report_path = args.output_dir / "sim_single_report.xlsx"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                write_error_report_xlsx(err_df, df, str(report_path))
                print(f"✓ Error report: {report_path}")

        out_path = args.output_dir / "sim_single"
        _write(df, out_path, args.format)
        print(f"✓ Wrote {out_path.with_suffix('.' + args.format)}  ({len(df)} rows)")
        return

    # ── Sweep mode ───────────────────────────────────────────────
    sweep_names = args.sweep_params or ["dry_mass", "propellant_mass", "thrust", "burn_time"]
    steps = args.sweep_steps

    # Validate
    for name in sweep_names:
        if name not in PARAM_RANGES:
            print(f"✗ Unknown parameter '{name}'. Valid: {', '.join(PARAM_RANGES)}", file=sys.stderr)
            sys.exit(1)

    # Build value lists
    axes: list[list[float]] = []
    for name in sweep_names:
        lo, hi = PARAM_RANGES[name]
        axes.append(np.linspace(lo, hi, steps).tolist())

    combos = list(itertools.product(*axes))
    total = len(combos)
    print(f"Sweeping {len(sweep_names)} params × {steps} steps = {total} simulations")

    for idx, values in enumerate(combos, 1):
        params = RocketParams()
        tag_parts = []
        for name, val in zip(sweep_names, values):
            setattr(params, name, val)
            tag_parts.append(f"{name}={val:.4g}")
        tag = "__".join(tag_parts)

        df = simulate_rocket(params)
        if not args.no_sensors:
            df = add_sensor_data(df, sensor_cfg)

        # ── Filter pipeline (sweep) ─────────────────────────────
        run_filter = args.filter or args.filter_report
        if run_filter:
            if not _HAS_FILTER:
                print("✗ aloe_core native lib not available.", file=sys.stderr)
                sys.exit(1)
            fcfg = FilterConfig(
                mag_declination_deg=args.mag_declination,
                home_lat_deg=args.home_lat,
                home_lon_deg=args.home_lon,
                home_alt_m=args.home_alt,
            )
            df = run_filter_pipeline(df, fcfg)
            err_df = compute_error_report(df)

            if args.filter_report:
                report_path = args.output_dir / f"sim_{tag}_report.xlsx"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                write_error_report_xlsx(err_df, df, str(report_path))

        out_path = args.output_dir / f"sim_{tag}"
        _write(df, out_path, args.format)
        print(f"  [{idx}/{total}] {tag}  ({len(df)} rows)")

    print(f"\n✓ All {total} simulations written to {args.output_dir}/")
