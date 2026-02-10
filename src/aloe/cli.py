import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

from aloe.sim import RocketParams, SensorConfig, simulate_rocket, add_sensor_data
from aloe.params import (
    get_param_ranges,
    ESKF_TUNING_PARAMS,
    ESKF_TUNING_DEFAULTS,
    FLIGHT_STAGES,
    NUM_STAGES,
)

try:
    from aloe.filter import (
        FilterConfig,
        run_filter_pipeline,
        compute_error_report,
        write_error_report_xlsx,
    )

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
        "--single",
        action="store_true",
        help="Run a single simulation with the given parameters instead of a sweep",
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
    p.add_argument(
        "--launch-delay",
        type=float,
        default=1.0,
        help="Pre-launch idle time on the pad (s, default: 1)",
    )
    p.add_argument(
        "--spin-rate",
        type=float,
        default=0.0,
        help="Rocket roll rate around longitudinal axis (°/s, default: 0)",
    )
    p.add_argument(
        "--thrust-cant",
        type=float,
        default=0.0,
        help="Thrust vector cant angle from body axis (°, default: 0)",
    )

    # ── Sweep control ────────────────────────────────────────────
    p.add_argument(
        "--sweep-params",
        nargs="*",
        default=None,
        help="Parameters to sweep (default: dry_mass propellant_mass thrust burn_time)",
    )
    p.add_argument(
        "--sweep-steps",
        type=int,
        default=5,
        help="Number of steps per swept parameter (default: 5)",
    )

    # ── Sensor options ───────────────────────────────────────────
    p.add_argument("--no-sensors", action="store_true", help="Skip adding artificial sensor data")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sensor noise (default: 42)")
    p.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Global noise multiplier (default: 1.0)",
    )
    p.add_argument(
        "--disable-sensor",
        nargs="*",
        default=[],
        help="Sensor groups to disable: bmi088_accel bmi088_gyro adxl375 ms5611 lis3mdl lc29h",
    )

    # ── Filter / Quantization (always enabled by default) ────────
    p.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip Kalman filter (by default, filter always runs if native lib available)",
    )
    p.add_argument(
        "--filter-report",
        action="store_true",
        help="Generate XLSX error report (requires filter)",
    )

    # ── ESKF tuning parameters ──────────────────────────────────
    # Scalar values set all 6 stages uniformly.
    tuning_group = p.add_argument_group(
        "ESKF Tuning",
        "Override Kalman filter tuning parameters (uniform across all stages)",
    )
    for tname, tdefaults in ESKF_TUNING_DEFAULTS.items():
        tuning_group.add_argument(
            f"--{tname.replace('_', '-')}",
            type=float,
            default=tdefaults[0],
            help=f"{ESKF_TUNING_PARAMS[tname].label} (default: {tdefaults[0]})",
        )

    # ── Tune-sweep mode ───────────────────────────────────
    p.add_argument(
        "--tune-sweep",
        action="store_true",
        help="Sweep ESKF tuning parameters per flight stage",
    )
    p.add_argument(
        "--tune-mode",
        choices=["greedy", "oat"],
        default="greedy",
        help="Tuning strategy: 'greedy' = coordinate descent (tune one param, lock best, "
        "move to next — finds a combined optimum); 'oat' = one-at-a-time sensitivity "
        "analysis (each param swept independently, ignores interactions). Default: greedy",
    )
    p.add_argument(
        "--tune-steps",
        type=int,
        default=15,
        help="Number of log-spaced steps per tuning parameter (default: 15)",
    )
    p.add_argument(
        "--tune-params",
        nargs="*",
        default=None,
        help="Tuning params to sweep (default: all). " f"Valid: {' '.join(ESKF_TUNING_DEFAULTS.keys())}",
    )
    p.add_argument(
        "--tune-stages",
        nargs="*",
        default=None,
        help=f"Flight stages to sweep (default: all). Valid: {' '.join(FLIGHT_STAGES)}",
    )
    p.add_argument(
        "--sensor-failure-test",
        action="store_true",
        help="Test tuning under various sensor failure scenarios (IMU lost, GPS lost, baro lost, etc.)",
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
    # Avoid Path.with_suffix() – sweep tags may contain dots (e.g. "thrust=2.5e+04")
    # which would cause with_suffix to truncate the filename.
    out = path.parent / f"{path.name}.{fmt}"
    if fmt == "parquet":
        df.write_parquet(out)
    elif fmt == "csv":
        df.write_csv(out)
    elif fmt == "xlsx":
        df.write_excel(out, worksheet="Flight Data")


# ── Sweep range definitions (matching GUI sliders) ───────────────────
PARAM_RANGES = get_param_ranges()


def _compute_tune_metrics(
    df: pl.DataFrame,
) -> dict:
    """Compute position/velocity/state-timing error metrics from a filtered DataFrame."""
    row: dict = {}
    if "eskf_pos_n" not in df.columns:
        return row
    truth_n = df["position_x_m"].to_numpy().astype(np.float32)
    truth_e = df["position_z_m"].to_numpy().astype(np.float32)
    truth_d = -df["altitude_m"].to_numpy().astype(np.float32)
    truth_vn = df["velocity_x_ms"].to_numpy().astype(np.float32)
    truth_ve = df["velocity_z_ms"].to_numpy().astype(np.float32)
    truth_vd = -df["velocity_y_ms"].to_numpy().astype(np.float32)

    ekf_n = df["eskf_pos_n"].to_numpy().astype(np.float32)
    ekf_e = df["eskf_pos_e"].to_numpy().astype(np.float32)
    ekf_d = df["eskf_pos_d"].to_numpy().astype(np.float32)
    ekf_vn = df["eskf_vel_n"].to_numpy().astype(np.float32)
    ekf_ve = df["eskf_vel_e"].to_numpy().astype(np.float32)
    ekf_vd = df["eskf_vel_d"].to_numpy().astype(np.float32)

    pos_err = np.sqrt((ekf_n - truth_n) ** 2 + (ekf_e - truth_e) ** 2 + (ekf_d - truth_d) ** 2)
    vel_err = np.sqrt((ekf_vn - truth_vn) ** 2 + (ekf_ve - truth_ve) ** 2 + (ekf_vd - truth_vd) ** 2)
    row["pos3d_rmse_m"] = float(np.sqrt(np.mean(pos_err**2)))
    row["pos3d_max_m"] = float(np.max(pos_err))
    row["pos3d_p95_m"] = float(np.percentile(pos_err, 95))
    row["vel3d_rmse_ms"] = float(np.sqrt(np.mean(vel_err**2)))
    row["vel3d_max_ms"] = float(np.max(vel_err))
    row["vel3d_p95_ms"] = float(np.percentile(vel_err, 95))
    row["pos_n_rmse_m"] = float(np.sqrt(np.mean((ekf_n - truth_n) ** 2)))
    row["pos_e_rmse_m"] = float(np.sqrt(np.mean((ekf_e - truth_e) ** 2)))
    row["pos_d_rmse_m"] = float(np.sqrt(np.mean((ekf_d - truth_d) ** 2)))

    for sname in ["burn", "coasting", "recovery"]:
        truth_col = f"truth_{sname}_time"
        eskf_col = f"eskf_{sname}_time"
        if truth_col in df.columns and eskf_col in df.columns:
            truth_t = float(df[truth_col][0])
            eskf_t = float(df[eskf_col][0])
            if not np.isnan(truth_t) and not np.isnan(eskf_t):
                row[f"state_{sname}_abserr_s"] = float(abs(eskf_t - truth_t))
    return row


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

    def _build_filter_cfg(**tuning_overrides) -> "FilterConfig":
        """Build FilterConfig from CLI args + optional per-stage tuning overrides.

        ``tuning_overrides`` can contain:
        - scalar float values → applied uniformly to all stages
        - (param, stage_idx, value) is handled by the caller setting the list directly
        - list[float] of length NUM_STAGES → used as-is
        """
        tuning_kw: dict[str, list[float]] = {}
        for tname in ESKF_TUNING_DEFAULTS:
            override = tuning_overrides.get(tname)
            if override is not None:
                if isinstance(override, list):
                    tuning_kw[tname] = override
                else:
                    # Scalar override → apply to all stages
                    tuning_kw[tname] = [float(override)] * NUM_STAGES
            else:
                # Use the CLI arg value (scalar) expanded to all stages
                val = getattr(args, tname.replace("-", "_"))
                tuning_kw[tname] = [float(val)] * NUM_STAGES
        return FilterConfig(
            mag_declination_deg=args.mag_declination,
            home_lat_deg=args.home_lat,
            home_lon_deg=args.home_lon,
            home_alt_m=args.home_alt,
            accel_noise_density=tuning_kw["accel_noise_density"],
            gyro_noise_density=tuning_kw["gyro_noise_density"],
            accel_bias_instability=tuning_kw["accel_bias_instability"],
            gyro_bias_instability=tuning_kw["gyro_bias_instability"],
            pos_process_noise=tuning_kw["pos_process_noise"],
            r_gps_pos=tuning_kw["r_gps_pos"],
            r_gps_vel=tuning_kw["r_gps_vel"],
            r_baro=tuning_kw["r_baro"],
            r_mag=tuning_kw["r_mag"],
        )

    # ── Tune-sweep mode ──────────────────────────────────────────
    if args.tune_sweep:
        if not _HAS_FILTER:
            print(
                "✗ tune-sweep requires aloe_core native lib. Build with: maturin develop --release",
                file=sys.stderr,
            )
            sys.exit(1)

        tune_names = args.tune_params or list(ESKF_TUNING_DEFAULTS.keys())
        for tname in tune_names:
            if tname not in ESKF_TUNING_DEFAULTS:
                print(
                    f"✗ Unknown tuning parameter '{tname}'. Valid: {', '.join(ESKF_TUNING_DEFAULTS)}",
                    file=sys.stderr,
                )
                sys.exit(1)

        tune_stage_names = args.tune_stages or list(FLIGHT_STAGES)
        for sname in tune_stage_names:
            if sname not in FLIGHT_STAGES:
                print(
                    f"✗ Unknown flight stage '{sname}'. Valid: {', '.join(FLIGHT_STAGES)}",
                    file=sys.stderr,
                )
                sys.exit(1)
        tune_stage_indices = [FLIGHT_STAGES.index(s) for s in tune_stage_names]

        steps = args.tune_steps
        mode = args.tune_mode

        # Generate sim+sensor data once (rocket params are held constant)
        base_params = RocketParams(
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
        base_df = simulate_rocket(base_params)
        if not args.no_sensors:
            base_df = add_sensor_data(base_df, sensor_cfg)

        # Compute baseline metrics (all defaults)
        baseline_cfg = _build_filter_cfg()
        baseline_df = run_filter_pipeline(base_df, baseline_cfg)
        baseline_metrics = _compute_tune_metrics(baseline_df)
        baseline_rmse = baseline_metrics.get("pos3d_rmse_m", float("inf"))
        print(f"Baseline pos3d_rmse = {baseline_rmse:.4f} m (all defaults)")

        # ── Sensor failure scenario definitions ──────────────────────
        FAILURE_SCENARIOS = [
            ("all_sensors", []),
            ("no_imu", ["bmi088_accel", "bmi088_gyro", "adxl375"]),
            ("no_gyro", ["bmi088_gyro"]),
            ("no_accel", ["bmi088_accel", "adxl375"]),
            ("no_gps", ["lc29h"]),
            ("no_baro", ["ms5611"]),
            ("no_mag", ["lis3mdl"]),
            (
                "gps_only",
                ["bmi088_accel", "bmi088_gyro", "adxl375", "ms5611", "lis3mdl"],
            ),
            (
                "baro_only",
                ["bmi088_accel", "bmi088_gyro", "adxl375", "lc29h", "lis3mdl"],
            ),
        ]

        # Pre-generate sensor data for all failure scenarios once
        scenario_dataframes: dict[str, pl.DataFrame] = {}
        if args.sensor_failure_test:
            print("\n── Pre-generating sensor data for failure scenarios ──")
            for scenario_name, disabled_sensors in FAILURE_SCENARIOS:
                test_sensor_cfg = SensorConfig(
                    noise_scale=args.noise_scale,
                    seed=args.seed,
                )
                for s in disabled_sensors:
                    test_sensor_cfg.enabled[s] = False

                test_df = simulate_rocket(base_params)
                if not args.no_sensors:
                    test_df = add_sensor_data(test_df, test_sensor_cfg)
                scenario_dataframes[scenario_name] = test_df
            print(f"  Generated {len(FAILURE_SCENARIOS)} scenarios")

        current_best: dict[str, list[float]] | None = None
        tune_summary_rows: list[dict] = []
        run_idx = 0

        def _evaluate_config(
            fcfg, scenario_dfs: dict[str, pl.DataFrame] | None = None
        ) -> tuple[float, dict[str, dict]]:
            """Evaluate a filter config against one or more scenarios.

            Returns:
                (worst_case_rmse, dict[scenario_name, metrics])
            """
            if scenario_dfs is None:
                # Single scenario (backward compat)
                df = run_filter_pipeline(base_df, fcfg)
                metrics = _compute_tune_metrics(df)
                rmse = metrics.get("pos3d_rmse_m", float("inf"))
                return rmse, {"default": metrics}

            # Multi-scenario evaluation
            scenario_metrics = {}
            worst_rmse = 0.0
            for scenario_name, scenario_df in scenario_dfs.items():
                df = run_filter_pipeline(scenario_df, fcfg)
                metrics = _compute_tune_metrics(df)
                rmse = metrics.get("pos3d_rmse_m", float("inf"))
                scenario_metrics[scenario_name] = metrics
                worst_rmse = max(worst_rmse, rmse)

            return worst_rmse, scenario_metrics

        if mode == "greedy":
            # ── Greedy coordinate descent ─────────────────────────
            # Sweep one (param, stage) at a time, lock the best value,
            # then move to the next. Each subsequent param benefits from
            # all previously-optimised values.
            n_dims = len(tune_names) * len(tune_stage_names)
            total = n_dims * steps

            if args.sensor_failure_test:
                print(
                    f"Robust greedy tune: {len(tune_names)} params × {len(tune_stage_names)} stages × {steps} steps × {len(FAILURE_SCENARIOS)} scenarios "
                    f"= {total * len(FAILURE_SCENARIOS)} total evaluations"
                )
                print("  Optimizing for worst-case RMSE across all sensor failure scenarios")
            else:
                print(
                    f"Greedy tune: {len(tune_names)} params × {len(tune_stage_names)} stages × {steps} steps "
                    f"= {total} evaluations (sequential coordinate descent)"
                )

            if args.sensor_failure_test:
                # ── Greedy coordinate descent PER SCENARIO ─────────────────
                # Run complete greedy optimization for each failure scenario
                # Each scenario gets independently optimized
                scenario_best_configs: dict[str, dict[str, list[float]]] = {}

                for scenario_name, scenario_df in scenario_dataframes.items():
                    print(f"\n── Optimizing for scenario: {scenario_name} ──")

                    # Start fresh for each scenario
                    current_best = {tname: list(ESKF_TUNING_DEFAULTS[tname]) for tname in ESKF_TUNING_DEFAULTS}

                    # Get baseline for this scenario
                    baseline_cfg = _build_filter_cfg()
                    baseline_df = run_filter_pipeline(scenario_df, baseline_cfg)
                    baseline_metrics = _compute_tune_metrics(baseline_df)
                    current_rmse = baseline_metrics.get("pos3d_rmse_m", float("inf"))
                    print(f"  Baseline RMSE: {current_rmse:.4f} m")

                    for tname in tune_names:
                        spec = ESKF_TUNING_PARAMS[tname]
                        values = np.logspace(np.log10(spec.min), np.log10(spec.max), steps).tolist()

                        for stage_idx in tune_stage_indices:
                            stage_name = FLIGHT_STAGES[stage_idx]
                            best_val_for_dim = current_best[tname][stage_idx]
                            best_rmse_for_dim = current_rmse

                            for val in values:
                                run_idx += 1
                                # Build config: current best + this (param, stage) override
                                trial = {k: list(v) for k, v in current_best.items()}
                                trial[tname][stage_idx] = val
                                fcfg = _build_filter_cfg(**trial)

                                # Evaluate on THIS scenario only
                                df = run_filter_pipeline(scenario_df, fcfg)
                                metrics = _compute_tune_metrics(df)
                                rmse = metrics.get("pos3d_rmse_m", float("inf"))

                                # Log with scenario tag
                                trow = {
                                    "tuning_param": tname,
                                    "stage": stage_name,
                                    "value": val,
                                    "scenario": scenario_name,
                                    **metrics,
                                }
                                tune_summary_rows.append(trow)

                                marker = ""
                                if rmse < best_rmse_for_dim:
                                    best_rmse_for_dim = rmse
                                    best_val_for_dim = val
                                    marker = " ★"

                                print(
                                    f"    [{run_idx}] {scenario_name}/{tname}/{stage_name}={val:.6g}  "
                                    f"rmse={rmse:.4f}{marker}"
                                )

                            # Lock in best for this dimension
                            current_best[tname][stage_idx] = best_val_for_dim
                            current_rmse = best_rmse_for_dim

                    # Store optimized config for this scenario
                    scenario_best_configs[scenario_name] = current_best
                    print(f"  → Optimized RMSE for {scenario_name}: {current_rmse:.4f} m")

                # Print summary of all optimized configs
                print("\n" + "=" * 70)
                print("OPTIMIZED CONFIGURATIONS PER SCENARIO")
                print("=" * 70)
                for scenario_name, config in scenario_best_configs.items():
                    print(f"\n{scenario_name}:")
                    for tname in tune_names:
                        print(f"  {tname}: {[f'{v:.4g}' for v in config[tname]]}")

                # Use "all_sensors" config as the main current_best for final comparison
                if "all_sensors" in scenario_best_configs:
                    current_best = scenario_best_configs["all_sensors"]
                    current_rmse = baseline_rmse  # Will be recomputed below

            else:
                # ── Standard greedy coordinate descent (single scenario) ───
                current_best = {tname: list(ESKF_TUNING_DEFAULTS[tname]) for tname in ESKF_TUNING_DEFAULTS}
                current_rmse = baseline_rmse

                for tname in tune_names:
                    spec = ESKF_TUNING_PARAMS[tname]
                    values = np.logspace(np.log10(spec.min), np.log10(spec.max), steps).tolist()

                    for stage_idx in tune_stage_indices:
                        stage_name = FLIGHT_STAGES[stage_idx]
                        best_val_for_dim = current_best[tname][stage_idx]
                        best_rmse_for_dim = current_rmse

                        for val in values:
                            run_idx += 1
                            # Build config: current best + this (param, stage) override
                            trial = {k: list(v) for k, v in current_best.items()}
                            trial[tname][stage_idx] = val
                            fcfg = _build_filter_cfg(**trial)

                            df = run_filter_pipeline(base_df, fcfg)
                            metrics = _compute_tune_metrics(df)
                            rmse = metrics.get("pos3d_rmse_m", float("inf"))

                            trow = {
                                "tuning_param": tname,
                                "stage": stage_name,
                                "value": val,
                                **metrics,
                            }
                            tune_summary_rows.append(trow)

                            marker = ""
                            if rmse < best_rmse_for_dim:
                                best_rmse_for_dim = rmse
                                best_val_for_dim = val
                                marker = " ★"

                            print(
                                f"  [{run_idx}/{total}] {tname}/{stage_name}={val:.6g}  "
                                f"pos3d_rmse={rmse:.4f}{marker}"
                            )

                        # Lock in this dimension's best
                        current_best[tname][stage_idx] = best_val_for_dim
                        current_rmse = best_rmse_for_dim
                        print(
                            f"  → locked {tname}/{stage_name} = {best_val_for_dim:.6g}  "
                            f"(cumulative rmse = {current_rmse:.4f})"
                        )

        else:
            # ── OAT sensitivity analysis ─────────────────────────
            total = len(tune_names) * len(tune_stage_names) * steps
            print(
                f"OAT sweep: {len(tune_names)} params × {len(tune_stage_names)} stages × {steps} steps "
                f"= {total} evaluations (independent, no interaction)"
            )

            for tname in tune_names:
                spec = ESKF_TUNING_PARAMS[tname]
                values = np.logspace(np.log10(spec.min), np.log10(spec.max), steps).tolist()

                for stage_idx in tune_stage_indices:
                    stage_name = FLIGHT_STAGES[stage_idx]
                    for val in values:
                        run_idx += 1
                        defaults_for_param = list(ESKF_TUNING_DEFAULTS[tname])
                        defaults_for_param[stage_idx] = val
                        overrides = {tname: defaults_for_param}
                        fcfg = _build_filter_cfg(**overrides)

                        df = run_filter_pipeline(base_df, fcfg)
                        metrics = _compute_tune_metrics(df)

                        tag = f"tune__{tname}__{stage_name}={val:.6g}"
                        out_path = args.output_dir / f"sim_{tag}"
                        _write(df, out_path, args.format)

                        trow = {
                            "tuning_param": tname,
                            "stage": stage_name,
                            "value": val,
                            **metrics,
                        }
                        tune_summary_rows.append(trow)
                        print(
                            f"  [{run_idx}/{total}] {tname}/{stage_name}={val:.6g}  "
                            f"pos3d_rmse={trow.get('pos3d_rmse_m', 'N/A')}"
                        )

        # ── Write summary CSV ────────────────────────────────────
        summary_df = pl.DataFrame(tune_summary_rows)
        csv_path = args.output_dir / "tune_sweep_summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.write_csv(csv_path)
        print(f"\n✓ Tune-sweep summary written to {csv_path}")

        # ── Print per-(parameter, stage) best values ─────────────
        print(f"\n── Best tuning values per (param, stage)  [mode={mode}] ──")
        print(
            f"  {'parameter':<28s} {'stage':<12s} {'best_value':>14s} "
            f"{'pos3d_rmse':>12s} {'default':>14s} {'change':>8s}"
        )
        print(f"  {'─' * 28} {'─' * 12} {'─' * 14} {'─' * 12} {'─' * 14} {'─' * 8}")
        for tname in tune_names:
            for stage_idx in tune_stage_indices:
                stage_name = FLIGHT_STAGES[stage_idx]
                sub = summary_df.filter((pl.col("tuning_param") == tname) & (pl.col("stage") == stage_name))
                if "pos3d_rmse_m" in sub.columns and len(sub) > 0:
                    best_idx = sub["pos3d_rmse_m"].arg_min()
                    if best_idx is not None:
                        best_val = sub["value"][best_idx]
                        best_rmse = sub["pos3d_rmse_m"][best_idx]
                        default = ESKF_TUNING_DEFAULTS[tname][stage_idx]
                        if default != 0:
                            ratio = best_val / default
                            change = f"{ratio:.2f}×"
                        else:
                            change = "N/A"
                        print(
                            f"  {tname:<28s} {stage_name:<12s} {best_val:>14.6g} "
                            f"{best_rmse:>12.4f} {default:>14.6g} {change:>8s}"
                        )

        # ── Final comparison: baseline vs optimised ──────────────
        if mode == "greedy" and current_best is not None:
            print("\n── Greedy result: baseline vs optimised ──")
            final_cfg = _build_filter_cfg(**current_best)
            final_df = run_filter_pipeline(base_df, final_cfg)
            final_metrics = _compute_tune_metrics(final_df)
            final_rmse = final_metrics.get("pos3d_rmse_m", float("inf"))

            pct = (baseline_rmse - final_rmse) / baseline_rmse * 100 if baseline_rmse > 0 else 0
            print(f"  Baseline  pos3d_rmse = {baseline_rmse:.4f} m")
            print(f"  Optimised pos3d_rmse = {final_rmse:.4f} m  ({pct:+.1f}%)")
            print(f"  Baseline  vel3d_rmse = {baseline_metrics.get('vel3d_rmse_ms', 0):.4f} m/s")
            print(f"  Optimised vel3d_rmse = {final_metrics.get('vel3d_rmse_ms', 0):.4f} m/s")

            # Print the final optimised config
            print("\n  Optimised per-stage config:")
            print(f"  {'parameter':<28s}", end="")
            for sn in tune_stage_names:
                print(f" {sn:>12s}", end="")
            print()
            print(f"  {'─' * 28}", end="")
            for _ in tune_stage_names:
                print(f" {'─' * 12}", end="")
            print()
            for tname in tune_names:
                print(f"  {tname:<28s}", end="")
                for si in tune_stage_indices:
                    print(f" {current_best[tname][si]:>12.6g}", end="")
                print()

            # Write optimised config JSON
            import json

            config_path = args.output_dir / "optimised_tuning.json"
            json_cfg = {
                tname: {FLIGHT_STAGES[si]: current_best[tname][si] for si in range(NUM_STAGES)}
                for tname in ESKF_TUNING_DEFAULTS
            }
            config_path.write_text(json.dumps(json_cfg, indent=2) + "\n")
            print(f"\n  ✓ Optimised config written to {config_path}")

        print(f"\n✓ Tune-sweep complete ({len(tune_summary_rows)} evaluations)")
        return

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

        # ── Filter pipeline (default: always run if available) ──────
        run_filter = not args.no_filter
        if run_filter:
            if not _HAS_FILTER:
                print(
                    "⚠ aloe_core native lib not available. Skipping filter. " "Build with: maturin develop --release",
                    file=sys.stderr,
                )
            else:
                fcfg = _build_filter_cfg()
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
        print(f"✓ Wrote {out_path.parent / (out_path.name + '.' + args.format)}  ({len(df)} rows)")
        return

    # ── Sweep mode ───────────────────────────────────────────────
    sweep_names = args.sweep_params or [
        "dry_mass",
        "propellant_mass",
        "thrust",
        "burn_time",
    ]
    steps = args.sweep_steps

    # Validate
    for name in sweep_names:
        if name not in PARAM_RANGES:
            print(
                f"✗ Unknown parameter '{name}'. Valid: {', '.join(PARAM_RANGES)}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Build value lists
    axes: list[list[float]] = []
    for name in sweep_names:
        lo, hi = PARAM_RANGES[name]
        axes.append(np.linspace(lo, hi, steps).tolist())

    combos = list(itertools.product(*axes))
    total = len(combos)
    print(f"Sweeping {len(sweep_names)} params × {steps} steps = {total} simulations")

    summary_rows: list[dict[str, float]] = []

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
        run_filter = not args.no_filter
        if run_filter and _HAS_FILTER:
            fcfg = _build_filter_cfg()
            df = run_filter_pipeline(df, fcfg)
            err_df = compute_error_report(df)

            if args.filter_report:
                report_path = args.output_dir / f"sim_{tag}_report.xlsx"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                write_error_report_xlsx(err_df, df, str(report_path))

        out_path = args.output_dir / f"sim_{tag}"
        _write(df, out_path, args.format)

        # Collect per-run statistics
        row: dict[str, float] = {name: val for name, val in zip(sweep_names, values)}
        row["apogee_m"] = float(df["altitude_m"].max())  # type: ignore[arg-type]
        row["max_velocity_ms"] = float(df["velocity_total_ms"].max())  # type: ignore[arg-type]
        row["max_accel_ms2"] = float(df["acceleration_total_ms2"].max())  # type: ignore[arg-type]
        row["flight_time_s"] = float(df["time_s"].max())  # type: ignore[arg-type]

        # ── Filter comparison metrics ────────────────────────────
        if "eskf_pos_n" in df.columns:
            truth_n = df["position_x_m"].to_numpy().astype(np.float32)
            truth_e = df["position_z_m"].to_numpy().astype(np.float32)
            truth_d = -df["altitude_m"].to_numpy().astype(np.float32)
            truth_vn = df["velocity_x_ms"].to_numpy().astype(np.float32)
            truth_ve = df["velocity_z_ms"].to_numpy().astype(np.float32)
            truth_vd = -df["velocity_y_ms"].to_numpy().astype(np.float32)

            ekf_n = df["eskf_pos_n"].to_numpy().astype(np.float32)
            ekf_e = df["eskf_pos_e"].to_numpy().astype(np.float32)
            ekf_d = df["eskf_pos_d"].to_numpy().astype(np.float32)
            ekf_vn = df["eskf_vel_n"].to_numpy().astype(np.float32)
            ekf_ve = df["eskf_vel_e"].to_numpy().astype(np.float32)
            ekf_vd = df["eskf_vel_d"].to_numpy().astype(np.float32)

            # ESKF vs Truth — 3D position RMSE & max
            eskf_pos_err = np.sqrt((ekf_n - truth_n) ** 2 + (ekf_e - truth_e) ** 2 + (ekf_d - truth_d) ** 2)
            row["eskf_pos3d_rmse_m"] = float(np.sqrt(np.mean(eskf_pos_err**2)))
            row["eskf_pos3d_max_m"] = float(np.max(eskf_pos_err))
            row["eskf_pos3d_p95_m"] = float(np.percentile(eskf_pos_err, 95))

            # ESKF vs Truth — 3D velocity RMSE
            eskf_vel_err = np.sqrt((ekf_vn - truth_vn) ** 2 + (ekf_ve - truth_ve) ** 2 + (ekf_vd - truth_vd) ** 2)
            row["eskf_vel3d_rmse_ms"] = float(np.sqrt(np.mean(eskf_vel_err**2)))
            row["eskf_vel3d_max_ms"] = float(np.max(eskf_vel_err))
            row["eskf_vel3d_p95_ms"] = float(np.percentile(eskf_vel_err, 95))

            # State detection timing errors
            for sname in ["burn", "coasting", "recovery"]:
                truth_col = f"truth_{sname}_time"
                eskf_col = f"eskf_{sname}_time"
                if truth_col in df.columns and eskf_col in df.columns:
                    truth_t = float(df[truth_col][0])
                    eskf_t = float(df[eskf_col][0])
                    if not np.isnan(truth_t) and not np.isnan(eskf_t):
                        row[f"state_{sname}_err_s"] = float(eskf_t - truth_t)
                        row[f"state_{sname}_abserr_s"] = float(abs(eskf_t - truth_t))

        if "q_flight_pos_n_m" in df.columns:
            qpn = df["q_flight_pos_n_m"].to_numpy().astype(np.float32)
            qpe = df["q_flight_pos_e_m"].to_numpy().astype(np.float32)
            q_d = -df["q_flight_alt_m"].to_numpy().astype(np.float32)
            qvn = df["q_flight_vel_n_ms"].to_numpy().astype(np.float32)
            qve = df["q_flight_vel_e_ms"].to_numpy().astype(np.float32)
            qvd = df["q_flight_vel_d_ms"].to_numpy().astype(np.float32)

            # Quantized vs Truth — 3D position RMSE & max
            qt_pos_err = np.sqrt((qpn - truth_n) ** 2 + (qpe - truth_e) ** 2 + (q_d - truth_d) ** 2)
            row["quant_pos3d_rmse_m"] = float(np.sqrt(np.mean(qt_pos_err**2)))
            row["quant_pos3d_max_m"] = float(np.max(qt_pos_err))
            row["quant_pos3d_p95_m"] = float(np.percentile(qt_pos_err, 95))

            # Quantized vs Truth — 3D velocity RMSE
            qt_vel_err = np.sqrt((qvn - truth_vn) ** 2 + (qve - truth_ve) ** 2 + (qvd - truth_vd) ** 2)
            row["quant_vel3d_rmse_ms"] = float(np.sqrt(np.mean(qt_vel_err**2)))
            row["quant_vel3d_max_ms"] = float(np.max(qt_vel_err))
            row["quant_vel3d_p95_ms"] = float(np.percentile(qt_vel_err, 95))

            # Quantization-only error (quant round-trip vs ESKF)
            rt_pos_err = np.sqrt((qpn - ekf_n) ** 2 + (qpe - ekf_e) ** 2 + (q_d - ekf_d) ** 2)
            row["quant_rt_pos3d_rmse_m"] = float(np.sqrt(np.mean(rt_pos_err**2)))
            row["quant_rt_pos3d_max_m"] = float(np.max(rt_pos_err))

            rt_vel_err = np.sqrt((qvn - ekf_vn) ** 2 + (qve - ekf_ve) ** 2 + (qvd - ekf_vd) ** 2)
            row["quant_rt_vel3d_rmse_ms"] = float(np.sqrt(np.mean(rt_vel_err**2)))
            row["quant_rt_vel3d_max_ms"] = float(np.max(rt_vel_err))

        summary_rows.append(row)

        print(f"  [{idx}/{total}] {tag}  ({len(df)} rows)")

    # ── Write sweep summary CSV ──────────────────────────────────
    summary_df = pl.DataFrame(summary_rows)
    csv_path = args.output_dir / "sweep_summary.csv"
    summary_df.write_csv(csv_path)
    print(f"\n✓ Sweep summary written to {csv_path}")

    # ── Print aggregate statistics ───────────────────────────────
    # Group stats by category for readability
    flight_cols = ["apogee_m", "max_velocity_ms", "max_accel_ms2", "flight_time_s"]
    eskf_cols = [c for c in summary_df.columns if c.startswith("eskf_")]
    state_cols = [c for c in summary_df.columns if c.startswith("state_")]
    quant_vs_truth_cols = [c for c in summary_df.columns if c.startswith("quant_") and "_rt_" not in c]
    quant_rt_cols = [c for c in summary_df.columns if "_rt_" in c]

    def _fmt(v: float | None) -> str:
        return f"{v:>12.3f}" if v is not None else f"{'N/A':>12s}"

    def _print_group(title: str, cols: list[str]) -> None:
        if not cols:
            return
        print(f"\n  {title}")
        print(f"  {'metric':<30s} {'min':>12s} {'max':>12s} {'mean':>12s} {'std':>12s}")
        print(f"  {'─' * 30} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 12}")
        for col in cols:
            s = summary_df[col].drop_nulls()
            if len(s) == 0:
                continue
            vstd = s.std() if len(s) > 1 else 0.0
            print(f"  {col:<30s} {_fmt(s.min())} {_fmt(s.max())} {_fmt(s.mean())} {_fmt(vstd)}")

    print("\n── Sweep Statistics ──")
    _print_group("Flight Parameters", flight_cols)
    _print_group("ESKF vs Truth", eskf_cols)
    _print_group("State Detection Timing Errors", state_cols)
    _print_group("Quantized vs Truth", quant_vs_truth_cols)
    _print_group("Quantization Round-Trip (ESKF→Quant→Dequant)", quant_rt_cols)

    print(f"\n✓ All {total} simulations written to {args.output_dir}/")
