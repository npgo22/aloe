# Aloe

Hobby-rocket flight simulator with sensor fusion (ES-EKF) and telemetry quantization. Python + Rust (via maturin/PyO3).

## Quick Start

```sh
uv sync
uv run maturin develop --release
uv run aloe gui          # web UI on :8080
uv run aloe cli --help   # batch mode
```

## CLI

```
uv run aloe cli [OPTIONS]
```

### Modes

| Flag | Description |
|------|-------------|
| `--single` | Single simulation run (default: parameter sweep) |
| *(no flag)* | Full sweep across `--sweep-params` |

### Simulation Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-mass` | 50 | Dry mass (kg) |
| `--propellant-mass` | 150 | Propellant mass (kg) |
| `--thrust` | 15000 | Thrust (N) |
| `--burn-time` | 25 | Burn time (s) |
| `--drag-coeff` | 0.40 | Drag coefficient |
| `--ref-area` | 0.03 | Reference area (m²) |
| `--gravity` | 9.81 | Gravitational acceleration (m/s²) |
| `--wind-speed` | 3.0 | Crosswind X (m/s) |
| `--wind-speed-z` | 0.0 | Crosswind Z (m/s) |
| `--air-density` | 1.225 | Air density (kg/m³) |
| `--launch-delay` | 0.0 | Pre-launch pad idle (s) |
| `--spin-rate` | 0.0 | Roll rate (°/s) |
| `--thrust-cant` | 0.0 | Thrust cant angle (°) |

### Sweep Control

| Flag | Default | Description |
|------|---------|-------------|
| `--sweep-params` | `dry_mass propellant_mass thrust burn_time` | Parameters to sweep |
| `--sweep-steps` | 5 | Steps per parameter |

### Sensors

| Flag | Default | Description |
|------|---------|-------------|
| `--no-sensors` | off | Skip sensor data generation |
| `--seed` | 42 | RNG seed |
| `--noise-scale` | 1.0 | Global noise multiplier |
| `--disable-sensor` | *(none)* | Disable sensors: `bmi088_accel bmi088_gyro adxl375 ms5611 lis3mdl lc29h` |

### Kalman Filter

| Flag | Default | Description |
|------|---------|-------------|
| `--no-filter` | off | Skip Kalman filter (runs by default) |
| `--filter-report` | off | Generate XLSX error report |
| `--mag-declination` | 0.0 | Magnetic declination (°) |
| `--home-lat` | 35.0 | Home latitude (°) |
| `--home-lon` | -106.0 | Home longitude (°) |
| `--home-alt` | 1500.0 | Home altitude MSL (m) |

#### Data Output Columns

- **eskf**: Error-State Kalman Filter estimates (floating-point fusion of sensor data)
- **quantized_flight**: Flight telemetry after quantization (simulates radio transmission limits)
- **quant_roundtrip**: Quantization error (difference between original and quantize→dequantize roundtrip)
- **quant_recovery**: Recovery phase quantized telemetry (parachute descent mode)

### ESKF Tune-Sweep

The `--tune-sweep` mode sweeps ESKF tuning parameters **per flight stage** to find
optimal values. The four flight stages are:

| Stage | Description |
|-------|-------------|
| `pad` | Stationary on the launch pad before ignition |
| `burn` | Motor ignition through burnout (sustained upward acceleration) |
| `coasting` | Post-burnout ascent, decelerating under gravity and drag |
| `recovery` | Apogee reached, descending under drag / parachute |

Two tuning strategies are available via `--tune-mode`:

| Mode | Strategy | When to use |
|------|----------|-------------|
| `greedy` (default) | **Coordinate descent** — sweep the first param, lock the best value, then sweep the next param with the previous optimum locked in. Captures interactions between parameters. | Finding a combined optimum to use in production |
| `oat` | **One-at-a-time** — each (param, stage) is swept independently while all others stay at defaults. Ignores interactions. | Helps determine which parameters matter most for each stage |

```sh
# Greedy coordinate descent (default, recommended)
uv run aloe cli --tune-sweep

# OAT sensitivity analysis
uv run aloe cli --tune-sweep --tune-mode oat

# Sweep specific params and/or stages
uv run aloe cli --tune-sweep --tune-params r_baro r_gps_pos --tune-stages burn coasting

# Fewer steps for a quick exploratory sweep
uv run aloe cli --tune-sweep --tune-steps 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tune-sweep` | off | Enable per-stage tuning sweep |
| `--tune-mode` | `greedy` | Strategy: `greedy` (coordinate descent) or `oat` (sensitivity) |
| `--tune-steps` | 15 | Log-spaced steps per parameter |
| `--tune-params` | *(all 9)* | Subset of tuning params to sweep |
| `--tune-stages` | *(all 4)* | Subset of flight stages to sweep |

**Outputs:**

- `tune_sweep_summary.csv` — every evaluated (param, stage, value) with error metrics
- `optimised_tuning.json` *(greedy mode only)* — the final optimised per-stage config,
  ready to load or copy into your application

**Interpreting the results:**

| Metric | Meaning |
|--------|---------|
| `pos3d_rmse_m` | 3D position RMSE vs truth (primary objective) |
| `vel3d_rmse_ms` | 3D velocity RMSE vs truth |
| `pos3d_p95_m` | 95th-percentile position error |
| `state_*_abserr_s` | Absolute error in detected flight state transition time |

In **greedy mode**, the console shows `★` markers when a new best is found, and each
dimension prints `→ locked param/stage = value (cumulative rmse = …)`. The final
summary compares baseline (all defaults) vs optimised and prints the full per-stage
config table. In **OAT mode**, you compare each parameter's sweep curve independently
— look for the value that minimises `pos3d_rmse_m` in each (param, stage) cell.

**GitHub Actions workflows:**

- `.github/workflows/tune-sweep.yml` — OAT sensitivity (parallelised, one job per param)
- `.github/workflows/tune-greedy.yml` — Greedy coordinate descent across multiple
  rocket configurations (default, heavy/low-thrust, light/high-thrust, windy, etc.)


### Output

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output-dir` | `output` | Output directory |
| `-f, --format` | `parquet` | Output format: `parquet`, `csv`, `xlsx` |

## Docker

```sh
docker pull ghcr.io/<owner>/aloe:main
docker run -p 8080:8080 ghcr.io/<owner>/aloe:main
```

## LUT Regeneration

```sh
python3 gen_lut.py   # writes rust/src/lut_data.rs
```

## Wishlist
* Harden the ESEKF to withstand sensor failures (currnetly completely falls apart if any of GPS, one of the two accelerometers, or the gyroscope fail)
* Make the 
* Add implementations for other sensor fusion algorithms
* Improve state detection
* Research ways to hoist errors into adaptive kalman filter so that bad readings may be discarded
* Find a barometer that doesn't suck so much that the readings get killed
* Make the simulated LIS3MDL reading useful by making it not a constant field with no attitude coupling.
* Decouple ESEKF from gyro availability (right now predictions are not ran if the Gyro is lost)
* Add emergency accelerometer fallback when attitude cannot be obtained
* Use a LUT for the barometer to improve readings as the single-lapse-rate model is terrible past 11km
* Add gating based on innovation-based fault detection:
```python
let normalized_innovation = innovation.norm() / s.diagonal().map(|v| v.sqrt()).norm();
if normalized_innovation > 5.0 {
    return; // reject this measurement
}
```
* Only use the ADXL375 if the BMI088 is saturated.