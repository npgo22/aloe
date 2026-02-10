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

The `--tune-sweep` mode performs a one-at-a-time (OAT) sweep of each ESKF tuning
parameter **per flight stage**, finding the optimal value for each (parameter, stage)
pair independently. The six flight stages are: `pad`, `ignition`, `burn`, `coasting`,
`apogee`, `recovery`.

```sh
# Full sweep: 9 params × 6 stages × 15 steps = 810 runs
uv run aloe cli --tune-sweep

# Sweep specific params and/or stages
uv run aloe cli --tune-sweep --tune-params r_baro r_gps_pos --tune-stages burn coasting

# Fewer steps for a quick exploratory sweep
uv run aloe cli --tune-sweep --tune-steps 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tune-sweep` | off | Enable per-stage tuning sweep |
| `--tune-steps` | 15 | Log-spaced steps per parameter |
| `--tune-params` | *(all 9)* | Subset of tuning params to sweep |
| `--tune-stages` | *(all 6)* | Subset of flight stages to sweep |

Outputs a `tune_sweep_summary.csv` with columns `tuning_param`, `stage`, `value`,
`pos3d_rmse_m`, and other error metrics. The best value per (param, stage) pair is
printed at the end. A GitHub Actions workflow (`.github/workflows/tune-sweep.yml`)
runs the sweep in parallel — one job per tuning parameter.


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