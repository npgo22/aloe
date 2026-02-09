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
| `--filter` | off | Run ES-EKF sensor fusion + quantization |
| `--filter-report` | off | Generate XLSX error report (implies `--filter`) |
| `--mag-declination` | 0.0 | Magnetic declination (°) |
| `--home-lat` | 35.0 | Home latitude (°) |
| `--home-lon` | -106.0 | Home longitude (°) |
| `--home-alt` | 1500.0 | Home altitude MSL (m) |

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
