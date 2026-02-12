# Aloe

A hobby-rocket flight simulator with sensor fusion (ES-EKF) and telemetry quantization. Fully implemented in Rust with a clean separation between `no_std` embedded code and standard library tools.

## Architecture

The project is organized as a Cargo workspace:

### `crates/aloe-core` (no_std)

Core estimation library for microcontroller deployment:
- **ESKF**: Error-State Kalman Filter for state estimation
- **Quantization**: Flight/recovery packet compression for telemetry
- **State Machine**: Flight phase detection (pad, burn, coasting, recovery)
- **LUT Data**: Atmosphere lookup tables

### `crates/aloe-sim` (std)

Simulation and modeling library:
- **Simulation**: 3-DoF rocket flight physics
- **Sensor Modeling**: Realistic sensor data generation with noise and latency
- **Parameters**: Complete parameter specifications

### `crates/aloe-cli`

Command-line interface for batch simulations.

### `crates/aloe`

Main binary with `cli` and `gui` subcommands.

## Quick Start

```bash
# Build everything
cargo build --release

# Run tests
cargo test --all

# Run GUI
cargo run -- gui

# Check formatting and linting
cargo fmt --all -- --check
cargo clippy --all
```

## GUI Usage

```bash
# Launch web GUI on default port 8080
cargo run -- gui

# Custom host/port
cargo run -- gui --host 0.0.0.0 --port 3000
```

The GUI provides an interactive interface for configuring rocket parameters, running simulations, and visualizing results with 3D trajectory plots and time-series data.

```bash
# Run with default parameters
cargo run --release -- cli

# Custom parameters
cargo run --release -- cli \
  --dry-mass 50.0 \
  --propellant-mass 150.0 \
  --thrust 15000.0 \
  --burn-time 25.0
```

### Simulation Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-mass` | 50.0 | Dry mass (kg) |
| `--propellant-mass` | 150.0 | Propellant mass (kg) |
| `--thrust` | 15000.0 | Thrust (N) |
| `--burn-time` | 25.0 | Burn time (s) |
| `--drag-coeff` | 0.40 | Drag coefficient |
| `--ref-area` | 0.03 | Reference area (m²) |
| `--gravity` | 9.81 | Gravity (m/s²) |
| `--wind-speed` | 3.0 | Wind X (m/s) |
| `--wind-speed-z` | 0.0 | Wind Z (m/s) |
| `--air-density` | 1.225 | Air density (kg/m³) |
| `--launch-delay` | 1.0 | Launch delay (s) |
| `--spin-rate` | 0.0 | Spin rate (°/s) |
| `--thrust-cant` | 0.0 | Thrust cant (°) |

## Library Usage

```rust
use aloe_sim::sim::{RocketParams, SimResult, simulate_6dof};

let params = RocketParams::default();
let result: SimResult = simulate_6dof(&params);
```

## Telemetry Packet Formats

### Flight Data (20 bytes)
- Position N/E: 2× i16 (±32 km, 1 m resolution)
- Altitude AGL: i32 (cm resolution)
- Velocity N/E/D: 3× i16 (0.1 m/s resolution)
- Roll/Pitch/Yaw: 3× u8 (0-255 → 0-360°)
- Status: u8

### Recovery Data (14 bytes)
- Latitude/Longitude: 2× i32 (degrees × 10^7)
- Altitude MSL: i16 (meters)
- Satellites/Fix info: u8
- Battery voltage: u8 (0.1 V resolution)

## Development

### Running Tests

```bash
cargo test --all
cargo test --doc
cargo test --lib
```

### Code Quality

```bash
# Format code
cargo fmt --all

# Run clippy
cargo clippy --all -- -D warnings

# Generate documentation
cargo doc --all --no-deps
```

## License

MIT OR Apache-2.0
