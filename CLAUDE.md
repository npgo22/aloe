# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aloe is a hobby-rocket 6-DoF (six degrees of freedom) flight simulator with sensor modeling and navigation filter capabilities. The project is designed for both simulation/analysis and embedded deployment on microcontrollers.

**Key Features:**
- High-fidelity 6-DoF rocket dynamics with quaternion-based attitude representation
- Sensor modeling (IMU, GPS, barometer, magnetometer) with realistic noise
- Error-State Kalman Filter (ESKF) for state estimation
- Flight state machine with 5 stages (Pad, Ascent, Coast, Descent, Landed)
- Per-stage filter tuning with greedy optimization
- CLI for batch simulations and parameter sweeps
- Web-based GUI for interactive visualization

## Build and Test Commands

```bash
# Build all crates (release mode for performance)
cargo build --release

# Run all tests
cargo test --release --all-features

# Run a specific crate's tests
cargo test -p aloe-core --release
cargo test -p aloe-sim --release

# Lint (clippy must pass without warnings)
cargo clippy --all-targets --all-features -- -D warnings

# Format check
cargo fmt --all -- --check

# Format code
cargo fmt --all
```

## Running the Simulator

The main `aloe` binary provides both CLI and GUI modes:

```bash
# Launch web GUI (default mode)
cargo run --release
cargo run --release -- gui --port 8080 --host 0.0.0.0

# Run CLI simulation
cargo run --release -- cli --single
cargo run --release -- cli --sweep-params thrust --sweep-steps 10

# Filter tuning with greedy optimization
cargo run --release -- cli --tune-sweep --tune-mode greedy --tune-steps 15

# Monte Carlo analysis with multiple seeds/noise levels
cargo run --release -- cli --tune-sweep --tune-seeds "42 43 44" --tune-noise-scales "0.5 1.0 2.0"
```

## Crate Architecture

This is a Cargo workspace with 5 crates organized by deployment target and functionality:

### aloe-core (no_std, embedded-ready)
Core rocket flight estimation library designed for microcontrollers. Contains only the algorithms needed to run on bare-metal embedded systems:
- **ESKF** (`eskf.rs`): 15-state Error-State Kalman Filter with quaternion attitude
- **State Machine** (`state_machine.rs`): Flight phase detection (Pad → Ascent → Coast → Descent → Landed)
- **Quantization** (`quantize.rs`): Telemetry compression for low-bandwidth downlinks
- **LUT Data** (`lut_data.rs`): Pre-computed atmosphere tables for no_std environments

**Key constraints:**
- No heap allocations
- No standard library (unless `std` feature enabled for tests)
- Fixed-size buffers (e.g., 256-entry ring buffer for GPS latency compensation)

### aloe-sim (std required)
Simulation library providing high-fidelity 6-DoF rocket dynamics:
- **6-DoF Simulator** (`sim.rs`): RK4 integration with quaternion dynamics, aerodynamics, time-varying mass
- **Sensor Models** (`sensor.rs`): Realistic noise generation for IMU, GPS, barometer, magnetometer
- **Filter Wrapper** (`filter.rs`): Drives aloe-core ESKF with simulated sensor data, supports per-stage tuning

### aloe-cli
Command-line interface for batch processing:
- Single simulations with configurable rocket/environment parameters
- Parameter sweeps (e.g., vary thrust, drag coefficient)
- **Filter tuning modes**: greedy coordinate descent to optimize ESKF parameters
- CSV/JSON output for post-processing

### aloe-gui
Web-based visualization using Axum server:
- Configuration panels for rocket, environment, sensors, and filter parameters
- Real-time simulation endpoint
- Chart generation (2D trajectories, 3D plots)
- Filter error statistics and quantization analysis

### aloe
Main binary that orchestrates CLI and GUI modes via subcommands.

## Coordinate System Conventions

**Critical:** The entire codebase uses **NED (North-East-Down)** coordinates:
- **X**: North
- **Y**: East
- **Z**: Down (positive Z points toward Earth's center)
- **Altitude**: Stored as negative Z (e.g., `pos.z = -1000.0` means 1000m AGL)
- **Gravity**: `[0, 0, +9.80665]` in NED frame
- **Quaternions**: Rotate vectors from body frame → NED frame

When working with altitude or vertical position, always remember that `pos.z` is **negative** for positions above ground.

## Filter Tuning Architecture

The ESKF supports **per-stage tuning** where each of the 5 flight stages can have different noise parameters:

**Tuning Parameters (9 per stage):**
- `accel_noise_density`: IMU accelerometer noise density
- `gyro_noise_density`: IMU gyroscope noise density
- `accel_bias_instability`: Accelerometer bias random walk
- `gyro_bias_instability`: Gyroscope bias random walk
- `pos_process_noise`: Position process noise
- `r_gps_pos`: GPS position measurement noise variance
- `r_gps_vel`: GPS velocity measurement noise variance
- `r_baro`: Barometer measurement noise variance
- `r_mag`: Magnetometer measurement noise variance

**Greedy Tuning Mode** (`--tune-mode greedy`):
Performs coordinate descent optimization where each parameter is swept independently for each stage, keeping the best value before moving to the next parameter. This is used in CI workflows to find optimal filter configurations.

## Important Implementation Details

### ESKF State and Error-State
The filter maintains both a nominal state (position, velocity, quaternion, biases) and a 15-element error-state vector. The error-state represents small deviations and is reset to zero after each update. See `aloe-core/src/eskf.rs` for the detailed layout:
```
δx[0..2]   = δpos   (position error)
δx[3..5]   = δvel   (velocity error)
δx[6..8]   = δθ     (attitude error as rotation vector)
δx[9..11]  = δb_a   (accelerometer bias error)
δx[12..14] = δb_g   (gyroscope bias error)
```

### GPS Latency Compensation
The ESKF maintains a 256-entry ring buffer of historical states to handle GPS latency (up to 150ms). When a GPS measurement arrives, it's matched to the closest historical state by timestamp.

### Flight State Machine
The state machine uses sensor-based heuristics for phase detection:
- **Pad → Ascent**: High acceleration (>20 m/s²) or velocity (>10 m/s)
- **Ascent → Coast**: Low acceleration (<2 m/s²) after minimum burn time
- **Coast → Descent**: Vertical velocity becomes positive (going down) after apogee
- **Descent → Landed**: Low velocity near ground with confirmation window

### RK4 Integration
The 6-DoF simulator uses Runge-Kutta 4th order integration with a fixed 1ms timestep (1000 Hz). This provides stability for the stiff differential equations arising from high thrust-to-weight ratios.

## Development Workflow Notes

### Testing Strategy
- Core library tests run with `--all-features` to enable std for assertions
- Simulation tests verify physical constraints (energy conservation, stability derivatives)
- Filter tests use Monte Carlo seeds to ensure robustness

### Performance Considerations
- Always run simulations in release mode (`--release`) for realistic performance
- The 6-DoF simulator is computationally intensive due to 1ms timestep
- Filter tuning sweeps can take significant time (use `--tune-steps` to control)

### Adding New Parameters to Rocket/Sensor Models
When adding parameters:
1. Add to struct in `aloe-sim/src/*.rs`
2. Add CLI argument in `aloe-cli/src/main.rs` (Args struct)
3. Update `build_rocket_params()` or `build_sensor_config()` helper
4. Update GUI query parameters in `aloe-gui/src/lib.rs`

### Modifying ESKF Tuning
The `FilterConfig` struct in `aloe-sim/src/filter.rs` holds per-stage tuning. Each parameter is a `Vec<f64>` with 5 elements (one per stage). Use `get_stage_tuning()` to extract the `EskfTuning` struct for a specific stage.
