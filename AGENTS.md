# AGENTS.md

## Project Overview

Aloe is a hobby-rocket flight simulator with sensor fusion (ES-EKF) and telemetry quantization. The backend is written in Rust (compiled via maturin/PyO3 into a native Python extension), and the frontend/orchestration is Python using NiceGUI, Polars, and Plotly.

## Repository Layout

```
├── src/aloe/           # Python package
│   ├── __main__.py     # Entry point (dispatches gui / cli)
│   ├── cli.py          # CLI argument parsing & sweep orchestration
│   ├── sim.py          # Simulation driver (calls into Rust)
│   ├── filter.py       # Kalman-filter pipeline (calls into Rust)
│   ├── gui.py          # NiceGUI web UI (plots, controls)
│   └── params.py       # Parameter dataclasses & defaults
├── rust/               # Rust crate "aloe_core"
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs          # PyO3 module entry (exports to Python)
│       ├── eskf.rs         # Error-State Kalman Filter
│       ├── sim.rs          # 6-DoF rocket simulation
│       ├── sensor.rs       # Sensor modelling (noise, bias, saturation)
│       ├── quantize.rs     # Telemetry quantization / dequantization
│       ├── state_machine.rs# Flight-state detection
│       └── lut_data.rs     # Atmosphere look-up tables (generated)
├── gen_lut.py          # Regenerate lut_data.rs from atmosphere model
├── pyproject.toml      # Python project config (maturin build-backend)
├── Dockerfile
└── output/             # Default output directory (git-ignored)
```

## Setup

```sh
uv sync                                        # install Python deps + dev tools
uv run maturin develop --release                # build the Rust extension in-place
```

## Development Commands

### Python

```sh
uv run black .                                  # format Python code
uv run ruff check --fix .                       # lint Python code (auto-fix)
```

### Rust

```sh
cd rust
cargo fmt                                       # format Rust code
cargo clippy --all-targets --all-features -- -D warnings   # lint Rust code
```

### Pre-commit

Pre-commit hooks are configured in `.pre-commit-config.yaml`. Install them once:

```sh
uv run pre-commit install
```

After that, every `git commit` automatically runs Black, Ruff, `cargo fmt`, and `cargo clippy` on the staged files.

## Running

```sh
uv run aloe gui           # web UI on :8080
uv run aloe cli --help    # batch / sweep mode
```

## Build & Test Workflow

1. Make changes to Rust code in `rust/src/`.
2. Rebuild the extension: `uv run maturin develop --release`
3. Run the GUI or CLI to verify.

The Python package imports the Rust extension as `aloe.aloe_core`. If the native extension is unavailable, the Python code falls back to a no-op stub (GUI/CLI will warn).

## Coding Conventions

- **Python**: Formatted with Black (line-length 120). Linted with Ruff.
- **Rust**: Formatted with `cargo fmt`. Linted with `cargo clippy` (deny warnings).
- All sensor noise parameters use SI units (m, m/s, rad, Pa, µT).
- Polars DataFrames are the standard tabular format throughout the pipeline.
- Column naming convention: `{source}_{quantity}_{axis}` (e.g., `eskf_pos_n`, `bmi088_accel_x`).

## Architecture Notes

- **Rust → Python bridge**: PyO3 via maturin. The `#[pymodule]` in `lib.rs` exposes classes and free functions. Columnar `Vec<f32>` arrays are passed across the boundary (not DataFrames).
- **Sensor pipeline**: `sim.rs` produces truth trajectory → `sensor.rs` adds realistic sensor readings (noise, bias, saturation, latency) → `eskf.rs` fuses them → `quantize.rs` compresses for telemetry.
- **State machine**: `state_machine.rs` detects pad / burn / coast / recovery phases from sensor data. Also runs on truth data for comparison.
- **GUI**: Single-page NiceGUI app. 3D trajectory plot (Plotly), parameter controls, live rebuild on parameter change.
