"""Parameter definitions and ranges for rocket simulation.

This module defines all configurable parameters, their bounds, and default steps
for both GUI sliders and CLI sweeps. All ranges match between GUI and CLI modes.
"""

from typing import NamedTuple


class ParamSpec(NamedTuple):
    """Parameter specification with label, bounds, and default step."""

    label: str
    min: float
    max: float
    step: float


# ── Rocket physical parameters ───────────────────────────────────────
ROCKET_PARAMS: dict[str, ParamSpec] = {
    "dry_mass": ParamSpec("Dry Mass (kg)", 5, 100, 1),
    "propellant_mass": ParamSpec("Propellant (kg)", 10, 300, 5),
    "thrust": ParamSpec("Thrust (N)", 500, 25000, 250),
    "burn_time": ParamSpec("Burn Time (s)", 2, 45, 0.5),
    "drag_coeff": ParamSpec("Drag Coeff.", 0.15, 1.5, 0.05),
    "ref_area": ParamSpec("Ref. Area (m²)", 0.005, 0.08, 0.002),
    "launch_delay": ParamSpec("Launch Delay (s)", 0, 30, 0.5),
    "spin_rate": ParamSpec("Spin Rate (°/s)", 0, 3600, 30),
    "thrust_cant": ParamSpec("Thrust Cant (°)", 0, 10, 0.1),
}

# ── Environmental parameters ──────────────────────────────────────────
ENV_PARAMS: dict[str, ParamSpec] = {
    "gravity": ParamSpec("Gravity (m/s²)", 1.0, 15, 0.1),
    "wind_speed": ParamSpec("Wind X (m/s)", 0, 25, 1),
    "wind_speed_z": ParamSpec("Crosswind Z (m/s)", -25, 25, 1),
    "air_density": ParamSpec("Air Density (kg/m³)", 0.3, 1.8, 0.02),
}

# ── Sensor configuration parameters ───────────────────────────────────
SENSOR_RATE_PARAMS: dict[str, ParamSpec] = {
    "bmi088_accel_hz": ParamSpec("BMI088 Accel Hz", 100, 1600, 100),
    "bmi088_gyro_hz": ParamSpec("BMI088 Gyro Hz", 100, 2000, 100),
    "adxl375_hz": ParamSpec("ADXL375 Hz", 100, 3200, 100),
    "ms5611_hz": ParamSpec("MS5611 Hz", 10, 122, 5),
    "lis3mdl_hz": ParamSpec("LIS3MDL Hz", 10, 155, 5),
    "lc29h_hz": ParamSpec("GPS Hz", 1, 10, 1),
}
SENSOR_LATENCY_PARAMS: dict[str, ParamSpec] = {
    "bmi088_accel_latency_ms": ParamSpec("BMI088 Accel (ms)", 0.1, 50, 0.5),
    "bmi088_gyro_latency_ms": ParamSpec("BMI088 Gyro (ms)", 1, 200, 5),
    "adxl375_latency_ms": ParamSpec("ADXL375 (ms)", 0.1, 10, 0.1),
    "ms5611_latency_ms": ParamSpec("MS5611 (ms)", 1, 50, 1),
    "lis3mdl_latency_ms": ParamSpec("LIS3MDL (ms)", 1, 100, 5),
    "lc29h_latency_ms": ParamSpec("GPS TTFF (ms)", 100, 26000, 100),
}

# ── ESKF tuning parameters ────────────────────────────────────────────
# Each value sweeps logarithmically from min to max.
# Extended ranges based on greedy tuning results showing optimal values at boundaries.
ESKF_TUNING_PARAMS: dict[str, ParamSpec] = {
    "accel_noise_density": ParamSpec("Accel Noise (m/s²/√Hz)", 0.001, 20.0, 0.1),
    "gyro_noise_density": ParamSpec("Gyro Noise (rad/s/√Hz)", 0.00001, 0.5, 0.001),
    "accel_bias_instability": ParamSpec("Accel Bias Inst.", 1e-8, 1e-2, 1e-5),
    "gyro_bias_instability": ParamSpec("Gyro Bias Inst.", 1e-8, 0.01, 1e-6),
    "pos_process_noise": ParamSpec("Pos Proc Noise (m/√s)", 0.0001, 10.0, 0.01),
    "r_gps_pos": ParamSpec("R GPS Pos (m²)", 0.01, 500.0, 1.0),
    "r_gps_vel": ParamSpec("R GPS Vel ((m/s)²)", 0.001, 100.0, 0.05),
    "r_baro": ParamSpec("R Baro (m²)", 0.001, 200.0, 0.5),
    "r_mag": ParamSpec("R Mag", 0.0001, 10.0, 0.01),
}

# ── Flight stages (indices match Rust FlightState enum) ───────────────
FLIGHT_STAGES: list[str] = ["pad", "burn", "coasting", "recovery"]
NUM_STAGES: int = len(FLIGHT_STAGES)

# ── Per-stage ESKF tuning defaults (must match Rust EskfTuning::default()) ─
# Each key maps to a list of 4 values: [pad, burn, coasting, recovery].
# Optimised via greedy sweep on the 30 km (default) rocket profile.
ESKF_TUNING_DEFAULTS: dict[str, list[float]] = {
    "accel_noise_density": [0.2236, 0.02430, 0.01, 0.01],
    "gyro_noise_density": [0.03728, 0.01389, 0.1, 0.01389],
    "accel_bias_instability": [0.01, 0.002683, 1e-6, 1e-6],
    "gyro_bias_instability": [3.728e-5, 1e-5, 1e-7, 1e-3],
    "pos_process_noise": [1.0, 0.1389, 0.004394, 0.007197],
    "r_gps_pos": [61.05, 0.1, 0.1, 0.1],
    "r_gps_vel": [0.07197, 0.04394, 0.04394, 0.01],
    "r_baro": [0.1, 2.236, 50.0, 50.0],
    "r_mag": [1.0, 0.01179, 1.0, 0.002683],
}
# ── Combined parameter definitions ────────────────────────────────────
# All parameters that can be swept/configured
ALL_PARAMS = {**ROCKET_PARAMS, **ENV_PARAMS}

# ── Helper functions for GUI ──────────────────────────────────────────


def get_rocket_sliders() -> list[tuple[str, str, float, float, float]]:
    """Return rocket parameter sliders in GUI format: (label, attr, min, max, step)."""
    return [(spec.label, attr, spec.min, spec.max, spec.step) for attr, spec in ROCKET_PARAMS.items()]


def get_env_sliders() -> list[tuple[str, str, float, float, float]]:
    """Return environment parameter sliders in GUI format: (label, attr, min, max, step)."""
    return [(spec.label, attr, spec.min, spec.max, spec.step) for attr, spec in ENV_PARAMS.items()]


def get_sensor_rate_sliders() -> list[tuple[str, str, float, float, float]]:
    """Return sensor rate sliders in GUI format: (label, attr, min, max, step)."""
    return [(spec.label, attr, spec.min, spec.max, spec.step) for attr, spec in SENSOR_RATE_PARAMS.items()]


def get_sensor_latency_sliders() -> list[tuple[str, str, float, float, float]]:
    """Return sensor latency sliders in GUI format: (label, attr, min, max, step)."""
    return [(spec.label, attr, spec.min, spec.max, spec.step) for attr, spec in SENSOR_LATENCY_PARAMS.items()]


# ── Helper functions for CLI ──────────────────────────────────────────


def get_param_ranges() -> dict[str, tuple[float, float]]:
    """Return parameter ranges in CLI format: {attr: (min, max)}."""
    return {attr: (spec.min, spec.max) for attr, spec in ALL_PARAMS.items()}
