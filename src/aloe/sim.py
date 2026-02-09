import polars as pl
from dataclasses import dataclass, field

from aloe.aloe_core import simulate_rocket_rs, add_sensor_data_rs


@dataclass
class RocketParams:
    """All inputs to the rocket flight simulation."""

    dry_mass: float = 50.0  # Mass of the rocket without propellant (kg)
    propellant_mass: float = 150.0  # Mass of the propellant (kg)
    thrust: float = 15000.0  # Engine thrust force (N)
    burn_time: float = 25.0  # Duration the engine fires (s)
    drag_coeff: float = 0.40  # Aerodynamic drag coefficient (dimensionless)
    ref_area: float = 0.03  # Reference cross-sectional area for drag (m²)
    gravity: float = 9.81  # Gravitational acceleration (m/s²)
    wind_speed: float = 3.0  # Wind speed along X axis (m/s)
    wind_speed_z: float = 0.0  # Crosswind speed along Z axis (m/s)
    air_density: float = 1.225  # Ambient air density (kg/m³)
    launch_delay: float = 1.0  # Pre-launch idle time on the pad (s)
    spin_rate: float = 0.0  # Rocket roll rate around longitudinal axis (°/s)
    thrust_cant: float = 0.0  # Thrust vector cant angle from longitudinal axis (°)


# ── Sensor specifications ────────────────────────────────────────────
# Each sensor has: sample rate (Hz), latency (ms), noise characteristics.
# Noise is modelled as additive white Gaussian noise (AWGN) with the
# noise-density figures from each datasheet.
#
# Sources:
#   BMI088 (gyro + low-g accel)
#     Datasheet: https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi088-ds001.pdf
#     Gyroscope noise density:        0.014 °/s/√Hz  (Table 3, page 12)
#     Accelerometer noise density:    175 µg/√Hz     (Table 7, page 17)
#     Gyro ODR max:                   2000 Hz        (Table 1, page 10)
#     Accel ODR max:                  1600 Hz        (Table 5, page 15)
#     Gyro start-up time:             55 ms          (Table 1, page 10)
#     Accel start-up time:            1 ms           (Table 5, page 15)
#
#   ADXL375 (high-g accel, ±200 g)
#     Datasheet: https://www.analog.com/media/en/technical-documentation/data-sheets/adxl375.pdf
#     Noise density:                  3.9 mg/√Hz     (Specifications, page 3)
#     Max ODR (SPI):                  3200 Hz        (Specifications, page 2)
#     Latency at 3200 Hz:             0.3 ms         (Specifications, page 3)
#
#   MS5611 (barometric altimeter)
#     Datasheet: https://www.te.com/commerce/DocumentDelivery/DDEController?Action=showdoc&DocId=Data+Sheet%7FMS5611-01BA03%7FB3%7Fpdf%7FEnglish%7FENG_DS_MS5611-01BA03_B3.pdf
#     Resolution (OSR=4096):          0.012 mbar     (Table, page 3)
#     ADC conversion time (OSR=4096): 8.22 ms        (Table, page 3)
#     Max sample rate ≈ 1/8.22ms:     ~122 Hz
#     RMS noise (OSR=4096):           0.024 mbar     (Table, page 3)
#
#   LIS3MDL (3-axis magnetometer)
#     Datasheet: https://www.st.com/resource/en/datasheet/lis3mdl.pdf
#     Max ODR (ultra-high-perf):      155 Hz         (Table 2, page 8)   [Note: 1000 Hz only for single-axis FAST_ODR]
#     Noise density (±4 gauss):       3.2 LSB RMS → ~0.0032 gauss RMS   (Table 3, page 9)
#     Turn-on time:                   40 ms          (Table 2, page 8)
#
#   Quectel LC29H (GPS/GNSS)
#     Datasheet: https://www.quectel.com/wp-content/uploads/2024/01/Quectel_LC29H_GNSS_Specification_V1.4.pdf
#     Position accuracy:              1.5 m CEP      (Section 3, page 10)
#     Velocity accuracy:              0.05 m/s       (Section 3, page 10)
#     Max update rate:                10 Hz           (Section 3, page 9)
#     TTFF (hot start):               1 s            (Section 3, page 9)
#     TTFF (cold start):              26 s           (Section 3, page 9)


@dataclass
class SensorConfig:
    """Configuration for artificial sensor data generation.

    noise_scale: global multiplier for all sensor noise (1.0 = datasheet values).
    seed:        RNG seed for reproducibility.
    enabled:     which sensor groups to include.
    """

    noise_scale: float = 1.0
    seed: int = 42
    enabled: dict[str, bool] = field(
        default_factory=lambda: {
            "bmi088_accel": True,  # BMI088 low-g accelerometer (3-axis, ±24 g)
            "bmi088_gyro": True,  # BMI088 gyroscope (3-axis, ±2000 °/s)
            "adxl375": True,  # ADXL375 high-g accelerometer (3-axis, ±200 g)
            "ms5611": True,  # MS5611 barometric altimeter
            "lis3mdl": True,  # LIS3MDL magnetometer (3-axis, ±4 gauss)
            "lc29h": True,  # Quectel LC29H GPS/GNSS
        }
    )

    # ── Per-sensor sample rates (Hz) ──────────────────────────────
    # Defaults reflect typical operational rates, not maximums.
    bmi088_accel_hz: float = 800.0  # Low-g accel ODR (max 1600 Hz)
    bmi088_gyro_hz: float = 1000.0  # Gyro ODR (max 2000 Hz)
    adxl375_hz: float = 800.0  # High-g accel ODR (max 3200 Hz)
    ms5611_hz: float = 50.0  # Baro sample rate (max ~122 Hz at OSR=4096)
    lis3mdl_hz: float = 80.0  # Magnetometer ODR (max 155 Hz UHP)
    lc29h_hz: float = 10.0  # GPS update rate (max 10 Hz)

    # ── Per-sensor latency (ms) ──────────────────────────────────
    bmi088_accel_latency_ms: float = 1.0  # Accel start-up time
    bmi088_gyro_latency_ms: float = 55.0  # Gyro start-up time
    adxl375_latency_ms: float = 0.3  # At max ODR
    ms5611_latency_ms: float = 8.22  # ADC conversion @ OSR=4096
    lis3mdl_latency_ms: float = 40.0  # Turn-on time
    lc29h_latency_ms: float = 1000.0  # TTFF hot start


# ── Preset configurations ────────────────────────────────────────────
PRESETS: dict[str, RocketParams] = {
    # 8 in rocket diameter -> .0324m^2
    # No idea what to set anything else to
    # (still need prop/vehicle data)
    "30 km (rip)": RocketParams(
        dry_mass=50,
        propellant_mass=150,
        thrust=15000,
        burn_time=25,
        drag_coeff=0.40,
        ref_area=0.0324,
        wind_speed=3,
        wind_speed_z=0,
    ),
    "12 km": RocketParams(
        dry_mass=30,
        propellant_mass=60,
        thrust=6000,
        burn_time=12,
        drag_coeff=0.45,
        ref_area=0.0324,
        wind_speed=5,
        wind_speed_z=2,
    ),
    "High-drag test": RocketParams(
        dry_mass=50,
        propellant_mass=150,
        thrust=15000,
        burn_time=25,
        drag_coeff=1.8,
        ref_area=0.324,
        wind_speed=0,
        wind_speed_z=0,
    ),
}


def simulate_rocket(p: RocketParams) -> pl.DataFrame:
    """Simulate rocket flight and return a Polars DataFrame of the trajectory.

    The physics integration runs entirely in Rust (``aloe_core.simulate_rocket_rs``).

    Parameters
    ----------
    p : RocketParams
        All physical and environmental parameters for the simulation.
    """
    data = simulate_rocket_rs(
        dry_mass=p.dry_mass,
        propellant_mass=p.propellant_mass,
        thrust=p.thrust,
        burn_time=p.burn_time,
        drag_coeff=p.drag_coeff,
        ref_area=p.ref_area,
        gravity=p.gravity,
        wind_speed=p.wind_speed,
        wind_speed_z=p.wind_speed_z,
        air_density=p.air_density,
        launch_delay=p.launch_delay,
        spin_rate=p.spin_rate,
        thrust_cant=p.thrust_cant,
    )
    return pl.DataFrame(data, strict=False)


def add_sensor_data(df: pl.DataFrame, sensor_cfg: SensorConfig) -> pl.DataFrame:
    """Overlay artificial sensor readings onto a simulation DataFrame.

    All sensor noise generation runs in Rust (``aloe_core.add_sensor_data_rs``).

    Each sensor is sampled at its own rate and only produces data at those
    sample instants; all other rows get null.  Noise is additive Gaussian
    using the datasheet noise-density figures, scaled by sensor_cfg.noise_scale.
    Latency is modelled as a fixed time-shift (the sensor column appears
    ``latency`` seconds *after* the real event).

    Parameters
    ----------
    df : pl.DataFrame
        Output of :func:`simulate_rocket`.
    sensor_cfg : SensorConfig
        Noise / sample-rate / latency knobs.
    """
    spin_rate = df["spin_rate_dps"][0] if "spin_rate_dps" in df.columns else 0.0

    sensor_data = add_sensor_data_rs(
        time_s=df["time_s"].to_list(),
        acceleration_x_ms2=df["acceleration_x_ms2"].to_list(),
        acceleration_y_ms2=df["acceleration_y_ms2"].to_list(),
        acceleration_z_ms2=df["acceleration_z_ms2"].to_list(),
        velocity_x_ms=df["velocity_x_ms"].to_list(),
        velocity_y_ms=df["velocity_y_ms"].to_list(),
        velocity_z_ms=df["velocity_z_ms"].to_list(),
        altitude_m=df["altitude_m"].to_list(),
        position_x_m=df["position_x_m"].to_list(),
        position_z_m=df["position_z_m"].to_list(),
        spin_rate_dps=float(spin_rate),
        noise_scale=sensor_cfg.noise_scale,
        seed=sensor_cfg.seed,
        bmi088_accel=sensor_cfg.enabled.get("bmi088_accel", True),
        bmi088_gyro=sensor_cfg.enabled.get("bmi088_gyro", True),
        adxl375=sensor_cfg.enabled.get("adxl375", True),
        ms5611=sensor_cfg.enabled.get("ms5611", True),
        lis3mdl=sensor_cfg.enabled.get("lis3mdl", True),
        lc29h=sensor_cfg.enabled.get("lc29h", True),
        bmi088_accel_hz=sensor_cfg.bmi088_accel_hz,
        bmi088_gyro_hz=sensor_cfg.bmi088_gyro_hz,
        adxl375_hz=sensor_cfg.adxl375_hz,
        ms5611_hz=sensor_cfg.ms5611_hz,
        lis3mdl_hz=sensor_cfg.lis3mdl_hz,
        lc29h_hz=sensor_cfg.lc29h_hz,
        bmi088_accel_latency_ms=sensor_cfg.bmi088_accel_latency_ms,
        bmi088_gyro_latency_ms=sensor_cfg.bmi088_gyro_latency_ms,
        adxl375_latency_ms=sensor_cfg.adxl375_latency_ms,
        ms5611_latency_ms=sensor_cfg.ms5611_latency_ms,
        lis3mdl_latency_ms=sensor_cfg.lis3mdl_latency_ms,
        lc29h_latency_ms=sensor_cfg.lc29h_latency_ms,
    )

    # Convert NaN → null for Polars (sensor columns use null for "no sample")
    for col_name, col_data in sensor_data.items():
        series = pl.Series(name=col_name, values=col_data)
        df = df.with_columns(
            pl.when(series.is_nan()).then(None).otherwise(series).alias(col_name)
        )

    return df
