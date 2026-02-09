import polars as pl
import numpy as np
import math
from dataclasses import dataclass, field


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
    launch_delay: float = 0.0  # Pre-launch idle time on the pad (s)
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

    Parameters
    ----------
    p : RocketParams
        All physical and environmental parameters for the simulation.
    """
    time_step = 0.01  # Integration time step (s)
    max_time = 300.0  # Maximum simulation duration (s)
    time_steps = int(max_time / time_step)

    # Arrays to store data
    times = []
    positions_x = []
    positions_y = []
    positions_z = []
    velocities_x = []
    velocities_y = []
    velocities_z = []
    accelerations_x = []
    accelerations_y = []
    accelerations_z = []
    masses = []
    drags = []
    thrusts = []

    # Initial conditions
    t = 0.0
    x = 0.0
    y = 0.0
    z = 0.0
    vx = 0.0
    vy = 0.0
    vz = 0.0

    for step in range(time_steps):
        # Time relative to ignition
        t_since_ignition = t - p.launch_delay

        # Pre-launch: rocket sits on the pad, no thrust, no movement
        on_pad = t_since_ignition < 0

        # Current mass
        if on_pad or t_since_ignition > p.burn_time:
            current_mass = p.dry_mass + (p.propellant_mass if on_pad else 0.0)
        else:
            mass_flow_rate = p.propellant_mass / p.burn_time
            current_mass = p.dry_mass + p.propellant_mass - (mass_flow_rate * t_since_ignition)

        # Thrust (only fires between ignition and ignition + burn_time)
        thrust_force = p.thrust if (0 <= t_since_ignition <= p.burn_time) else 0.0

        # Velocity magnitude
        v_total = math.sqrt(vx**2 + vy**2 + vz**2)

        # Drag force
        if v_total > 0:
            drag_force = 0.5 * p.air_density * v_total**2 * p.drag_coeff * p.ref_area
            drag_x = -drag_force * (vx / v_total)
            drag_y = -drag_force * (vy / v_total)
            drag_z = -drag_force * (vz / v_total)
        else:
            drag_force = 0.0
            drag_x = 0.0
            drag_y = 0.0
            drag_z = 0.0

        # Forces — thrust vector with cant angle rotating at spin_rate
        # The cant angle tilts thrust off the body axis; spin sweeps it
        # around in a cone, producing lateral forces that create a spiral.
        cant_rad = math.radians(p.thrust_cant)
        if thrust_force > 0 and cant_rad > 0 and p.spin_rate != 0:
            spin_rad_s = math.radians(p.spin_rate)
            phase = spin_rad_s * t_since_ignition
            thrust_y = thrust_force * math.cos(cant_rad)
            thrust_x = thrust_force * math.sin(cant_rad) * math.cos(phase)
            thrust_z = thrust_force * math.sin(cant_rad) * math.sin(phase)
        else:
            thrust_x = 0.0
            thrust_y = thrust_force
            thrust_z = 0.0
        wind_force_x = p.wind_speed * 0.5
        wind_force_y = 0.0
        wind_force_z = p.wind_speed_z * 0.5
        gravity_x = 0.0
        gravity_y = -current_mass * p.gravity
        gravity_z = 0.0

        # Total forces
        total_force_x = thrust_x + drag_x + wind_force_x + gravity_x
        total_force_y = thrust_y + drag_y + wind_force_y + gravity_y
        total_force_z = thrust_z + drag_z + wind_force_z + gravity_z

        # Accelerations — on the pad, forces are balanced (ground reaction)
        if on_pad:
            ax = 0.0
            ay = 0.0
            az = 0.0
            drag_force = 0.0
        else:
            ax = total_force_x / current_mass
            ay = total_force_y / current_mass
            az = total_force_z / current_mass

        # Store data
        times.append(t)
        positions_x.append(x)
        positions_y.append(y)
        positions_z.append(z)
        velocities_x.append(vx)
        velocities_y.append(vy)
        velocities_z.append(vz)
        accelerations_x.append(ax)
        accelerations_y.append(ay)
        accelerations_z.append(az)
        masses.append(current_mass)
        drags.append(drag_force)
        thrusts.append(thrust_force)

        # Update state — on the pad, no movement
        if not on_pad:
            vx += ax * time_step
            vy += ay * time_step
            vz += az * time_step
            x += vx * time_step
            y += vy * time_step
            z += vz * time_step

        # Stop if rocket hits ground (after launch)
        if y < 0 and t > p.launch_delay:
            break

        t += time_step

    # Create DataFrame
    df = pl.DataFrame(
        {
            "time_s": times,
            "position_x_m": positions_x,
            "altitude_m": positions_y,
            "position_z_m": positions_z,
            "velocity_x_ms": velocities_x,
            "velocity_y_ms": velocities_y,
            "velocity_z_ms": velocities_z,
            "acceleration_x_ms2": accelerations_x,
            "acceleration_y_ms2": accelerations_y,
            "acceleration_z_ms2": accelerations_z,
            "mass_kg": masses,
            "drag_force_N": drags,
            "thrust_N": thrusts,
            "spin_rate_dps": [p.spin_rate] * len(times),
        },
        strict=False,
    )

    df = df.with_columns(
        [
            (pl.col("velocity_x_ms") ** 2 + pl.col("velocity_y_ms") ** 2 + pl.col("velocity_z_ms") ** 2)
            .sqrt()
            .alias("velocity_total_ms"),
            (pl.col("acceleration_x_ms2") ** 2 + pl.col("acceleration_y_ms2") ** 2 + pl.col("acceleration_z_ms2") ** 2)
            .sqrt()
            .alias("acceleration_total_ms2"),
        ]
    )

    return df


def add_sensor_data(df: pl.DataFrame, sensor_cfg: SensorConfig) -> pl.DataFrame:
    """Overlay artificial sensor readings onto a simulation DataFrame.

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
    rng = np.random.default_rng(sensor_cfg.seed)
    dt = 0.01  # sim time step (must match simulate_rocket)
    n = len(df)
    times = df["time_s"].to_numpy()
    ns = sensor_cfg.noise_scale

    new_cols: dict[str, list[float | None]] = {}

    def _sample_mask(hz: float, latency_ms: float) -> np.ndarray:
        """Return a boolean mask where True = sensor produces a sample.

        The sensor begins producing data after `latency_ms` has elapsed
        and then fires every 1/hz seconds.
        """
        period = 1.0 / hz
        latency_s = latency_ms / 1000.0
        mask = np.zeros(n, dtype=bool)
        for i, t in enumerate(times):
            if t < latency_s:
                continue
            elapsed = t - latency_s
            # Allow sample if within half a sim-step of the nearest sample instant
            remainder = elapsed % period
            if remainder < dt / 2 or (period - remainder) < dt / 2:
                mask[i] = True
        return mask

    def _add_sensor_col(
        name: str,
        truth: np.ndarray,
        noise_sigma: float,
        hz: float,
        latency_ms: float,
    ) -> None:
        """Create one sensor column with noise + sample-rate + latency."""
        mask = _sample_mask(hz, latency_ms)
        noise = rng.normal(0.0, noise_sigma * ns, size=n)
        col: list[float | None] = [None] * n
        for i in range(n):
            if mask[i]:
                col[i] = float(truth[i] + noise[i])
        new_cols[name] = col

    # ── BMI088 low-g accelerometer (±24 g) ───────────────────────
    # Noise density 175 µg/√Hz → σ = 175e-6 * 9.81 * √(hz) m/s²
    # https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi088-ds001.pdf  (Table 7)
    if sensor_cfg.enabled.get("bmi088_accel", True):
        hz = sensor_cfg.bmi088_accel_hz
        lat = sensor_cfg.bmi088_accel_latency_ms
        sigma = 175e-6 * 9.81 * math.sqrt(hz)  # m/s²
        _add_sensor_col("bmi088_accel_x_ms2", df["acceleration_x_ms2"].to_numpy(), sigma, hz, lat)
        _add_sensor_col("bmi088_accel_y_ms2", df["acceleration_y_ms2"].to_numpy(), sigma, hz, lat)
        _add_sensor_col("bmi088_accel_z_ms2", df["acceleration_z_ms2"].to_numpy(), sigma, hz, lat)

    # ── BMI088 gyroscope (±2000 °/s) ─────────────────────────────
    # Noise density 0.014 °/s/√Hz → σ = 0.014 * √(hz) °/s
    # The sim doesn't track angular rate, so we synthesize it from
    # the change in velocity direction.  For a rigid 1-DoF rocket
    # this is effectively the pitch rate ≈ d(atan2(vy, vx))/dt.
    # https://www.bosch-sensortec.com/media/boschsensortec/downloads/datasheets/bst-bmi088-ds001.pdf  (Table 3)
    if sensor_cfg.enabled.get("bmi088_gyro", True):
        hz = sensor_cfg.bmi088_gyro_hz
        lat = sensor_cfg.bmi088_gyro_latency_ms
        sigma = 0.014 * math.sqrt(hz)  # °/s

        # Approximate angular rates from velocity changes
        vx = df["velocity_x_ms"].to_numpy()
        vy = df["velocity_y_ms"].to_numpy()
        vz = df["velocity_z_ms"].to_numpy()
        speed = np.sqrt(vx**2 + vy**2 + vz**2)

        # pitch rate ≈ d/dt(atan2(vy, vx)) in °/s
        pitch = np.degrees(np.arctan2(vy, np.sqrt(vx**2 + vz**2)))
        yaw = np.degrees(np.arctan2(vz, vx))
        # Roll rate from RocketParams.spin_rate (constant spin around body axis)
        roll_truth = np.full(n, df["spin_rate_dps"].to_numpy()[0] if "spin_rate_dps" in df.columns else 0.0)

        gyro_x = np.gradient(pitch, dt)  # pitch rate
        gyro_y = np.gradient(yaw, dt)  # yaw rate
        gyro_z = roll_truth  # roll rate = spin_rate

        # Near apogee (speed ≈ 0) the atan2 derivatives produce unrealistic
        # spikes because the direction of a near-zero velocity vector is
        # undefined.  Smoothly fade the computed rates to zero when speed is
        # below a threshold so the ESKF gets "no rotation" rather than garbage.
        FADE_LO = 5.0   # m/s — start fading rates
        FADE_HI = 15.0  # m/s — full-confidence rates
        alpha = np.clip((speed - FADE_LO) / (FADE_HI - FADE_LO), 0.0, 1.0)
        gyro_x = gyro_x * alpha
        gyro_y = gyro_y * alpha

        _add_sensor_col("bmi088_gyro_x_dps", gyro_x, sigma, hz, lat)
        _add_sensor_col("bmi088_gyro_y_dps", gyro_y, sigma, hz, lat)
        _add_sensor_col("bmi088_gyro_z_dps", gyro_z, sigma, hz, lat)

    # ── ADXL375 high-g accelerometer (±200 g) ────────────────────
    # Noise density 3.9 mg/√Hz → σ = 3.9e-3 * 9.81 * √(hz) m/s²
    # https://www.analog.com/media/en/technical-documentation/data-sheets/adxl375.pdf  (Specifications, page 3)
    if sensor_cfg.enabled.get("adxl375", True):
        hz = sensor_cfg.adxl375_hz
        lat = sensor_cfg.adxl375_latency_ms
        sigma = 3.9e-3 * 9.81 * math.sqrt(hz)  # m/s²
        _add_sensor_col("adxl375_accel_x_ms2", df["acceleration_x_ms2"].to_numpy(), sigma, hz, lat)
        _add_sensor_col("adxl375_accel_y_ms2", df["acceleration_y_ms2"].to_numpy(), sigma, hz, lat)
        _add_sensor_col("adxl375_accel_z_ms2", df["acceleration_z_ms2"].to_numpy(), sigma, hz, lat)

    # ── MS5611 barometric altimeter ──────────────────────────────
    # RMS noise at OSR=4096: 0.024 mbar ≈ 0.20 m altitude noise  (via barometric formula)
    # We convert altitude → pressure with a simple ISA model, add noise, keep as pressure.
    # https://www.te.com/commerce/DocumentDelivery/DDEController?Action=showdoc&DocId=Data+Sheet%7FMS5611-01BA03%7FB3%7Fpdf  (Table, page 3)
    if sensor_cfg.enabled.get("ms5611", True):
        hz = sensor_cfg.ms5611_hz
        lat = sensor_cfg.ms5611_latency_ms
        alt = df["altitude_m"].to_numpy()
        # ISA barometric formula: P = P0 * (1 - L*h/T0)^(g*M/(R*L))
        # P0=101325 Pa, L=0.0065 K/m, T0=288.15 K, g=9.80665, M=0.0289644, R=8.31447
        P0 = 101325.0
        L = 0.0065
        T0 = 288.15
        exponent = 9.80665 * 0.0289644 / (8.31447 * L)  # ≈ 5.2559
        pressure_pa = P0 * np.power(np.maximum(1.0 - L * alt / T0, 0.001), exponent)
        pressure_mbar = pressure_pa / 100.0
        noise_sigma_mbar = 0.024  # RMS noise from datasheet
        _add_sensor_col("ms5611_pressure_mbar", pressure_mbar, noise_sigma_mbar, hz, lat)

    # ── LIS3MDL magnetometer (±4 gauss) ──────────────────────────
    # RMS noise ~3.2 LSB → at ±4 gauss (6842 LSB/gauss) ≈ 0.00047 gauss
    # We model Earth's field as ~0.5 gauss pointing roughly north + down.
    # https://www.st.com/resource/en/datasheet/lis3mdl.pdf  (Table 3, page 9)
    if sensor_cfg.enabled.get("lis3mdl", True):
        hz = sensor_cfg.lis3mdl_hz
        lat = sensor_cfg.lis3mdl_latency_ms
        sigma = 0.00047  # gauss RMS
        # Simple Earth field model: Bx≈0.2, By≈-0.4 (downward), Bz≈0.1 gauss
        # (rough Northern-hemisphere mid-latitude values)
        mag_x = np.full(n, 0.2)
        mag_y = np.full(n, -0.4)
        mag_z = np.full(n, 0.1)
        _add_sensor_col("lis3mdl_mag_x_gauss", mag_x, sigma, hz, lat)
        _add_sensor_col("lis3mdl_mag_y_gauss", mag_y, sigma, hz, lat)
        _add_sensor_col("lis3mdl_mag_z_gauss", mag_z, sigma, hz, lat)

    # ── Quectel LC29H GPS/GNSS ───────────────────────────────────
    # Position accuracy 1.5 m CEP → σ ≈ 1.5 / 1.1774 ≈ 1.27 m per axis
    # Velocity accuracy 0.05 m/s
    # https://www.quectel.com/wp-content/uploads/2024/01/Quectel_LC29H_GNSS_Specification_V1.4.pdf  (Section 3)
    if sensor_cfg.enabled.get("lc29h", True):
        hz = sensor_cfg.lc29h_hz
        lat = sensor_cfg.lc29h_latency_ms
        pos_sigma = 1.5 / 1.1774  # CEP → 1-σ per axis ≈ 1.27 m
        vel_sigma = 0.05  # m/s
        _add_sensor_col("gps_pos_x_m", df["position_x_m"].to_numpy(), pos_sigma, hz, lat)
        _add_sensor_col("gps_pos_y_m", df["altitude_m"].to_numpy(), pos_sigma, hz, lat)
        _add_sensor_col("gps_pos_z_m", df["position_z_m"].to_numpy(), pos_sigma, hz, lat)
        _add_sensor_col("gps_vel_x_ms", df["velocity_x_ms"].to_numpy(), vel_sigma, hz, lat)
        _add_sensor_col("gps_vel_y_ms", df["velocity_y_ms"].to_numpy(), vel_sigma, hz, lat)
        _add_sensor_col("gps_vel_z_ms", df["velocity_z_ms"].to_numpy(), vel_sigma, hz, lat)

    # Build new columns into the DataFrame
    for col_name, col_data in new_cols.items():
        df = df.with_columns(pl.Series(name=col_name, values=col_data))

    return df
