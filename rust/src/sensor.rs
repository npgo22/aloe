//! Artificial sensor data generation.
//!
//! Overlays noisy sensor readings onto a rocket trajectory at each sensor's
//! sample rate.  Non-sample rows are NaN.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

// ISA (International Standard Atmosphere) model constants
const G0: f64 = 9.80665; // m/s^2
const R: f64 = 287.05; // J/(kg·K) specific gas constant for air

/// ISA atmospheric layer: (base_altitude_m, base_temp_K, lapse_rate_K_per_m, base_pressure_Pa)
/// Base pressures are pre-computed for each layer boundary.
const ISA_LAYERS: &[(f64, f64, f64, f64)] = &[
    (0.0, 288.15, -0.0065, 101325.0), // Troposphere: 0-11km
    (11000.0, 216.65, 0.0, 22632.0),  // Tropopause: 11-20km (isothermal)
    (20000.0, 216.65, 0.001, 5474.8), // Stratosphere 1: 20-32km
    (32000.0, 228.65, 0.0028, 868.0), // Stratosphere 2: 32-47km
    (47000.0, 270.65, 0.0, 110.9),    // Stratopause: 47-51km (isothermal)
];

/// Calculate pressure at given altitude using full ISA model.
/// Valid from sea level to ~51km.
fn isa_pressure_at_altitude(alt_m: f64) -> f64 {
    let alt_m = alt_m.max(0.0);

    // Find which layer we're in
    for i in 0..ISA_LAYERS.len() {
        let (base_alt, base_temp, lapse, base_pressure) = ISA_LAYERS[i];
        let next_base_alt = if i + 1 < ISA_LAYERS.len() {
            ISA_LAYERS[i + 1].0
        } else {
            51000.0
        };

        if alt_m <= next_base_alt {
            let layer_alt = alt_m - base_alt;

            if lapse.abs() < 1e-10 {
                // Isothermal layer: p = p_base * exp(-g0 * h / (R * T))
                return base_pressure * (-G0 * layer_alt / (R * base_temp)).exp();
            } else {
                // Lapse rate layer: p = p_base * (T / T_base)^(g0 / (R * L))
                let t = base_temp + lapse * layer_alt;
                let exponent = -G0 / (R * lapse);
                return base_pressure * (t / base_temp).powf(exponent);
            }
        }
    }

    // Above defined layers, return pressure at top of last layer
    110.9 // Pressure at 47km
}

/// Check if time `t` falls on a sensor sample instant.
#[inline]
fn is_sample(t: f64, hz: f64, latency_s: f64) -> bool {
    if t < latency_s {
        return false;
    }
    let period = 1.0 / hz;
    let rem = (t - latency_s) % period;
    rem < 0.005 || (period - rem) < 0.005
}

/// Sample truth + Gaussian noise at the sensor's rate; NaN elsewhere.
fn sensor_col(
    times: &[f64],
    truth: &[f64],
    sigma: f64,
    hz: f64,
    latency_ms: f64,
    rng: &mut rand::rngs::StdRng,
) -> Vec<f64> {
    let lat_s = latency_ms / 1000.0;
    let dist = Normal::new(0.0, sigma).unwrap();
    times
        .iter()
        .zip(truth)
        .map(|(&t, &v)| {
            if is_sample(t, hz, lat_s) {
                v + dist.sample(rng)
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Like `sensor_col` but saturates output to ±`range_ms2` (m/s²), modelling
/// the hard clipping that occurs when acceleration exceeds the sensor's
/// measurement range (e.g. BMI088 ±24g, ADXL375 ±200g).
fn sensor_col_saturating(
    times: &[f64],
    truth: &[f64],
    sigma: f64,
    hz: f64,
    latency_ms: f64,
    range_ms2: f64,
    rng: &mut rand::rngs::StdRng,
) -> Vec<f64> {
    let lat_s = latency_ms / 1000.0;
    let dist = Normal::new(0.0, sigma).unwrap();
    times
        .iter()
        .zip(truth)
        .map(|(&t, &v)| {
            if is_sample(t, hz, lat_s) {
                (v + dist.sample(rng)).clamp(-range_ms2, range_ms2)
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// PyO3 entry point — generates all sensor columns from trajectory arrays.
#[pyfunction]
#[pyo3(signature = (
    time_s,
    acceleration_x_ms2,
    acceleration_y_ms2,
    acceleration_z_ms2,
    velocity_x_ms,
    velocity_y_ms,
    velocity_z_ms,
    altitude_m,
    position_x_m,
    position_z_m,
    spin_rate_dps = 0.0,
    noise_scale = 1.0,
    seed = 42,
    bmi088_accel = true,
    bmi088_gyro = true,
    adxl375 = true,
    ms5611 = true,
    lis3mdl = true,
    lc29h = true,
    bmi088_accel_hz = 800.0,
    bmi088_gyro_hz = 1000.0,
    adxl375_hz = 800.0,
    ms5611_hz = 50.0,
    lis3mdl_hz = 80.0,
    lc29h_hz = 10.0,
    bmi088_accel_latency_ms = 1.0,
    bmi088_gyro_latency_ms = 55.0,
    adxl375_latency_ms = 0.3,
    ms5611_latency_ms = 8.22,
    lis3mdl_latency_ms = 40.0,
    lc29h_latency_ms = 1000.0,
))]
#[allow(clippy::too_many_arguments)]
pub fn add_sensor_data_rs(
    py: Python<'_>,
    time_s: Vec<f64>,
    acceleration_x_ms2: Vec<f64>,
    acceleration_y_ms2: Vec<f64>,
    acceleration_z_ms2: Vec<f64>,
    velocity_x_ms: Vec<f64>,
    velocity_y_ms: Vec<f64>,
    velocity_z_ms: Vec<f64>,
    altitude_m: Vec<f64>,
    position_x_m: Vec<f64>,
    position_z_m: Vec<f64>,
    spin_rate_dps: f64,
    noise_scale: f64,
    seed: u64,
    bmi088_accel: bool,
    bmi088_gyro: bool,
    adxl375: bool,
    ms5611: bool,
    lis3mdl: bool,
    lc29h: bool,
    bmi088_accel_hz: f64,
    bmi088_gyro_hz: f64,
    adxl375_hz: f64,
    ms5611_hz: f64,
    lis3mdl_hz: f64,
    lc29h_hz: f64,
    bmi088_accel_latency_ms: f64,
    bmi088_gyro_latency_ms: f64,
    adxl375_latency_ms: f64,
    ms5611_latency_ms: f64,
    lis3mdl_latency_ms: f64,
    lc29h_latency_ms: f64,
) -> PyResult<Py<PyDict>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = time_s.len();
    let ns = noise_scale;
    let dt = 0.01_f64;
    let dict = PyDict::new(py);

    // BMI088 low-g accel — ±24 g range (saturates at ±235.44 m/s²)
    if bmi088_accel {
        let s = 175e-6 * 9.81 * bmi088_accel_hz.sqrt() * ns;
        let range = 24.0 * 9.81; // ±24 g in m/s²
        dict.set_item(
            "bmi088_accel_x_ms2",
            sensor_col_saturating(
                &time_s,
                &acceleration_x_ms2,
                s,
                bmi088_accel_hz,
                bmi088_accel_latency_ms,
                range,
                &mut rng,
            ),
        )?;
        dict.set_item(
            "bmi088_accel_y_ms2",
            sensor_col_saturating(
                &time_s,
                &acceleration_y_ms2,
                s,
                bmi088_accel_hz,
                bmi088_accel_latency_ms,
                range,
                &mut rng,
            ),
        )?;
        dict.set_item(
            "bmi088_accel_z_ms2",
            sensor_col_saturating(
                &time_s,
                &acceleration_z_ms2,
                s,
                bmi088_accel_hz,
                bmi088_accel_latency_ms,
                range,
                &mut rng,
            ),
        )?;
    }

    // BMI088 gyro
    if bmi088_gyro {
        let s = 0.014 * bmi088_gyro_hz.sqrt() * ns;
        let mut gyro_x = vec![0.0; n];
        let mut gyro_y = vec![0.0; n];
        let gyro_z = vec![spin_rate_dps; n];
        let mut pitch = vec![0.0; n];
        let mut yaw = vec![0.0; n];
        let mut speed = vec![0.0; n];
        for i in 0..n {
            let (vx, vy, vz) = (velocity_x_ms[i], velocity_y_ms[i], velocity_z_ms[i]);
            speed[i] = (vx * vx + vy * vy + vz * vz).sqrt();
            pitch[i] = vy.atan2((vx * vx + vz * vz).sqrt()).to_degrees();
            yaw[i] = vz.atan2(vx).to_degrees();
        }
        if n >= 2 {
            gyro_x[0] = (pitch[1] - pitch[0]) / dt;
            gyro_y[0] = (yaw[1] - yaw[0]) / dt;
            for i in 1..n - 1 {
                gyro_x[i] = (pitch[i + 1] - pitch[i - 1]) / (2.0 * dt);
                gyro_y[i] = (yaw[i + 1] - yaw[i - 1]) / (2.0 * dt);
            }
            gyro_x[n - 1] = (pitch[n - 1] - pitch[n - 2]) / dt;
            gyro_y[n - 1] = (yaw[n - 1] - yaw[n - 2]) / dt;
        }
        for i in 0..n {
            let alpha = ((speed[i] - 5.0) / 10.0).clamp(0.0, 1.0);
            gyro_x[i] *= alpha;
            gyro_y[i] *= alpha;
        }
        dict.set_item(
            "bmi088_gyro_x_dps",
            sensor_col(
                &time_s,
                &gyro_x,
                s,
                bmi088_gyro_hz,
                bmi088_gyro_latency_ms,
                &mut rng,
            ),
        )?;
        dict.set_item(
            "bmi088_gyro_y_dps",
            sensor_col(
                &time_s,
                &gyro_y,
                s,
                bmi088_gyro_hz,
                bmi088_gyro_latency_ms,
                &mut rng,
            ),
        )?;
        dict.set_item(
            "bmi088_gyro_z_dps",
            sensor_col(
                &time_s,
                &gyro_z,
                s,
                bmi088_gyro_hz,
                bmi088_gyro_latency_ms,
                &mut rng,
            ),
        )?;
    }

    // ADXL375 high-g accel — ±200 g range (saturates at ±1962 m/s²)
    // Only active when total acceleration > 50g to avoid noise during low-g phases
    if adxl375 {
        let s = 3.9e-3 * 9.81 * adxl375_hz.sqrt() * ns;
        let range = 200.0 * 9.81; // ±200 g in m/s²
        let threshold = 50.0 * 9.81; // 50g threshold for activation

        // Calculate acceleration magnitude
        let accel_magnitude: Vec<f64> = acceleration_x_ms2
            .iter()
            .zip(&acceleration_y_ms2)
            .zip(&acceleration_z_ms2)
            .map(|((ax, ay), az)| (ax * ax + ay * ay + az * az).sqrt())
            .collect();

        // Get raw sensor data
        let mut adxl_x = sensor_col_saturating(
            &time_s,
            &acceleration_x_ms2,
            s,
            adxl375_hz,
            adxl375_latency_ms,
            range,
            &mut rng,
        );
        let mut adxl_y = sensor_col_saturating(
            &time_s,
            &acceleration_y_ms2,
            s,
            adxl375_hz,
            adxl375_latency_ms,
            range,
            &mut rng,
        );
        let mut adxl_z = sensor_col_saturating(
            &time_s,
            &acceleration_z_ms2,
            s,
            adxl375_hz,
            adxl375_latency_ms,
            range,
            &mut rng,
        );

        // Filter: only output when acceleration is high enough
        for i in 0..n {
            if accel_magnitude[i] < threshold {
                adxl_x[i] = f64::NAN;
                adxl_y[i] = f64::NAN;
                adxl_z[i] = f64::NAN;
            }
        }

        dict.set_item("adxl375_accel_x_ms2", adxl_x)?;
        dict.set_item("adxl375_accel_y_ms2", adxl_y)?;
        dict.set_item("adxl375_accel_z_ms2", adxl_z)?;
    }

    // MS5611 baro
    if ms5611 {
        let s = 0.024 * ns;
        // Use full ISA model for pressure calculation (valid up to ~50km)
        let pressure: Vec<f64> = altitude_m
            .iter()
            .map(|&h| isa_pressure_at_altitude(h) / 100.0) // Convert Pa to mbar
            .collect();
        dict.set_item(
            "ms5611_pressure_mbar",
            sensor_col(
                &time_s,
                &pressure,
                s,
                ms5611_hz,
                ms5611_latency_ms,
                &mut rng,
            ),
        )?;
    }

    // LIS3MDL mag
    if lis3mdl {
        let s = 0.00047 * ns;
        let mx = vec![0.2; n];
        let my = vec![-0.4; n];
        let mz = vec![0.1; n];
        dict.set_item(
            "lis3mdl_mag_x_gauss",
            sensor_col(&time_s, &mx, s, lis3mdl_hz, lis3mdl_latency_ms, &mut rng),
        )?;
        dict.set_item(
            "lis3mdl_mag_y_gauss",
            sensor_col(&time_s, &my, s, lis3mdl_hz, lis3mdl_latency_ms, &mut rng),
        )?;
        dict.set_item(
            "lis3mdl_mag_z_gauss",
            sensor_col(&time_s, &mz, s, lis3mdl_hz, lis3mdl_latency_ms, &mut rng),
        )?;
    }

    // LC29H GPS
    if lc29h {
        let ps = 1.5 / 1.1774 * ns;
        let vs = 0.05 * ns;
        dict.set_item(
            "gps_pos_x_m",
            sensor_col(
                &time_s,
                &position_x_m,
                ps,
                lc29h_hz,
                lc29h_latency_ms,
                &mut rng,
            ),
        )?;
        dict.set_item(
            "gps_pos_y_m",
            sensor_col(
                &time_s,
                &altitude_m,
                ps,
                lc29h_hz,
                lc29h_latency_ms,
                &mut rng,
            ),
        )?;
        dict.set_item(
            "gps_pos_z_m",
            sensor_col(
                &time_s,
                &position_z_m,
                ps,
                lc29h_hz,
                lc29h_latency_ms,
                &mut rng,
            ),
        )?;
        dict.set_item(
            "gps_vel_x_ms",
            sensor_col(
                &time_s,
                &velocity_x_ms,
                vs,
                lc29h_hz,
                lc29h_latency_ms,
                &mut rng,
            ),
        )?;
        dict.set_item(
            "gps_vel_y_ms",
            sensor_col(
                &time_s,
                &velocity_y_ms,
                vs,
                lc29h_hz,
                lc29h_latency_ms,
                &mut rng,
            ),
        )?;
        dict.set_item(
            "gps_vel_z_ms",
            sensor_col(
                &time_s,
                &velocity_z_ms,
                vs,
                lc29h_hz,
                lc29h_latency_ms,
                &mut rng,
            ),
        )?;
    }

    Ok(dict.into())
}
