use crate::sim::SimResult;
use nalgebra::Vector3;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal}; // Assuming sim is in the same crate

const GPS_LATENCY_S: f64 = 0.12;
const BARO_TIME_CONSTANT_S: f64 = 0.08;

/// Configuration for chaos/fault injection testing
#[derive(Debug, Clone)]
pub struct ChaosConfig {
    /// GPS dropout time range (start, end) in seconds - GPS returns None during this period
    pub gps_dropout_range: Option<(f64, f64)>,

    /// Accelerometer spike times with magnitude multiplier
    /// (time, multiplier) - at specified time, accel reading is multiplied
    pub accel_spikes: Vec<(f64, f64)>,

    /// Gyro drift injection (time, drift_rate_rad_per_s)
    /// Starting at time, adds cumulative drift to gyro readings
    pub gyro_drift: Vec<(f64, f64)>,

    /// Barometer spike times with pressure offset (Pa)
    /// (time, pressure_offset)
    pub baro_spikes: Vec<(f64, f64)>,

    /// Magnetometer interference times with field offset (Gauss)
    /// (time, offset_vector)
    pub mag_interference: Vec<(f64, Vector3<f64>)>,

    /// Random fault probability per sample (0.0 = never, 1.0 = always)
    /// When triggered, sensor returns None for that sample
    pub random_fault_prob: f64,
}

impl Default for ChaosConfig {
    fn default() -> Self {
        Self {
            gps_dropout_range: None,
            accel_spikes: vec![],
            gyro_drift: vec![],
            baro_spikes: vec![],
            mag_interference: vec![],
            random_fault_prob: 0.0,
        }
    }
}

impl ChaosConfig {
    /// Create a config with GPS dropout during ascent
    pub fn with_gps_dropout(start: f64, end: f64) -> Self {
        Self {
            gps_dropout_range: Some((start, end)),
            ..Default::default()
        }
    }

    /// Create a config with gyro drift starting at a given time
    pub fn with_gyro_drift(start_time: f64, drift_rate: f64) -> Self {
        Self {
            gyro_drift: vec![(start_time, drift_rate)],
            ..Default::default()
        }
    }

    /// Create a config with random sensor faults
    pub fn with_random_faults(probability: f64) -> Self {
        Self {
            random_fault_prob: probability,
            ..Default::default()
        }
    }
}

pub struct SensorConfig {
    pub noise_scale: f64,
    pub accel_noise_std: f64,   // m/s^2
    pub gyro_noise_std: f64,    // rad/s
    pub mag_noise_std: f64,     // Gauss
    pub baro_noise_std: f64,    // Pascals or meters
    pub gps_pos_noise_std: f64, // meters
    pub gps_vel_noise_std: f64, // m/s

    // Biases (Walking bias not implemented for simplicity, just static)
    pub accel_bias: Vector3<f64>,
    pub gyro_bias: Vector3<f64>,

    pub seed: u64,

    // Enable flags
    pub accel_enabled: bool,
    pub gyro_enabled: bool,
    pub mag_enabled: bool,
    pub baro_enabled: bool,
    pub gps_enabled: bool,

    // Sample rates (Hz)
    pub bmi088_accel_rate_hz: f64,
    pub bmi088_gyro_rate_hz: f64,
    pub adxl375_rate_hz: f64,
    pub lis3mdl_rate_hz: f64,
    pub ms5611_rate_hz: f64,
    pub gps_rate_hz: f64,

    // Saturation limits (realistic sensor ranges)
    pub accel_saturation: f64, // m/s² (e.g., 200.0 for BMI088)
    pub gyro_saturation: f64,  // rad/s (e.g., 34.9 for 2000 deg/s)

    /// Chaos/fault injection configuration
    pub chaos: ChaosConfig,
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            noise_scale: 1.0,
            accel_noise_std: 0.01,
            gyro_noise_std: 0.001,
            mag_noise_std: 0.001,
            baro_noise_std: 0.1,
            gps_pos_noise_std: 1.0,
            gps_vel_noise_std: 0.1,
            accel_bias: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
            seed: 42,
            accel_enabled: true,
            gyro_enabled: true,
            mag_enabled: true,
            baro_enabled: true,
            gps_enabled: true,
            bmi088_accel_rate_hz: 1000.0,
            bmi088_gyro_rate_hz: 1000.0,
            adxl375_rate_hz: 3200.0,
            lis3mdl_rate_hz: 100.0,
            ms5611_rate_hz: 50.0,
            gps_rate_hz: 10.0,
            accel_saturation: 200.0, // BMI088: ±200 m/s² (~20g)
            gyro_saturation: 34.9,   // BMI088: 2000 deg/s = 34.9 rad/s
            chaos: ChaosConfig::default(),
        }
    }
}

pub struct SensorData {
    pub time: Vec<f64>,
    pub accel_meas: Vec<Option<Vector3<f64>>>,
    pub bmi088_accel_meas: Vec<Option<Vector3<f64>>>,
    pub adxl375_accel_meas: Vec<Option<Vector3<f64>>>,
    pub gyro_meas: Vec<Option<Vector3<f64>>>,
    pub mag_meas: Vec<Option<Vector3<f64>>>,
    pub baro_pressure: Vec<Option<f64>>,
    pub gps_pos: Vec<Option<Vector3<f64>>>,
    pub gps_vel: Vec<Option<Vector3<f64>>>,
}

fn should_sample(t: f64, next_sample_time: &mut f64, rate_hz: f64) -> bool {
    if rate_hz <= 0.0 {
        return false;
    }

    let interval = 1.0 / rate_hz;
    if t + 1e-9 < *next_sample_time {
        return false;
    }

    while *next_sample_time <= t + 1e-9 {
        *next_sample_time += interval;
    }
    true
}

fn index_at_or_before_time(time: &[f64], target_time: f64, current_index: usize) -> usize {
    let search_hi = current_index.min(time.len().saturating_sub(1));
    if time.is_empty() || search_hi == 0 || target_time <= time[0] {
        return 0;
    }

    match time[..=search_hi].binary_search_by(|probe| probe.total_cmp(&target_time)) {
        Ok(i) => i,
        Err(0) => 0,
        Err(i) => i.saturating_sub(1),
    }
}

pub fn generate_sensor_data(sim: &SimResult, cfg: &SensorConfig) -> SensorData {
    let mut rng = StdRng::seed_from_u64(cfg.seed);

    let n = sim.time.len();
    let mut data = SensorData {
        time: sim.time.clone(),
        accel_meas: Vec::with_capacity(n),
        bmi088_accel_meas: Vec::with_capacity(n),
        adxl375_accel_meas: Vec::with_capacity(n),
        gyro_meas: Vec::with_capacity(n),
        mag_meas: Vec::with_capacity(n),
        baro_pressure: Vec::with_capacity(n),
        gps_pos: Vec::with_capacity(n),
        gps_vel: Vec::with_capacity(n),
    };

    // Distributions
    let d_accel = Normal::new(0.0, cfg.noise_scale * cfg.accel_noise_std).unwrap();
    let d_gyro = Normal::new(0.0, cfg.noise_scale * cfg.gyro_noise_std).unwrap();
    let d_mag = Normal::new(0.0, cfg.noise_scale * cfg.mag_noise_std).unwrap();
    let d_baro = Normal::new(0.0, cfg.noise_scale * cfg.baro_noise_std).unwrap();
    let d_gps_p = Normal::new(0.0, cfg.noise_scale * cfg.gps_pos_noise_std).unwrap();
    let d_gps_v = Normal::new(0.0, cfg.noise_scale * cfg.gps_vel_noise_std).unwrap();

    // Distribution for random faults
    let d_uniform = rand_distr::Uniform::new(0.0, 1.0).unwrap();

    // Constant Field Definitions (NED)
    // Example: ~0.5 Gauss, dipping down (Northern hemisphere)
    let mag_field_ned = Vector3::new(0.25, 0.0, 0.45);

    // Track accumulated gyro drift
    let mut accumulated_gyro_drift: Vector3<f64> = Vector3::zeros();
    let mut next_bmi088_accel_time = 0.0;
    let mut next_bmi088_gyro_time = 0.0;
    let mut next_adxl375_time = 0.0;
    let mut next_lis3mdl_time = 0.0;
    let mut next_ms5611_time = 0.0;
    let mut next_gps_time = 0.0;
    let mut baro_state_pressure = 101325.0;

    for i in 0..n {
        let t = sim.time[i];
        let dt_s = if i > 0 {
            (sim.time[i] - sim.time[i - 1]).max(0.0)
        } else {
            0.0
        };

        // Update accumulated gyro drift based on chaos config
        for (drift_start, drift_rate) in &cfg.chaos.gyro_drift {
            if t >= *drift_start {
                // Add drift at each timestep (drift_rate is rad/s per axis)
                accumulated_gyro_drift.x += drift_rate * dt_s;
                accumulated_gyro_drift.y += drift_rate * dt_s;
                accumulated_gyro_drift.z += drift_rate * dt_s;
            }
        }

        // Check for random fault injection
        let random_fault = if cfg.chaos.random_fault_prob > 0.0 {
            d_uniform.sample(&mut rng) < cfg.chaos.random_fault_prob
        } else {
            false
        };

        // 1. IMU (Accelerometer)
        let mut bmi088_sample = None;
        let mut adxl375_sample = None;
        if cfg.accel_enabled && !random_fault {
            let proper_accel_true = sim.accel_body[i];

            let mut ax = proper_accel_true.x + cfg.accel_bias.x + d_accel.sample(&mut rng);
            let mut ay = proper_accel_true.y + cfg.accel_bias.y + d_accel.sample(&mut rng);
            let mut az = proper_accel_true.z + cfg.accel_bias.z + d_accel.sample(&mut rng);

            // Apply accel spikes from chaos config
            for (spike_time, multiplier) in &cfg.chaos.accel_spikes {
                if (t - spike_time).abs() < 0.01 {
                    // Within 10ms of spike time
                    ax *= multiplier;
                    ay *= multiplier;
                    az *= multiplier;
                }
            }

            // Apply saturation (clamp to sensor range)
            let ax_sat = ax.clamp(-cfg.accel_saturation, cfg.accel_saturation);
            let ay_sat = ay.clamp(-cfg.accel_saturation, cfg.accel_saturation);
            let az_sat = az.clamp(-cfg.accel_saturation, cfg.accel_saturation);

            let sample = Vector3::new(ax_sat, ay_sat, az_sat);
            if should_sample(t, &mut next_bmi088_accel_time, cfg.bmi088_accel_rate_hz) {
                bmi088_sample = Some(sample);
            }
            if should_sample(t, &mut next_adxl375_time, cfg.adxl375_rate_hz) {
                adxl375_sample = Some(sample);
            }
        }
        data.accel_meas.push(bmi088_sample.or(adxl375_sample));
        data.bmi088_accel_meas.push(bmi088_sample);
        data.adxl375_accel_meas.push(adxl375_sample);

        // 2. Gyroscope
        if cfg.gyro_enabled
            && !random_fault
            && should_sample(t, &mut next_bmi088_gyro_time, cfg.bmi088_gyro_rate_hz)
        {
            let gx: f64 = sim.ang_vel[i].x
                + cfg.gyro_bias.x
                + d_gyro.sample(&mut rng)
                + accumulated_gyro_drift.x;
            let gy: f64 = sim.ang_vel[i].y
                + cfg.gyro_bias.y
                + d_gyro.sample(&mut rng)
                + accumulated_gyro_drift.y;
            let gz: f64 = sim.ang_vel[i].z
                + cfg.gyro_bias.z
                + d_gyro.sample(&mut rng)
                + accumulated_gyro_drift.z;

            // Apply saturation (clamp to sensor range)
            let gx_sat = gx.clamp(-cfg.gyro_saturation, cfg.gyro_saturation);
            let gy_sat = gy.clamp(-cfg.gyro_saturation, cfg.gyro_saturation);
            let gz_sat = gz.clamp(-cfg.gyro_saturation, cfg.gyro_saturation);

            data.gyro_meas
                .push(Some(Vector3::new(gx_sat, gy_sat, gz_sat)));
        } else {
            data.gyro_meas.push(None);
        }

        // 3. Magnetometer
        if cfg.mag_enabled
            && !random_fault
            && should_sample(t, &mut next_lis3mdl_time, cfg.lis3mdl_rate_hz)
        {
            let mag_body = sim.orientation[i].inverse_transform_vector(&mag_field_ned);

            let mut mx = mag_body.x + d_mag.sample(&mut rng);
            let mut my = mag_body.y + d_mag.sample(&mut rng);
            let mut mz = mag_body.z + d_mag.sample(&mut rng);

            // Apply magnetometer interference from chaos config
            for (interference_time, offset) in &cfg.chaos.mag_interference {
                if t >= *interference_time {
                    mx += offset.x;
                    my += offset.y;
                    mz += offset.z;
                }
            }

            data.mag_meas.push(Some(Vector3::new(mx, my, mz)));
        } else {
            data.mag_meas.push(None);
        }

        // 4. Barometer (Pressure)
        if cfg.baro_enabled
            && !random_fault
            && should_sample(t, &mut next_ms5611_time, cfg.ms5611_rate_hz)
        {
            let true_alt = -sim.pos[i].z;
            let p0 = 101325.0; // Pa
            let h_scale = 8500.0; // m
            let static_pressure = p0 * (-true_alt / h_scale).exp();

            // Add dynamic pressure effect (pitot-static error)
            // At high velocities, airflow into the static port causes errors
            let v_mag = sim.vel[i].norm();
            let rho = p0 / (287.0 * 288.0) * (-true_alt / h_scale).exp(); // Air density
            let q_dynamic = 0.5 * rho * v_mag.powi(2); // Dynamic pressure

            // Model coupling of dynamic pressure into static port
            // Realistic coupling: ~5-15% depending on velocity and rocket geometry
            // At high speeds (>Mach 0.5), flow separation increases coupling
            let mach = v_mag / 343.0; // Speed of sound at sea level
            let coupling_factor = if mach < 0.3 {
                0.05 // Subsonic, good design
            } else if mach < 0.8 {
                0.05 + 0.15 * (mach - 0.3) / 0.5 // Transonic increase
            } else {
                0.20 // High transonic/supersonic regime
            };
            let pressure_error = q_dynamic * coupling_factor;

            // Velocity-dependent noise: vibration and turbulence increase with speed
            let velocity_noise_factor = 1.0 + (v_mag / 100.0).min(5.0);
            let noise_component = d_baro.sample(&mut rng) * velocity_noise_factor;

            // First-order barometer response lag
            let pressure_target = static_pressure + pressure_error;
            let alpha = if dt_s > 0.0 {
                1.0 - (-dt_s / BARO_TIME_CONSTANT_S).exp()
            } else {
                1.0
            };
            baro_state_pressure += alpha * (pressure_target - baro_state_pressure);

            let mut meas_pressure = baro_state_pressure + noise_component;

            // Apply barometer spikes from chaos config
            for (spike_time, pressure_offset) in &cfg.chaos.baro_spikes {
                if (t - spike_time).abs() < 0.1 {
                    // Within 100ms of spike time
                    meas_pressure += pressure_offset;
                }
            }

            data.baro_pressure.push(Some(meas_pressure));
        } else {
            data.baro_pressure.push(None);
        }

        // 5. GPS
        // Check for GPS dropout from chaos config
        let gps_in_dropout = cfg
            .chaos
            .gps_dropout_range
            .map(|(start, end)| t >= start && t <= end)
            .unwrap_or(false);

        if cfg.gps_enabled
            && !random_fault
            && !gps_in_dropout
            && should_sample(t, &mut next_gps_time, cfg.gps_rate_hz)
        {
            let gps_truth_index = index_at_or_before_time(&sim.time, t - GPS_LATENCY_S, i);
            let gps_pos_truth = sim.pos[gps_truth_index];
            let gps_vel_truth = sim.vel[gps_truth_index];

            let px = gps_pos_truth.x + d_gps_p.sample(&mut rng);
            let py = gps_pos_truth.y + d_gps_p.sample(&mut rng);
            let pz = gps_pos_truth.z + d_gps_p.sample(&mut rng); // GPS Altitude usually noisy
            data.gps_pos.push(Some(Vector3::new(px, py, pz)));

            let vx = gps_vel_truth.x + d_gps_v.sample(&mut rng);
            let vy = gps_vel_truth.y + d_gps_v.sample(&mut rng);
            let vz = gps_vel_truth.z + d_gps_v.sample(&mut rng);
            data.gps_vel.push(Some(Vector3::new(vx, vy, vz)));
        } else {
            data.gps_pos.push(None);
            data.gps_vel.push(None);
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::UnitQuaternion;

    fn simple_sim_result(
        time: Vec<f64>,
        pos: Vec<Vector3<f64>>,
        vel: Vec<Vector3<f64>>,
    ) -> SimResult {
        let n = time.len();
        SimResult {
            time,
            pos,
            vel,
            accel_body: vec![Vector3::zeros(); n],
            ang_vel: vec![Vector3::zeros(); n],
            orientation: vec![UnitQuaternion::identity(); n],
            ascent_time: None,
            coast_time: None,
            descent_time: None,
        }
    }

    #[test]
    fn gyro_drift_scales_with_sim_timestep() {
        let sim = simple_sim_result(
            vec![0.0, 0.1, 0.3],
            vec![Vector3::zeros(); 3],
            vec![Vector3::zeros(); 3],
        );

        let cfg = SensorConfig {
            noise_scale: 0.0,
            gyro_noise_std: 0.0,
            gyro_enabled: true,
            bmi088_gyro_rate_hz: 1000.0,
            chaos: ChaosConfig::with_gyro_drift(0.0, 1.0),
            ..SensorConfig::default()
        };

        let data = generate_sensor_data(&sim, &cfg);
        let g0 = data.gyro_meas[0].unwrap();
        let g1 = data.gyro_meas[1].unwrap();
        let g2 = data.gyro_meas[2].unwrap();

        assert!(g0.x.abs() < 1e-9);
        assert!((g1.x - 0.1).abs() < 1e-9);
        assert!((g2.x - 0.3).abs() < 1e-9);
    }

    #[test]
    fn gps_measurements_follow_latency_truth_history() {
        let time = vec![0.0, 0.1, 0.2, 0.3];
        let pos = time
            .iter()
            .map(|t| Vector3::new(10.0 * t, 0.0, 0.0))
            .collect::<Vec<_>>();
        let vel = vec![Vector3::new(10.0, 0.0, 0.0); time.len()];
        let sim = simple_sim_result(time, pos, vel);

        let cfg = SensorConfig {
            noise_scale: 0.0,
            gps_enabled: true,
            gps_rate_hz: 1000.0,
            gps_pos_noise_std: 0.0,
            gps_vel_noise_std: 0.0,
            ..SensorConfig::default()
        };

        let data = generate_sensor_data(&sim, &cfg);
        let x0 = data.gps_pos[0].unwrap().x;
        let x1 = data.gps_pos[1].unwrap().x;
        let x2 = data.gps_pos[2].unwrap().x;
        let x3 = data.gps_pos[3].unwrap().x;

        assert!((x0 - 0.0).abs() < 1e-9);
        assert!((x1 - 0.0).abs() < 1e-9);
        assert!((x2 - 0.0).abs() < 1e-9);
        assert!((x3 - 1.0).abs() < 1e-9);
    }
}
