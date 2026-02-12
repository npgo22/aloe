use crate::sensor::SensorData;
use crate::sim::SimResult;
use aloe_core::eskf::{EskfTuning, RocketEsKf};
use aloe_core::state_machine::{StateInput, StateMachine};
use nalgebra::Vector3;

/// Struct to hold the output of the filter simulation.
#[derive(Clone)]
pub struct FilterResult {
    pub time: Vec<f64>,
    pub position: Vec<Vector3<f64>>,
    pub velocity: Vec<Vector3<f64>>,
    pub orientation_euler: Vec<Vector3<f64>>,
}

/// Filter configuration with per-stage tuning.
#[derive(Clone)]
pub struct FilterConfig {
    pub ground_pressure_mbar: f64,
    pub mag_declination_deg: f64,
    pub mag_dip_deg: f64,
    pub home_lat_deg: f64,
    pub home_lon_deg: f64,
    pub home_alt_m: f64,
    pub accel_noise_density: Vec<f64>,
    pub gyro_noise_density: Vec<f64>,
    pub accel_bias_instability: Vec<f64>,
    pub gyro_bias_instability: Vec<f64>,
    pub pos_process_noise: Vec<f64>,
    pub r_gps_pos: Vec<f64>,
    pub r_gps_vel: Vec<f64>,
    pub r_baro: Vec<f64>,
    pub r_mag: Vec<f64>,
}

impl FilterConfig {
    pub fn get_stage_tuning(&self, stage_idx: usize) -> EskfTuning {
        EskfTuning {
            accel_noise_density: self.accel_noise_density[stage_idx],
            gyro_noise_density: self.gyro_noise_density[stage_idx],
            accel_bias_instability: self.accel_bias_instability[stage_idx],
            gyro_bias_instability: self.gyro_bias_instability[stage_idx],
            pos_process_noise: self.pos_process_noise[stage_idx],
            r_gps_pos: self.r_gps_pos[stage_idx],
            r_gps_vel: self.r_gps_vel[stage_idx],
            r_baro: self.r_baro[stage_idx],
            r_mag: self.r_mag[stage_idx],
        }
    }

    pub fn get_stage_param(&self, stage_idx: usize, param_name: &str) -> f64 {
        match param_name {
            "accel_noise_density" => self.accel_noise_density[stage_idx],
            "gyro_noise_density" => self.gyro_noise_density[stage_idx],
            "accel_bias_instability" => self.accel_bias_instability[stage_idx],
            "gyro_bias_instability" => self.gyro_bias_instability[stage_idx],
            "pos_process_noise" => self.pos_process_noise[stage_idx],
            "r_gps_pos" => self.r_gps_pos[stage_idx],
            "r_gps_vel" => self.r_gps_vel[stage_idx],
            "r_baro" => self.r_baro[stage_idx],
            "r_mag" => self.r_mag[stage_idx],
            _ => panic!("Unknown param: {}", param_name),
        }
    }

    pub fn set_stage_param(&mut self, stage_idx: usize, param_name: &str, value: f64) {
        match param_name {
            "accel_noise_density" => self.accel_noise_density[stage_idx] = value,
            "gyro_noise_density" => self.gyro_noise_density[stage_idx] = value,
            "accel_bias_instability" => self.accel_bias_instability[stage_idx] = value,
            "gyro_bias_instability" => self.gyro_bias_instability[stage_idx] = value,
            "pos_process_noise" => self.pos_process_noise[stage_idx] = value,
            "r_gps_pos" => self.r_gps_pos[stage_idx] = value,
            "r_gps_vel" => self.r_gps_vel[stage_idx] = value,
            "r_baro" => self.r_baro[stage_idx] = value,
            "r_mag" => self.r_mag[stage_idx] = value,
            _ => panic!("Unknown param: {}", param_name),
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "ground_pressure_mbar": self.ground_pressure_mbar,
            "mag_declination_deg": self.mag_declination_deg,
            "mag_dip_deg": self.mag_dip_deg,
            "home_lat_deg": self.home_lat_deg,
            "home_lon_deg": self.home_lon_deg,
            "home_alt_m": self.home_alt_m,
            "accel_noise_density": self.accel_noise_density,
            "gyro_noise_density": self.gyro_noise_density,
            "accel_bias_instability": self.accel_bias_instability,
            "gyro_bias_instability": self.gyro_bias_instability,
            "pos_process_noise": self.pos_process_noise,
            "r_gps_pos": self.r_gps_pos,
            "r_gps_vel": self.r_gps_vel,
            "r_baro": self.r_baro,
            "r_mag": self.r_mag,
        })
    }
}

impl Default for FilterConfig {
    fn default() -> Self {
        use crate::params::{eskf_tuning_default, NUM_STAGES};
        let mut accel_noise_density = Vec::with_capacity(NUM_STAGES);
        let mut gyro_noise_density = Vec::with_capacity(NUM_STAGES);
        let mut accel_bias_instability = Vec::with_capacity(NUM_STAGES);
        let mut gyro_bias_instability = Vec::with_capacity(NUM_STAGES);
        let mut pos_process_noise = Vec::with_capacity(NUM_STAGES);
        let mut r_gps_pos = Vec::with_capacity(NUM_STAGES);
        let mut r_gps_vel = Vec::with_capacity(NUM_STAGES);
        let mut r_baro = Vec::with_capacity(NUM_STAGES);
        let mut r_mag = Vec::with_capacity(NUM_STAGES);
        for i in 0..NUM_STAGES {
            accel_noise_density.push(eskf_tuning_default("accel_noise_density", i).unwrap() as f64);
            gyro_noise_density.push(eskf_tuning_default("gyro_noise_density", i).unwrap() as f64);
            accel_bias_instability
                .push(eskf_tuning_default("accel_bias_instability", i).unwrap() as f64);
            gyro_bias_instability
                .push(eskf_tuning_default("gyro_bias_instability", i).unwrap() as f64);
            pos_process_noise.push(eskf_tuning_default("pos_process_noise", i).unwrap() as f64);
            r_gps_pos.push(eskf_tuning_default("r_gps_pos", i).unwrap() as f64);
            r_gps_vel.push(eskf_tuning_default("r_gps_vel", i).unwrap() as f64);
            r_baro.push(eskf_tuning_default("r_baro", i).unwrap() as f64);
            r_mag.push(eskf_tuning_default("r_mag", i).unwrap() as f64);
        }
        Self {
            ground_pressure_mbar: 1013.25,
            mag_declination_deg: 0.0,
            mag_dip_deg: 60.0,
            home_lat_deg: 35.0,
            home_lon_deg: -106.0,
            home_alt_m: 1500.0,
            accel_noise_density,
            gyro_noise_density,
            accel_bias_instability,
            gyro_bias_instability,
            pos_process_noise,
            r_gps_pos,
            r_gps_vel,
            r_baro,
            r_mag,
        }
    }
}

/// Run the ESKF against the generated sensor data.
pub fn run_filter(
    _sim_result: &SimResult,
    sensor_data: &SensorData,
    config: &FilterConfig,
) -> FilterResult {
    // Initialize Filter with pad tuning
    let mut tuning = config.get_stage_tuning(0);
    let mut eskf = RocketEsKf::new(
        (config.ground_pressure_mbar * 100.0) as f32,
        config.mag_declination_deg as f32,
        config.mag_dip_deg as f32,
        tuning,
    );
    eskf.set_home_location(
        config.home_lat_deg as f32,
        config.home_lon_deg as f32,
        config.home_alt_m as f32,
    );

    let mut state_machine =
        StateMachine::new(aloe_core::state_machine::StateMachineConfig::default());

    // Outputs
    let mut time_out = Vec::new();
    let mut pos_out = Vec::new();
    let mut vel_out = Vec::new();
    let mut att_euler_out = Vec::new();

    let n = sensor_data.time.len();

    for i in 0..n {
        let t = sensor_data.time[i];
        let t_us = (t * 1_000_000.0) as u64;

        // -------------------------------------------------------------------
        // PREDICT (IMU)
        // -------------------------------------------------------------------
        let accel = sensor_data.accel_meas[i];
        let gyro = sensor_data.gyro_meas[i];

        // Cast f64 sensor data to f32 for the filter
        let accel_f32 = Vector3::new(accel.x as f32, accel.y as f32, accel.z as f32);
        let gyro_f32 = Vector3::new(gyro.x as f32, gyro.y as f32, gyro.z as f32);

        eskf.predict(Some(gyro_f32), Some(accel_f32), Some(accel_f32), t_us);

        // -------------------------------------------------------------------
        // UPDATES (Mag, Baro, GPS)
        // -------------------------------------------------------------------

        // Mag
        let mag = sensor_data.mag_meas[i];
        if mag.x.is_finite() {
            let mag_f32 = Vector3::new(mag.x as f32, mag.y as f32, mag.z as f32);
            eskf.update_mag(mag_f32);
        }

        // Baro
        let pressure = sensor_data.baro_pressure[i];
        if pressure.is_finite() {
            eskf.update_baro(pressure as f32);
        }

        // GPS
        let gps_pos = sensor_data.gps_pos[i];
        let gps_vel = sensor_data.gps_vel[i];
        if gps_pos.x.is_finite() {
            // Approx conversion: 1 deg lat ~ 111,111 m
            let lat = gps_pos.x / 111111.0;
            let lon = gps_pos.y / (111111.0 * 1.0_f64.cos());
            let alt = -gps_pos.z; // GPS Altitude usually noisy

            let lat_deg = config.home_lat_deg + lat;
            let lon_deg = config.home_lon_deg + lon;
            let alt_msl = config.home_alt_m + alt;

            let vel_f32 = Vector3::new(gps_vel.x as f32, gps_vel.y as f32, gps_vel.z as f32);

            eskf.update_gps(
                lat_deg as f32,
                lon_deg as f32,
                alt_msl as f32,
                vel_f32,
                t_us,
            );
        }

        // Step the state machine and switch tuning
        let dt = if i > 0 {
            t - sensor_data.time[i - 1]
        } else {
            0.0
        };
        // Compute vertical acceleration (NED down = positive) using finite difference
        // accel_down = d/dt(velocity_down). Use previous filter velocity if available.
        let accel_down = if i > 0 {
            let prev_vz = vel_out
                .last()
                .map(|v: &Vector3<f64>| v.z as f32)
                .unwrap_or(eskf.state.velocity.z as f32);
            let dt_f32 = if dt > 0.0 { dt as f32 } else { 0.0 };
            if dt_f32 > 0.0 {
                (eskf.state.velocity.z as f32 - prev_vz) / dt_f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        let input = StateInput {
            time: t as f32,
            altitude: -eskf.state.position.z as f32, // AGL up
            velocity_down: eskf.state.velocity.z as f32, // NED D
            accel_down,
        };
        let flight_state = state_machine.update(input, dt as f32);
        let tuning_stage = match flight_state {
            aloe_core::state_machine::FlightState::Pad => 0,
            aloe_core::state_machine::FlightState::Ascent => 1,
            aloe_core::state_machine::FlightState::Coast => 2,
            aloe_core::state_machine::FlightState::Descent => 3,
            aloe_core::state_machine::FlightState::Landed => 3,
        };
        let new_tuning = config.get_stage_tuning(tuning_stage);
        if new_tuning != tuning {
            tuning = new_tuning;
            eskf.tuning = tuning;
        }

        // -------------------------------------------------------------------
        // RECORD STATE
        // -------------------------------------------------------------------
        time_out.push(t);

        let p = eskf.state.position;
        pos_out.push(Vector3::new(p.x, p.y, p.z));

        let v = eskf.state.velocity;
        vel_out.push(Vector3::new(v.x, v.y, v.z));

        // Euler angles
        let (roll, pitch, yaw) = eskf.state.orientation.euler_angles();
        att_euler_out.push(Vector3::new(roll, pitch, yaw));
    }

    FilterResult {
        time: time_out,
        position: pos_out,
        velocity: vel_out,
        orientation_euler: att_euler_out,
    }
}

/// Legacy function for backward compatibility.
pub fn run_filter_default(sim_result: &SimResult, sensor_data: &SensorData) -> FilterResult {
    run_filter(sim_result, sensor_data, &FilterConfig::default())
}
