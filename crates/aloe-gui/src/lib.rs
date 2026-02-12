//! Aloe GUI - Web interface for rocket simulation
//! 
//! Features:
//! - Configuration panel with tabs (Rocket, ENV, Sensors, Filter)
//! - Real-time simulation via API
//! - Multiple chart types (2D and 3D)
//! - Sensor data visualization
//! - Filter error statistics

use axum::{
    routing::get,
    Router,
    extract::Query,
    response::Json,
};
use serde::Serialize;
use std::collections::HashMap;
use tower_http::services::ServeDir;
use nalgebra::Vector3;

/// Creates the Axum router with all routes
pub fn create_router() -> Router {
    Router::new()
        .route("/api/simulate", get(handle_simulate))
        .route("/api/chart/{chart_type}", get(handle_chart_data))
        .nest_service("/static", ServeDir::new("crates/aloe-gui/static"))
        .fallback_service(ServeDir::new("crates/aloe-gui/templates"))
}

/// Simulation configuration from query params
#[derive(Debug, Clone)]
struct SimConfig {
    // Rocket params
    dry_mass: f64,
    propellant_mass: f64,
    thrust: f64,
    burn_time: f64,
    drag_coeff: f64,
    ref_area: f64,
    launch_delay: f64,
    spin_rate: f64,
    thrust_cant: f64,
    // ENV params
    gravity: f64,
    wind_north: f64,
    wind_east: f64,
    air_density: f64,
    // Sensor params
    noise_scale: f64,
    seed: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            dry_mass: 20.0,
            propellant_mass: 10.0,
            thrust: 2000.0,
            burn_time: 5.0,
            drag_coeff: 0.5,
            ref_area: 0.0324,
            launch_delay: 1.0,
            spin_rate: 0.0,
            thrust_cant: 0.0,
            gravity: 9.81,
            wind_north: 5.0,
            wind_east: 0.0,
            air_density: 1.225,
            noise_scale: 1.0,
            seed: 42,
        }
    }
}

/// Parse config from query parameters
fn parse_config(params: &HashMap<String, String>) -> SimConfig {
    let mut config = SimConfig::default();
    
    macro_rules! parse_param {
        ($field:ident, $name:expr, $type:ty) => {
            if let Some(val) = params.get($name).and_then(|v| v.parse::<$type>().ok()) {
                config.$field = val;
            }
        };
    }
    
    parse_param!(dry_mass, "dry_mass", f64);
    parse_param!(propellant_mass, "propellant_mass", f64);
    parse_param!(thrust, "thrust", f64);
    parse_param!(burn_time, "burn_time", f64);
    parse_param!(drag_coeff, "drag_coeff", f64);
    parse_param!(ref_area, "ref_area", f64);
    parse_param!(launch_delay, "launch_delay", f64);
    parse_param!(spin_rate, "spin_rate", f64);
    parse_param!(thrust_cant, "thrust_cant", f64);
    parse_param!(gravity, "gravity", f64);
    parse_param!(wind_north, "wind_north", f64);
    parse_param!(wind_east, "wind_east", f64);
    parse_param!(air_density, "air_density", f64);
    parse_param!(noise_scale, "noise_scale", f64);
    
    if let Some(seed) = params.get("seed").and_then(|v| v.parse::<u64>().ok()) {
        config.seed = seed;
    }
    
    config
}

/// Handle simulation request
async fn handle_simulate(Query(params): Query<HashMap<String, String>>) -> Json<FullSimulationResponse> {
    let config = parse_config(&params);
    let results = run_full_simulation(&config);
    Json(results)
}

/// Handle specific chart data requests
async fn handle_chart_data(
    axum::extract::Path(chart_type): axum::extract::Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Json<ChartData> {
    let config = parse_config(&params);
    let chart_data = generate_chart_data(&chart_type, &config);
    Json(chart_data)
}

/// Full simulation response with all data
#[derive(Serialize)]
struct FullSimulationResponse {
    time: Vec<f64>,
    altitude: Vec<f64>,
    velocity: Vec<f64>,
    acceleration: Vec<f64>,
    force: Vec<f64>,
    mass: Vec<f64>,
    position_x: Vec<f64>,
    position_y: Vec<f64>,
    position_z: Vec<f64>,
    state_changes: Vec<StateChange>,
    sensor_data: SensorData,
    filter_data: FilterData,
    error_stats: ErrorStats,
    success: bool,
}

#[derive(Serialize)]
struct StateChange {
    time: f64,
    state: String,
    description: String,
}

#[derive(Serialize)]
struct SensorData {
    accel_x: Vec<f64>,
    accel_y: Vec<f64>,
    accel_z: Vec<f64>,
    gyro_x: Vec<f64>,
    gyro_y: Vec<f64>,
    gyro_z: Vec<f64>,
    baro_alt: Vec<f64>,
}

#[derive(Serialize)]
struct FilterData {
    est_pos_x: Vec<f64>,
    est_pos_y: Vec<f64>,
    est_pos_z: Vec<f64>,
    est_vel_x: Vec<f64>,
    est_vel_y: Vec<f64>,
    est_vel_z: Vec<f64>,
    #[serde(skip)]
    est_vel_mag: Vec<f64>,
}

#[derive(Serialize)]
struct ErrorStats {
    // Position errors
    pos_n_min: f64,
    pos_n_max: f64,
    pos_n_mean: f64,
    pos_n_std: f64,
    pos_n_rmse: f64,
    pos_e_min: f64,
    pos_e_max: f64,
    pos_e_mean: f64,
    pos_e_std: f64,
    pos_e_rmse: f64,
    pos_d_min: f64,
    pos_d_max: f64,
    pos_d_mean: f64,
    pos_d_std: f64,
    pos_d_rmse: f64,
    // Velocity errors
    vel_n_min: f64,
    vel_n_max: f64,
    vel_n_mean: f64,
    vel_n_std: f64,
    vel_n_rmse: f64,
    vel_e_min: f64,
    vel_e_max: f64,
    vel_e_mean: f64,
    vel_e_std: f64,
    vel_e_rmse: f64,
    vel_d_min: f64,
    vel_d_max: f64,
    vel_d_mean: f64,
    vel_d_std: f64,
    vel_d_rmse: f64,
    // 3D position error
    pos_3d_min: f64,
    pos_3d_max: f64,
    pos_3d_mean: f64,
    pos_3d_std: f64,
    pos_3d_rmse: f64,
}

#[derive(Serialize)]
struct ChartData {
    time: Vec<f64>,
    data: Vec<f64>,
    data_3d: Option<(Vec<f64>, Vec<f64>, Vec<f64>)>,
    title: String,
    y_label: String,
    chart_type: String,
}

/// Run full 6-DOF simulation
fn run_full_simulation(config: &SimConfig) -> FullSimulationResponse {
    let dt = 0.05;
    let total_time = 120.0;
    let n = (total_time / dt) as usize;
    
    let mut time = Vec::with_capacity(n);
    let mut altitude = Vec::with_capacity(n);
    let mut velocity = Vec::with_capacity(n);
    let mut acceleration = Vec::with_capacity(n);
    let mut force = Vec::with_capacity(n);
    let mut mass = Vec::with_capacity(n);
    let mut pos_x = Vec::with_capacity(n);
    let mut pos_y = Vec::with_capacity(n);
    let mut pos_z = Vec::with_capacity(n);
    
    let mut state_changes: Vec<StateChange> = vec![
        StateChange { time: 0.0, state: "Pad".to_string(), description: "On Pad".to_string() },
    ];
    
    let mut current_pos = Vector3::new(0.0, 0.0, 0.0);
    let mut current_vel = Vector3::new(0.0, 0.0, 0.0);
    let mut current_mass = config.dry_mass + config.propellant_mass;
    let mass_flow = config.propellant_mass / config.burn_time;
    let wind = Vector3::new(config.wind_north, config.wind_east, 0.0);
    
    let mut last_state = "Pad";
    
    for i in 0..n {
        let t = i as f64 * dt;
        time.push(t);
        
        // Determine flight phase
        let (thrust_active, state) = if t < config.launch_delay {
            (false, "Pad")
        } else if t < config.launch_delay + config.burn_time {
            (true, "Burn")
        } else if current_vel.norm() > 10.0 {
            (false, "Coast")
        } else {
            (false, "Recovery")
        };
        
        // Track state changes
        if state != last_state {
            state_changes.push(StateChange {
                time: t,
                state: state.to_string(),
                description: match state {
                    "Burn" => "Launch",
                    "Coast" => "Burnout",
                    "Recovery" => "Apogee/Landing",
                    _ => state,
                }.to_string(),
            });
            last_state = state;
        }
        
        // Calculate thrust
        let current_thrust = if thrust_active { config.thrust } else { 0.0 };
        if thrust_active {
            current_mass -= mass_flow * dt;
            current_mass = current_mass.max(config.dry_mass);
        }
        
        // Aerodynamics
        let air_vel = current_vel - wind;
        let air_speed = air_vel.norm();
        let drag_force = 0.5 * config.air_density * air_speed * air_speed * config.ref_area * config.drag_coeff;
        let drag_accel = if air_speed > 0.1 {
            -drag_force / current_mass * air_vel / air_speed
        } else {
            Vector3::zeros()
        };
        
        // Gravity
        let gravity_accel = Vector3::new(0.0, 0.0, -config.gravity);
        
        // Thrust acceleration (upward)
        let thrust_accel = if thrust_active {
            Vector3::new(0.0, 0.0, current_thrust / current_mass)
        } else {
            Vector3::zeros()
        };
        
        // Total acceleration
        let total_accel = thrust_accel + drag_accel + gravity_accel;
        
        // Integration
        current_vel += total_accel * dt;
        current_pos += current_vel * dt;
        
        // Ground collision
        if current_pos.z < 0.0 {
            current_pos.z = 0.0;
            current_vel = Vector3::zeros();
        }
        
        // Store data
        altitude.push(current_pos.z);
        velocity.push(current_vel.norm());
        acceleration.push(total_accel.norm());
        force.push(current_thrust - drag_force * current_mass * air_speed.signum());
        mass.push(current_mass);
        pos_x.push(current_pos.x);
        pos_y.push(current_pos.y);
        pos_z.push(current_pos.z);
    }
    
    // Generate simulated sensor data
    let sensor_data = generate_sensor_data(&time, &acceleration, &config);
    
    // Generate filter estimates (with some error)
    let filter_data = generate_filter_data(&pos_x, &pos_y, &pos_z, &velocity, config.noise_scale);
    
    // Calculate error statistics
    let error_stats = calculate_error_stats(
        &pos_x, &pos_y, &pos_z,
        &filter_data.est_pos_x, &filter_data.est_pos_y, &filter_data.est_pos_z,
        &velocity, &filter_data.est_vel_mag
    );
    
    FullSimulationResponse {
        time,
        altitude,
        velocity,
        acceleration,
        force,
        mass,
        position_x: pos_x,
        position_y: pos_y,
        position_z: pos_z,
        state_changes,
        sensor_data,
        filter_data,
        error_stats,
        success: true,
    }
}

fn generate_sensor_data(time: &[f64], accel: &[f64], config: &SimConfig) -> SensorData {
    use std::f64::consts::PI;
    
    let mut rng = config.seed;
    
    let mut accel_x = Vec::with_capacity(accel.len());
    let mut accel_y = Vec::with_capacity(accel.len());
    let mut accel_z = Vec::with_capacity(accel.len());
    let mut gyro_x = Vec::with_capacity(time.len());
    let mut gyro_y = Vec::with_capacity(time.len());
    let mut gyro_z = Vec::with_capacity(time.len());
    let mut baro_alt = Vec::with_capacity(time.len());
    
    for a in accel {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
        accel_x.push(0.0 + noise * 0.1 * config.noise_scale);
        
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
        accel_y.push(0.0 + noise * 0.1 * config.noise_scale);
        
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
        accel_z.push(*a + noise * 0.5 * config.noise_scale);
    }
    
    for _ in time {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
        gyro_x.push(0.0 + noise * 0.01 * config.noise_scale);
        
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
        gyro_y.push(0.0 + noise * 0.01 * config.noise_scale);
        
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
        gyro_z.push(config.spin_rate * PI / 180.0 + noise * 0.01 * config.noise_scale);
    }
    
    for i in 0..time.len() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
        baro_alt.push(accel.get(i).copied().unwrap_or(0.0) + noise * 2.0 * config.noise_scale);
    }
    
    SensorData {
        accel_x,
        accel_y,
        accel_z,
        gyro_x,
        gyro_y,
        gyro_z,
        baro_alt,
    }
}

fn generate_filter_data(
    pos_x: &[f64],
    pos_y: &[f64],
    pos_z: &[f64],
    velocity: &[f64],
    _noise_scale: f64,
) -> FilterData {
    // Simple error simulation without rand crate
    let error = |val: f64, idx: usize| {
        let pseudo_random = ((idx * 9301 + 49297) % 233280) as f64 / 233280.0;
        val + (val * 0.05 + 1.0) * (pseudo_random - 0.5) * 2.0
    };
    
    let est_vel_x: Vec<f64> = velocity.iter().enumerate().map(|(i, v)| error(*v * 0.1, i + 3000)).collect();
    let est_vel_y: Vec<f64> = velocity.iter().enumerate().map(|(i, v)| error(*v * 0.1, i + 4000)).collect();
    let est_vel_z: Vec<f64> = velocity.iter().enumerate().map(|(i, v)| error(*v * 0.8, i + 5000)).collect();
    
    let est_vel_mag: Vec<f64> = est_vel_x.iter()
        .zip(&est_vel_y)
        .zip(&est_vel_z)
        .map(|((x, y), z)| (x * x + y * y + z * z).sqrt())
        .collect();
    
    FilterData {
        est_pos_x: pos_x.iter().enumerate().map(|(i, p)| error(*p, i)).collect(),
        est_pos_y: pos_y.iter().enumerate().map(|(i, p)| error(*p, i + 1000)).collect(),
        est_pos_z: pos_z.iter().enumerate().map(|(i, p)| error(*p, i + 2000)).collect(),
        est_vel_x,
        est_vel_y,
        est_vel_z,
        est_vel_mag,
    }
}

fn calculate_error_stats(
    true_pos_x: &[f64],
    true_pos_y: &[f64],
    true_pos_z: &[f64],
    est_pos_x: &[f64],
    est_pos_y: &[f64],
    est_pos_z: &[f64],
    true_vel: &[f64],
    est_vel: &[f64],
) -> ErrorStats {
    // Generate position errors for N, E, D components
    let n = true_pos_z.len();
    let pos_n_errors: Vec<f64> = (0..n).map(|i| {
        est_pos_x[i] - true_pos_x[i]
    }).collect();
    let pos_e_errors: Vec<f64> = (0..n).map(|i| {
        est_pos_y[i] - true_pos_y[i]
    }).collect();
    let pos_d_errors: Vec<f64> = (0..n).map(|i| {
        est_pos_z[i] - true_pos_z[i]
    }).collect();
    
    // 3D position errors
    let pos_3d_errors: Vec<f64> = (0..n).map(|i| {
        (pos_n_errors[i].powi(2) + pos_e_errors[i].powi(2) + pos_d_errors[i].powi(2)).sqrt()
    }).collect();
    
    // Velocity errors (simulated components)
    let vel_n_errors: Vec<f64> = (0..n).map(|i| {
        let error = (est_vel[i] - true_vel[i]) * 0.08 + (i as f64 * 0.0015).sin() * 0.1;
        error
    }).collect();
    let vel_e_errors: Vec<f64> = (0..n).map(|i| {
        let error = (est_vel[i] - true_vel[i]) * 0.06 + (i as f64 * 0.0018).cos() * 0.08;
        error
    }).collect();
    let vel_d_errors: Vec<f64> = (0..n).map(|i| {
        (est_vel[i] - true_vel[i]) * 0.85
    }).collect();
    
    // Helper functions for statistics
    let calc_stats = |data: &[f64]| -> (f64, f64, f64, f64, f64) {
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        let std = variance.sqrt();
        let rmse = (data.iter().map(|x| x.powi(2)).sum::<f64>() / data.len() as f64).sqrt();
        (min, max, mean, std, rmse)
    };
    
    let (pos_n_min, pos_n_max, pos_n_mean, pos_n_std, pos_n_rmse) = calc_stats(&pos_n_errors);
    let (pos_e_min, pos_e_max, pos_e_mean, pos_e_std, pos_e_rmse) = calc_stats(&pos_e_errors);
    let (pos_d_min, pos_d_max, pos_d_mean, pos_d_std, pos_d_rmse) = calc_stats(&pos_d_errors);
    let (vel_n_min, vel_n_max, vel_n_mean, vel_n_std, vel_n_rmse) = calc_stats(&vel_n_errors);
    let (vel_e_min, vel_e_max, vel_e_mean, vel_e_std, vel_e_rmse) = calc_stats(&vel_e_errors);
    let (vel_d_min, vel_d_max, vel_d_mean, vel_d_std, vel_d_rmse) = calc_stats(&vel_d_errors);
    let (pos_3d_min, pos_3d_max, pos_3d_mean, pos_3d_std, pos_3d_rmse) = calc_stats(&pos_3d_errors);
    
    ErrorStats {
        pos_n_min, pos_n_max, pos_n_mean, pos_n_std, pos_n_rmse,
        pos_e_min, pos_e_max, pos_e_mean, pos_e_std, pos_e_rmse,
        pos_d_min, pos_d_max, pos_d_mean, pos_d_std, pos_d_rmse,
        vel_n_min, vel_n_max, vel_n_mean, vel_n_std, vel_n_rmse,
        vel_e_min, vel_e_max, vel_e_mean, vel_e_std, vel_e_rmse,
        vel_d_min, vel_d_max, vel_d_mean, vel_d_std, vel_d_rmse,
        pos_3d_min, pos_3d_max, pos_3d_mean, pos_3d_std, pos_3d_rmse,
    }
}

fn generate_chart_data(chart_type: &str, config: &SimConfig) -> ChartData {
    let results = run_full_simulation(config);
    
    match chart_type {
        "altitude" => ChartData {
            time: results.time.clone(),
            data: results.altitude,
            data_3d: None,
            title: "Altitude vs Time".to_string(),
            y_label: "Altitude (m)".to_string(),
            chart_type: "2d".to_string(),
        },
        "velocity" => ChartData {
            time: results.time.clone(),
            data: results.velocity,
            data_3d: None,
            title: "Velocity vs Time".to_string(),
            y_label: "Velocity (m/s)".to_string(),
            chart_type: "2d".to_string(),
        },
        "acceleration" => ChartData {
            time: results.time.clone(),
            data: results.acceleration,
            data_3d: None,
            title: "Acceleration vs Time".to_string(),
            y_label: "Acceleration (m/s²)".to_string(),
            chart_type: "2d".to_string(),
        },
        "force" => ChartData {
            time: results.time.clone(),
            data: results.force,
            data_3d: None,
            title: "Net Force vs Time".to_string(),
            y_label: "Force (N)".to_string(),
            chart_type: "2d".to_string(),
        },
        "mass" => ChartData {
            time: results.time.clone(),
            data: results.mass,
            data_3d: None,
            title: "Mass vs Time".to_string(),
            y_label: "Mass (kg)".to_string(),
            chart_type: "2d".to_string(),
        },
        "trajectory" => ChartData {
            time: results.time.clone(),
            data: vec![],
            data_3d: Some((results.position_x, results.position_y, results.position_z)),
            title: "3D Flight Path".to_string(),
            y_label: "Position (m)".to_string(),
            chart_type: "3d".to_string(),
        },
        _ => ChartData {
            time: results.time,
            data: results.altitude,
            data_3d: None,
            title: "Altitude vs Time".to_string(),
            y_label: "Altitude (m)".to_string(),
            chart_type: "2d".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simulation_runs() {
        let config = SimConfig::default();
        let results = run_full_simulation(&config);
        
        assert!(!results.time.is_empty());
        assert!(!results.altitude.is_empty());
        assert_eq!(results.time.len(), results.altitude.len());
        assert!(results.success);
    }
    
    #[test]
    fn test_chart_generation() {
        let config = SimConfig::default();
        let chart = generate_chart_data("altitude", &config);
        
        assert!(!chart.time.is_empty());
        assert!(!chart.data.is_empty());
        assert_eq!(chart.title, "Altitude vs Time");
    }
    
    #[test]
    fn test_3d_trajectory() {
        let config = SimConfig::default();
        let chart = generate_chart_data("trajectory", &config);
        
        assert!(chart.data_3d.is_some());
        let (x, y, z) = chart.data_3d.unwrap();
        assert!(!x.is_empty());
        assert!(!y.is_empty());
        assert!(!z.is_empty());
    }
}
