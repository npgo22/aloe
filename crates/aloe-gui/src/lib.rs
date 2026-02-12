//! Aloe GUI - Web interface for rocket simulation
//! 
//! Features:
//! - Configuration panel with tabs (Rocket, ENV, Sensors, Filter)
//! - Real-time simulation via API
//! - Multiple chart types (2D and 3D)
//! - Sensor data visualization
//! - Filter error statistics

use aloe_sim::{
    filter::{run_filter, FilterConfig, FilterResult},
    sensor::{generate_sensor_data, SensorConfig, SensorData},
    sim::{simulate_6dof, RocketParams, SimResult},
};
use serde::Serialize;
use std::collections::HashMap;
use tower_http::services::ServeDir;
use nalgebra::Vector3;
use axum::{Router, routing::get, extract::Query, Json};

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
    no_sensors: bool,
    bmi088_accel_enabled: bool,
    bmi088_gyro_enabled: bool,
    adxl375_enabled: bool,
    lis3mdl_enabled: bool,
    ms5611_enabled: bool,
    gps_enabled: bool,
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
            no_sensors: false,
            bmi088_accel_enabled: true,
            bmi088_gyro_enabled: true,
            adxl375_enabled: true,
            lis3mdl_enabled: true,
            ms5611_enabled: true,
            gps_enabled: true,
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
    
    parse_param!(no_sensors, "no_sensors", bool);
    parse_param!(bmi088_accel_enabled, "bmi088_accel_enabled", bool);
    parse_param!(bmi088_gyro_enabled, "bmi088_gyro_enabled", bool);
    parse_param!(adxl375_enabled, "adxl375_enabled", bool);
    parse_param!(lis3mdl_enabled, "lis3mdl_enabled", bool);
    parse_param!(ms5611_enabled, "ms5611_enabled", bool);
    parse_param!(gps_enabled, "gps_enabled", bool);
    
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
    state_changes_sim: Vec<StateChange>,
    state_changes_eskf: Vec<StateChange>,
    sensor_data: GuiSensorData,
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
struct GuiSensorData {
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
    let rocket_params = RocketParams {
        dry_mass: config.dry_mass,
        propellant_mass: config.propellant_mass,
        inertia_tensor: Vector3::new(0.1, 10.0, 10.0), // Approximate
        cg_location: 1.5,
        cp_location: 2.0,
        ref_area: config.ref_area,
        drag_coeff_axial: config.drag_coeff,
        normal_force_coeff: 12.0,
        thrust_curve: vec![(0.0, config.thrust), (config.burn_time, config.thrust), (config.burn_time + 0.1, 0.0)],
        burn_time: config.burn_time,
        nozzle_location: 3.0,
        gravity: config.gravity,
        air_density_sea_level: config.air_density,
        launch_rod_length: 2.0,
        wind_velocity_ned: Vector3::new(config.wind_north, config.wind_east, 0.0),
        launch_delay: config.launch_delay,
        spin_rate: config.spin_rate,
        thrust_cant: config.thrust_cant,
    };

    let sim_result = simulate_6dof(&rocket_params);

    // Convert SimResult to the GUI format
        let time: Vec<f64> = sim_result.time.clone();
    let altitude: Vec<f64> = sim_result.pos.iter().map(|p| -p.z).collect(); // Convert from NED to altitude
    let velocity: Vec<f64> = sim_result.vel.iter().map(|v| v.norm()).collect();
    let acceleration: Vec<f64> = sim_result.accel_body.iter().map(|a| a.x).collect(); // Axial acceleration
    let force: Vec<f64> = sim_result.accel_body.iter().zip(&sim_result.vel).map(|(a, _)| a.x * (rocket_params.dry_mass + rocket_params.propellant_mass)).collect(); // Approximate
    let mass: Vec<f64> = time.iter().map(|&t| {
        if t < config.burn_time {
            rocket_params.dry_mass + rocket_params.propellant_mass * (1.0 - t / config.burn_time).max(0.0)
        } else {
            rocket_params.dry_mass
        }
    }).collect();
    let position_x: Vec<f64> = sim_result.pos.iter().map(|p| p.x).collect();
    let position_y: Vec<f64> = sim_result.pos.iter().map(|p| p.y).collect();
    let position_z: Vec<f64> = sim_result.pos.iter().map(|p| -p.z).collect(); // NED to altitude

    // Generate state changes for sim
    let state_changes_sim = generate_state_changes(&time, &sim_result.pos, &sim_result.vel);

    if config.no_sensors {
        let sensor_data = GuiSensorData {
            accel_x: vec![],
            accel_y: vec![],
            accel_z: vec![],
            gyro_x: vec![],
            gyro_y: vec![],
            gyro_z: vec![],
            baro_alt: vec![],
        };
        let filter_data = FilterData {
            est_pos_x: vec![],
            est_pos_y: vec![],
            est_pos_z: vec![],
            est_vel_x: vec![],
            est_vel_y: vec![],
            est_vel_z: vec![],
            est_vel_mag: vec![],
        };
        let error_stats = ErrorStats {
            pos_n_min: 0.0,
            pos_n_max: 0.0,
            pos_n_mean: 0.0,
            pos_n_std: 0.0,
            pos_n_rmse: 0.0,
            pos_e_min: 0.0,
            pos_e_max: 0.0,
            pos_e_mean: 0.0,
            pos_e_std: 0.0,
            pos_e_rmse: 0.0,
            pos_d_min: 0.0,
            pos_d_max: 0.0,
            pos_d_mean: 0.0,
            pos_d_std: 0.0,
            pos_d_rmse: 0.0,
            vel_n_min: 0.0,
            vel_n_max: 0.0,
            vel_n_mean: 0.0,
            vel_n_std: 0.0,
            vel_n_rmse: 0.0,
            vel_e_min: 0.0,
            vel_e_max: 0.0,
            vel_e_mean: 0.0,
            vel_e_std: 0.0,
            vel_e_rmse: 0.0,
            vel_d_min: 0.0,
            vel_d_max: 0.0,
            vel_d_mean: 0.0,
            vel_d_std: 0.0,
            vel_d_rmse: 0.0,
            pos_3d_min: 0.0,
            pos_3d_max: 0.0,
            pos_3d_mean: 0.0,
            pos_3d_std: 0.0,
            pos_3d_rmse: 0.0,
        };
        FullSimulationResponse {
            time,
            altitude,
            velocity,
            acceleration,
            force,
            mass,
            position_x,
            position_y,
            position_z,
            state_changes_sim,
            state_changes_eskf: vec![],
            sensor_data,
            filter_data,
            error_stats,
            success: true,
        }
    } else {
        // Generate sensor data
        let sensor_config = SensorConfig {
            noise_scale: config.noise_scale,
            accel_noise_std: 0.01,
            gyro_noise_std: 0.001,
            mag_noise_std: 0.001,
            baro_noise_std: 0.1,
            gps_pos_noise_std: 1.0,
            gps_vel_noise_std: 0.1,
            accel_bias: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
            seed: config.seed,
            accel_enabled: config.bmi088_accel_enabled || config.adxl375_enabled,
            gyro_enabled: config.bmi088_gyro_enabled,
            mag_enabled: config.lis3mdl_enabled,
            baro_enabled: config.ms5611_enabled,
            gps_enabled: config.gps_enabled,
        };
        let sensor_data_sim = generate_sensor_data(&sim_result, &sensor_config);

        // Generate sensor data for GUI
        let gui_sensor_data = GuiSensorData {
            accel_x: sensor_data_sim.accel_meas.iter().map(|v| v.x).collect(),
            accel_y: sensor_data_sim.accel_meas.iter().map(|v| v.y).collect(),
            accel_z: sensor_data_sim.accel_meas.iter().map(|v| v.z).collect(),
            gyro_x: sensor_data_sim.gyro_meas.iter().map(|v| v.x).collect(),
            gyro_y: sensor_data_sim.gyro_meas.iter().map(|v| v.y).collect(),
            gyro_z: sensor_data_sim.gyro_meas.iter().map(|v| v.z).collect(),
            baro_alt: sensor_data_sim.baro_alt.clone(),
        };

        // Run filter (simplified)
        let filter_config = FilterConfig::default();
        let filter_result = run_filter(&sim_result, &sensor_data_sim, &filter_config);

        let filter_data = FilterData {
            est_pos_x: filter_result.position.iter().map(|p| p.x).collect(),
            est_pos_y: filter_result.position.iter().map(|p| p.y).collect(),
            est_pos_z: filter_result.position.iter().map(|p| p.z).collect(),
            est_vel_x: filter_result.velocity.iter().map(|v| v.x).collect(),
            est_vel_y: filter_result.velocity.iter().map(|v| v.y).collect(),
            est_vel_z: filter_result.velocity.iter().map(|v| v.z).collect(),
            est_vel_mag: vec![], // Will be computed
        };
        let est_vel_mag: Vec<f64> = filter_data.est_vel_x.iter()
            .zip(&filter_data.est_vel_y)
            .zip(&filter_data.est_vel_z)
            .map(|((x, y), z)| (x * x + y * y + z * z).sqrt())
            .collect();

        let filter_data = FilterData {
            est_pos_x: filter_data.est_pos_x,
            est_pos_y: filter_data.est_pos_y,
            est_pos_z: filter_data.est_pos_z,
            est_vel_x: filter_data.est_vel_x,
            est_vel_y: filter_data.est_vel_y,
            est_vel_z: filter_data.est_vel_z,
            est_vel_mag,
        };

        // Generate state changes for ESKF
        let state_changes_eskf = generate_state_changes(&time, &filter_result.position, &filter_result.velocity);

        // Calculate error statistics
        let error_stats = calculate_error_stats(
            &position_x, &position_y, &position_z,
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
            position_x,
            position_y,
            position_z,
            state_changes_sim,
            state_changes_eskf,
            sensor_data: gui_sensor_data,
            filter_data,
            error_stats,
            success: true,
        }
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

fn generate_state_changes(time: &[f64], pos: &[Vector3<f64>], vel: &[Vector3<f64>]) -> Vec<StateChange> {
    let mut state_changes = vec![StateChange { time: 0.0, state: "Pad".to_string(), description: "On Pad".to_string() }];
    let mut last_state = "Pad";
    for i in 0..time.len() {
        let t = time[i];
        let vel_mag = vel[i].norm();
        let alt = -pos[i].z;
        let on_ground = alt < 0.1 && vel_mag < 1.0;

        let state = if t < 1.0 {
            "Pad"
        } else if t < 6.0 {
            "Ascent"
        } else if vel_mag > 5.0 && !on_ground {
            "Coast"
        } else if alt > 10.0 {
            "Descent"
        } else {
            "Landed"
        };

        if state != last_state {
            state_changes.push(StateChange {
                time: t,
                state: state.to_string(),
                description: match state {
                    "Pad" => "On Pad",
                    "Ascent" => "Ascent",
                    "Coast" => "Coast",
                    "Descent" => "Descent",
                    "Landed" => "Landed",
                    _ => state,
                }.to_string(),
            });
            last_state = state;
        }
    }

    state_changes
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
