//! Aloe CLI - Command line interface for rocket simulation.

use aloe_core::eskf::EskfTuning;
use aloe_sim::filter::{run_filter, FilterResult};
use aloe_sim::sensor::{generate_sensor_data, SensorConfig, SensorData};
use aloe_sim::sim::{simulate_6dof, RocketParams, SimResult};
use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use nalgebra::Vector3;
use std::collections::HashMap;
use std::path::PathBuf;

/// Public function that can be called from the main binary
pub fn run_cli_main(args: &[&str]) -> Result<()> {
    let args = Args::parse_from(args);
    main_inner(args)
}

#[derive(Parser, Debug)]
#[command(name = "aloe-cli")]
#[command(about = "Hobby-rocket 6-DoF flight simulator")]
#[command(version)]
pub struct Args {
    /// Output directory
    #[arg(short, long, default_value = "output")]
    output_dir: PathBuf,

    /// Output file format
    #[arg(short, long, value_enum, default_value = "csv")]
    format: OutputFormat,

    /// Run a single simulation
    #[arg(long)]
    single: bool,

    // ── Simulation parameters ─────────────────────────────────
    #[arg(long, default_value_t = 20.0)]
    dry_mass: f64,

    #[arg(long, default_value_t = 10.0)]
    propellant_mass: f64,

    #[arg(long, default_value_t = 2000.0)]
    thrust: f64,

    #[arg(long, default_value_t = 5.0)]
    burn_time: f64,

    #[arg(long, default_value_t = 0.5)]
    drag_coeff: f64,

    #[arg(long, default_value_t = 0.018)]
    ref_area: f64,

    #[arg(long, default_value_t = 9.81)]
    gravity: f64,

    #[arg(long, default_value_t = 5.0)]
    wind_speed: f64, // North

    #[arg(long, default_value_t = 0.0)]
    wind_speed_z: f64, // East

    #[arg(long, default_value_t = 1.225)]
    air_density: f64,

    #[arg(long, default_value_t = 1.0)]
    launch_delay: f64,

    #[arg(long, default_value_t = 0.0)]
    spin_rate: f64,

    #[arg(long, default_value_t = 0.0)]
    thrust_cant: f64,

    // ── Sensor options ────────────────────────────────────────
    #[arg(long)]
    no_sensors: bool,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value_t = 1.0)]
    noise_scale: f64,

    // ── Filter options ────────────────────────────────────────
    #[arg(long)]
    no_filter: bool,

    #[arg(long, default_value_t = 0.5)]
    accel_noise_density: f32, // Tuning param for filter

    // ── Sweep options ──────────────────────────────────────────
    #[arg(long, value_delimiter = ' ')]
    sweep_params: Option<Vec<String>>,

    #[arg(long, default_value_t = 5)]
    sweep_steps: usize,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OutputFormat {
    Csv,
    Json,
}

fn main() -> Result<()> {
    let args = Args::parse();
    main_inner(args)
}

fn main_inner(args: Args) -> Result<()> {
    println!("Aloe Rocket Simulator (6-DoF)");
    println!("=============================\n");

    if args.single {
        run_single(&args)?;
    } else {
        run_sweep(&args)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Single Run
// ---------------------------------------------------------------------------
fn run_single(args: &Args) -> Result<()> {
    println!("Running single simulation...");

    // 1. Setup & Simulate
    let params = build_rocket_params(args);
    let result = simulate_6dof(&params);

    print_sim_stats(&result);

    // 2. Generate Sensors
    let (sensor_data, filter_result) = if !args.no_sensors {
        let config = build_sensor_config(args);
        let s_data = generate_sensor_data(&result, &config);

        // 3. Run Filter (if requested)
        let f_res = if !args.no_filter {
            println!("Running Navigation Filter...");
            Some(run_filter(&result, &s_data))
        } else {
            None
        };
        (Some(s_data), f_res)
    } else {
        (None, None)
    };

    // 4. Export
    write_output(args, &result, sensor_data.as_ref(), filter_result.as_ref())?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Sweep Run
// ---------------------------------------------------------------------------
fn run_sweep(args: &Args) -> Result<()> {
    // Simplified sweep implementation
    // We only sweep mass/thrust to demonstrate the new pipeline
    let default_sweep = vec!["thrust".to_string()];
    let sweep_names = args.sweep_params.as_ref().unwrap_or(&default_sweep);

    // Hardcoded simple sweep for demonstration
    // In a full CLI, you'd parse ranges dynamically
    println!("Sweeping parameters: {:?}", sweep_names);

    let steps = args.sweep_steps;
    let mut summary_rows = Vec::new();

    // Just sweeping Thrust for example
    let start_thrust = args.thrust * 0.5;
    let end_thrust = args.thrust * 1.5;

    for i in 0..steps {
        let val = start_thrust + (end_thrust - start_thrust) * (i as f64 / (steps - 1) as f64);

        let mut params = build_rocket_params(args);
        params.thrust_curve = vec![
            (0.0, val),
            (args.burn_time, val),
            (args.burn_time + 0.01, 0.0),
        ];

        let res = simulate_6dof(&params);
        let apogee = res
            .pos
            .iter()
            .map(|p| -p.z)
            .fold(f64::NEG_INFINITY, f64::max);

        println!(
            "Run {}/{} | Thrust: {:.1} N -> Apogee: {:.1} m",
            i + 1,
            steps,
            val,
            apogee
        );

        let mut row = HashMap::new();
        row.insert("thrust".to_string(), val);
        row.insert("apogee".to_string(), apogee);
        summary_rows.push(row);
    }

    // Write summary
    let path = args.output_dir.join("sweep_summary.csv");
    std::fs::create_dir_all(&args.output_dir)?;
    let mut wtr = csv::Writer::from_path(&path)?;
    wtr.write_record(&["thrust", "apogee"])?;
    for row in summary_rows {
        wtr.write_record(&[
            format!("{:.2}", row["thrust"]),
            format!("{:.2}", row["apogee"]),
        ])?;
    }

    println!("\nSweep complete. Summary at {:?}", path);
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_rocket_params(args: &Args) -> RocketParams {
    // Construct the 6-DoF params from simple CLI args
    RocketParams {
        dry_mass: args.dry_mass,
        propellant_mass: args.propellant_mass,
        // Approximate inertia for a cylinder: I = 1/12 * M * L^2
        // Assuming Length = 3.0m
        inertia_tensor: Vector3::new(0.1, 15.0, 15.0),
        cg_location: 1.5,
        cp_location: 2.0, // Stable (behind CG)
        ref_area: args.ref_area,
        drag_coeff_axial: args.drag_coeff,
        normal_force_coeff: 12.0,
        // Create a square pulse thrust curve
        thrust_curve: vec![
            (0.0, args.thrust),
            (args.burn_time, args.thrust),
            (args.burn_time + 0.01, 0.0),
        ],
        burn_time: args.burn_time,
        nozzle_location: 3.0,
        gravity: args.gravity,
        air_density_sea_level: args.air_density,
        launch_rod_length: 2.0,
        // Map wind args (North/East) to NED
        wind_velocity_ned: Vector3::new(args.wind_speed, args.wind_speed_z, 0.0),
        launch_delay: args.launch_delay,
        spin_rate: args.spin_rate,
        thrust_cant: args.thrust_cant,
    }
}

fn build_sensor_config(args: &Args) -> SensorConfig {
    // Map scalar "noise_scale" to individual sensor sigmas
    let s = args.noise_scale;
    SensorConfig {
        accel_noise_std: 0.1 * s,   // m/s^2
        gyro_noise_std: 0.002 * s,  // rad/s
        mag_noise_std: 0.001 * s,   // Gauss
        baro_noise_std: 0.5 * s,    // meters
        gps_pos_noise_std: 2.0 * s, // meters
        gps_vel_noise_std: 0.1 * s, // m/s
        accel_bias: Vector3::zeros(),
        gyro_bias: Vector3::zeros(),
        seed: args.seed,
    }
}

fn print_sim_stats(result: &SimResult) {
    let max_alt = result
        .pos
        .iter()
        .map(|p| -p.z)
        .fold(f64::NEG_INFINITY, f64::max);
    let flight_time = result.time.last().copied().unwrap_or(0.0);

    println!("\nSimulation Stats:");
    println!("  Steps:       {}", result.time.len());
    println!("  Flight Time: {:.2} s", flight_time);
    println!("  Apogee:      {:.2} m (AGL)", max_alt);
    println!("-----------------------------");
}

fn write_output(
    args: &Args,
    sim: &SimResult,
    sensors: Option<&SensorData>,
    filter: Option<&FilterResult>,
) -> Result<()> {
    std::fs::create_dir_all(&args.output_dir)?;
    let path = args.output_dir.join("simulation.csv");
    let mut wtr = csv::Writer::from_path(&path)?;

    // Header
    // We flatten vectors: pos_n, pos_e, pos_d
    wtr.write_record(&[
        "time",
        // Sim Truth
        "true_pos_n",
        "true_pos_e",
        "true_pos_d",
        "true_vel_n",
        "true_vel_e",
        "true_vel_d",
        "true_accel_x",
        "true_accel_y",
        "true_accel_z", // Body frame proper accel
        // Filter Estimates (if available)
        "est_pos_n",
        "est_pos_e",
        "est_pos_d",
        "est_vel_n",
        "est_vel_e",
        "est_vel_d",
        "est_roll",
        "est_pitch",
        "est_yaw",
    ])?;

    let n = sim.time.len();
    for i in 0..n {
        let t = sim.time[i];
        let p = sim.pos[i];
        let v = sim.vel[i];
        let a = sim.accel_body[i];

        // Default "empty" values if filter didn't run
        let (ep, ev, erpy) = if let Some(f) = filter {
            if i < f.time.len() {
                (f.position[i], f.velocity[i], f.orientation_euler[i])
            } else {
                (Vector3::zeros(), Vector3::zeros(), Vector3::zeros())
            }
        } else {
            (Vector3::zeros(), Vector3::zeros(), Vector3::zeros())
        };

        wtr.write_record(&[
            format!("{:.4}", t),
            // Truth
            format!("{:.4}", p.x),
            format!("{:.4}", p.y),
            format!("{:.4}", p.z),
            format!("{:.4}", v.x),
            format!("{:.4}", v.y),
            format!("{:.4}", v.z),
            format!("{:.4}", a.x),
            format!("{:.4}", a.y),
            format!("{:.4}", a.z),
            // Est
            format!("{:.4}", ep.x),
            format!("{:.4}", ep.y),
            format!("{:.4}", ep.z),
            format!("{:.4}", ev.x),
            format!("{:.4}", ev.y),
            format!("{:.4}", ev.z),
            format!("{:.4}", erpy.x),
            format!("{:.4}", erpy.y),
            format!("{:.4}", erpy.z),
        ])?;
    }

    wtr.flush()?;
    println!("Data written to {:?}", path);
    Ok(())
}
