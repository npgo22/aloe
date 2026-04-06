use aloe_jsbsim_sys::{
    aloe_jsbsim_create, aloe_jsbsim_destroy, aloe_jsbsim_last_error, aloe_jsbsim_run,
    AloeJsbsimConfig, AloeJsbsimOutput,
};
use anyhow::{anyhow, Result};
use nalgebra::{Quaternion, UnitQuaternion, Vector3};
use std::ffi::CStr;

const MAX_SIM_SAMPLES: usize = 1_000_000;

#[derive(Clone, Debug)]
pub struct JsbsimRocketParams {
    pub dry_mass: f64,
    pub fuel_mass: f64,
    pub oxidizer_mass: f64,
    pub inertia_tensor: Vector3<f64>,
    pub cg_full: f64,
    pub cg_empty: f64,
    pub cp_location: f64,
    pub ref_area: f64,
    pub drag_coeff_axial: f64,
    pub normal_force_coeff: f64,
    pub thrust_curve: Vec<(f64, f64)>,
    pub burn_time: f64,
    pub isp: f64,
    pub nozzle_location: f64,
    pub gravity: f64,
    pub air_density_sea_level: f64,
    pub launch_rod_length: f64,
    pub wind_velocity_ned: Vector3<f64>,
    pub launch_delay: f64,
    pub spin_rate: f64,
    pub thrust_cant: f64,
    pub nozzle_exit_pressure_psf: f64,
    pub nozzle_area_ft2: f64,
    pub pad_static_friction: f64,
    pub pad_dynamic_friction: f64,
    pub pad_spring_coeff_lbs_ft: f64,
    pub pad_damping_coeff_lbs_ft_s: f64,
    pub sim_dt: f64,
    pub max_time: f64,
}

#[derive(Clone, Debug)]
pub struct JsbsimSimResult {
    pub time: Vec<f64>,
    pub pos: Vec<Vector3<f64>>,
    pub vel: Vec<Vector3<f64>>,
    pub accel_body: Vec<Vector3<f64>>,
    pub ang_vel: Vec<Vector3<f64>>,
    pub orientation: Vec<UnitQuaternion<f64>>,
}

pub fn simulate(params: &JsbsimRocketParams) -> Result<JsbsimSimResult> {
    if !params.sim_dt.is_finite() || params.sim_dt <= 0.0 {
        return Err(anyhow!("invalid JSBSim sim_dt: {}", params.sim_dt));
    }
    if !params.max_time.is_finite() || params.max_time <= 0.0 {
        return Err(anyhow!("invalid JSBSim max_time: {}", params.max_time));
    }

    let sample_count = (params.max_time / params.sim_dt).ceil();
    if !sample_count.is_finite() || sample_count > MAX_SIM_SAMPLES as f64 {
        return Err(anyhow!(
            "JSBSim sample count exceeds limit ({MAX_SIM_SAMPLES}); increase sim_dt or reduce max_time"
        ));
    }

    let handle = unsafe { aloe_jsbsim_create() };
    if handle.is_null() {
        return Err(anyhow!("failed to create JSBSim handle"));
    }

    let max_samples = (sample_count as usize + 1).max(1);
    let mut time = vec![0.0; max_samples];
    let mut pos_n = vec![0.0; max_samples];
    let mut pos_e = vec![0.0; max_samples];
    let mut pos_d = vec![0.0; max_samples];
    let mut vel_n = vec![0.0; max_samples];
    let mut vel_e = vec![0.0; max_samples];
    let mut vel_d = vec![0.0; max_samples];
    let mut accel_bx = vec![0.0; max_samples];
    let mut accel_by = vec![0.0; max_samples];
    let mut accel_bz = vec![0.0; max_samples];
    let mut p = vec![0.0; max_samples];
    let mut q = vec![0.0; max_samples];
    let mut r = vec![0.0; max_samples];
    let mut qw = vec![0.0; max_samples];
    let mut qx = vec![0.0; max_samples];
    let mut qy = vec![0.0; max_samples];
    let mut qz = vec![0.0; max_samples];

    let thrust = params
        .thrust_curve
        .iter()
        .map(|(_, thrust)| *thrust)
        .fold(0.0_f64, f64::max);
    let thrust_curve_time: Vec<f64> = params.thrust_curve.iter().map(|(time, _)| *time).collect();
    let thrust_curve_thrust: Vec<f64> = params
        .thrust_curve
        .iter()
        .map(|(_, thrust)| *thrust)
        .collect();
    let config = AloeJsbsimConfig {
        dry_mass_kg: params.dry_mass,
        fuel_mass_kg: params.fuel_mass,
        oxidizer_mass_kg: params.oxidizer_mass,
        inertia_xx_kg_m2: params.inertia_tensor.x,
        inertia_yy_kg_m2: params.inertia_tensor.y,
        inertia_zz_kg_m2: params.inertia_tensor.z,
        cg_full_m: params.cg_full,
        cg_empty_m: params.cg_empty,
        cp_location_m: params.cp_location,
        ref_area_m2: params.ref_area,
        drag_coeff_axial: params.drag_coeff_axial,
        normal_force_coeff: params.normal_force_coeff,
        thrust_newtons: thrust,
        burn_time_s: params.burn_time,
        isp_s: params.isp,
        nozzle_location_m: params.nozzle_location,
        gravity_mps2: params.gravity,
        air_density_sea_level_kg_m3: params.air_density_sea_level,
        launch_rod_length_m: params.launch_rod_length,
        wind_north_mps: params.wind_velocity_ned.x,
        wind_east_mps: params.wind_velocity_ned.y,
        wind_down_mps: params.wind_velocity_ned.z,
        launch_delay_s: params.launch_delay,
        spin_rate_deg_per_s: params.spin_rate,
        thrust_cant_deg: params.thrust_cant,
        nozzle_exit_pressure_psf: params.nozzle_exit_pressure_psf,
        nozzle_area_ft2: params.nozzle_area_ft2,
        pad_static_friction: params.pad_static_friction,
        pad_dynamic_friction: params.pad_dynamic_friction,
        pad_spring_coeff_lbs_ft: params.pad_spring_coeff_lbs_ft,
        pad_damping_coeff_lbs_ft_s: params.pad_damping_coeff_lbs_ft_s,
        dt_s: params.sim_dt,
        max_time_s: params.max_time,
        thrust_curve_len: thrust_curve_time.len(),
        thrust_curve_time_s: thrust_curve_time.as_ptr(),
        thrust_curve_thrust_n: thrust_curve_thrust.as_ptr(),
    };
    let mut output = AloeJsbsimOutput {
        len: max_samples,
        time_s: time.as_mut_ptr(),
        pos_n_m: pos_n.as_mut_ptr(),
        pos_e_m: pos_e.as_mut_ptr(),
        pos_d_m: pos_d.as_mut_ptr(),
        vel_n_mps: vel_n.as_mut_ptr(),
        vel_e_mps: vel_e.as_mut_ptr(),
        vel_d_mps: vel_d.as_mut_ptr(),
        accel_bx_mps2: accel_bx.as_mut_ptr(),
        accel_by_mps2: accel_by.as_mut_ptr(),
        accel_bz_mps2: accel_bz.as_mut_ptr(),
        p_rad_s: p.as_mut_ptr(),
        q_rad_s: q.as_mut_ptr(),
        r_rad_s: r.as_mut_ptr(),
        quat_w: qw.as_mut_ptr(),
        quat_x: qx.as_mut_ptr(),
        quat_y: qy.as_mut_ptr(),
        quat_z: qz.as_mut_ptr(),
    };

    let status = unsafe { aloe_jsbsim_run(handle, &config, &mut output) };

    if status != 0 {
        let message = unsafe {
            let ptr = aloe_jsbsim_last_error(handle);
            if ptr.is_null() {
                None
            } else {
                Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
            }
        };
        unsafe { aloe_jsbsim_destroy(handle) };
        return Err(anyhow!(
            "JSBSim run failed with status {status}: {}",
            message.unwrap_or_else(|| "unknown JSBSim error".to_string())
        ));
    }

    unsafe { aloe_jsbsim_destroy(handle) };

    time.truncate(output.len);
    pos_n.truncate(output.len);
    pos_e.truncate(output.len);
    pos_d.truncate(output.len);
    vel_n.truncate(output.len);
    vel_e.truncate(output.len);
    vel_d.truncate(output.len);
    accel_bx.truncate(output.len);
    accel_by.truncate(output.len);
    accel_bz.truncate(output.len);
    p.truncate(output.len);
    q.truncate(output.len);
    r.truncate(output.len);
    qw.truncate(output.len);
    qx.truncate(output.len);
    qy.truncate(output.len);
    qz.truncate(output.len);

    Ok(JsbsimSimResult {
        pos: pos_n
            .iter()
            .zip(&pos_e)
            .zip(&pos_d)
            .map(|((x, y), z)| Vector3::new(*x, *y, *z))
            .collect(),
        vel: vel_n
            .iter()
            .zip(&vel_e)
            .zip(&vel_d)
            .map(|((x, y), z)| Vector3::new(*x, *y, *z))
            .collect(),
        accel_body: accel_bx
            .iter()
            .zip(&accel_by)
            .zip(&accel_bz)
            .map(|((x, y), z)| Vector3::new(*x, *y, *z))
            .collect(),
        ang_vel: p
            .iter()
            .zip(&q)
            .zip(&r)
            .map(|((x, y), z)| Vector3::new(*x, *y, *z))
            .collect(),
        orientation: qw
            .iter()
            .zip(&qx)
            .zip(&qy)
            .zip(&qz)
            .map(|(((w, x), y), z)| {
                UnitQuaternion::from_quaternion(Quaternion::new(*w, *x, *y, *z))
            })
            .collect(),
        time,
    })
}
