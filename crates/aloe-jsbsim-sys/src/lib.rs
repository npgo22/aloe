use std::os::raw::{c_char, c_int};

#[repr(C)]
pub struct AloeJsbsimOpaque {
    _private: [u8; 0],
}

#[repr(C)]
pub struct AloeJsbsimConfig {
    pub dry_mass_kg: f64,
    pub fuel_mass_kg: f64,
    pub oxidizer_mass_kg: f64,
    pub inertia_xx_kg_m2: f64,
    pub inertia_yy_kg_m2: f64,
    pub inertia_zz_kg_m2: f64,
    pub cg_full_m: f64,
    pub cg_empty_m: f64,
    pub cp_location_m: f64,
    pub ref_area_m2: f64,
    pub drag_coeff_axial: f64,
    pub normal_force_coeff: f64,
    pub thrust_newtons: f64,
    pub burn_time_s: f64,
    pub isp_s: f64,
    pub nozzle_location_m: f64,
    pub gravity_mps2: f64,
    pub air_density_sea_level_kg_m3: f64,
    pub launch_rod_length_m: f64,
    pub wind_north_mps: f64,
    pub wind_east_mps: f64,
    pub wind_down_mps: f64,
    pub launch_delay_s: f64,
    pub spin_rate_deg_per_s: f64,
    pub thrust_cant_deg: f64,
    pub nozzle_exit_pressure_psf: f64,
    pub nozzle_area_ft2: f64,
    pub pad_static_friction: f64,
    pub pad_dynamic_friction: f64,
    pub pad_spring_coeff_lbs_ft: f64,
    pub pad_damping_coeff_lbs_ft_s: f64,
    pub dt_s: f64,
    pub max_time_s: f64,
    pub thrust_curve_len: usize,
    pub thrust_curve_time_s: *const f64,
    pub thrust_curve_thrust_n: *const f64,
}

#[repr(C)]
pub struct AloeJsbsimOutput {
    pub len: usize,
    pub time_s: *mut f64,
    pub pos_n_m: *mut f64,
    pub pos_e_m: *mut f64,
    pub pos_d_m: *mut f64,
    pub vel_n_mps: *mut f64,
    pub vel_e_mps: *mut f64,
    pub vel_d_mps: *mut f64,
    pub accel_bx_mps2: *mut f64,
    pub accel_by_mps2: *mut f64,
    pub accel_bz_mps2: *mut f64,
    pub p_rad_s: *mut f64,
    pub q_rad_s: *mut f64,
    pub r_rad_s: *mut f64,
    pub quat_w: *mut f64,
    pub quat_x: *mut f64,
    pub quat_y: *mut f64,
    pub quat_z: *mut f64,
}

unsafe extern "C" {
    pub fn aloe_jsbsim_create() -> *mut AloeJsbsimOpaque;
    pub fn aloe_jsbsim_destroy(handle: *mut AloeJsbsimOpaque);
    pub fn aloe_jsbsim_run(
        handle: *mut AloeJsbsimOpaque,
        config: *const AloeJsbsimConfig,
        output: *mut AloeJsbsimOutput,
    ) -> c_int;
    pub fn aloe_jsbsim_last_error(handle: *const AloeJsbsimOpaque) -> *const c_char;
}
