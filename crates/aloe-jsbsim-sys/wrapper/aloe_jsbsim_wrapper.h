#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AloeJsbsimOpaque AloeJsbsimOpaque;

typedef struct AloeJsbsimConfig {
  double dry_mass_kg;
  double fuel_mass_kg;
  double oxidizer_mass_kg;
  double inertia_xx_kg_m2;
  double inertia_yy_kg_m2;
  double inertia_zz_kg_m2;
  double cg_full_m;
  double cg_empty_m;
  double cp_location_m;
  double ref_area_m2;
  double drag_coeff_axial;
  double normal_force_coeff;
  double thrust_newtons;
  double burn_time_s;
  double isp_s;
  double nozzle_location_m;
  double gravity_mps2;
  double air_density_sea_level_kg_m3;
  double launch_rod_length_m;
  double wind_north_mps;
  double wind_east_mps;
  double wind_down_mps;
  double launch_delay_s;
  double spin_rate_deg_per_s;
  double thrust_cant_deg;
  double nozzle_exit_pressure_psf;
  double nozzle_area_ft2;
  double pad_static_friction;
  double pad_dynamic_friction;
  double pad_spring_coeff_lbs_ft;
  double pad_damping_coeff_lbs_ft_s;
  double dt_s;
  double max_time_s;
  size_t thrust_curve_len;
  const double* thrust_curve_time_s;
  const double* thrust_curve_thrust_n;
} AloeJsbsimConfig;

typedef struct AloeJsbsimOutput {
  size_t len;
  double* time_s;
  double* pos_n_m;
  double* pos_e_m;
  double* pos_d_m;
  double* vel_n_mps;
  double* vel_e_mps;
  double* vel_d_mps;
  double* accel_bx_mps2;
  double* accel_by_mps2;
  double* accel_bz_mps2;
  double* p_rad_s;
  double* q_rad_s;
  double* r_rad_s;
  double* quat_w;
  double* quat_x;
  double* quat_y;
  double* quat_z;
} AloeJsbsimOutput;

AloeJsbsimOpaque* aloe_jsbsim_create(void);
void aloe_jsbsim_destroy(AloeJsbsimOpaque* handle);
int aloe_jsbsim_run(AloeJsbsimOpaque* handle, const AloeJsbsimConfig* config, AloeJsbsimOutput* output);
const char* aloe_jsbsim_last_error(const AloeJsbsimOpaque* handle);

#ifdef __cplusplus
}
#endif
