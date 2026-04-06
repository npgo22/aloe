#include "aloe_jsbsim_wrapper.h"

#include "FGFDMExec.h"
#include "input_output/FGLog.h"
#include "initialization/FGInitialCondition.h"
#include "models/FGAccelerations.h"
#include "models/FGAtmosphere.h"
#include "models/FGPropagate.h"
#include "simgear/misc/sg_path.hxx"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kFeetPerMeter = 3.280839895013123;
constexpr double kPoundsPerKilogram = 2.2046226218487757;
constexpr double kSlugFt2PerKgM2 = 23.73036040423184;
constexpr double kInchesPerMeter = 39.37007874015748;
constexpr double kSquareFeetPerSquareMeter = 10.763910416709722;
constexpr double kPoundsForcePerNewton = 0.22480894387096263;
constexpr double kMetersPerFoot = 0.3048;
constexpr double kEarthRadiusFt = 20925646.32546;
constexpr double kStandardSeaLevelPressurePsf = 2116.228;
constexpr double kStandardSeaLevelDensityKgPerM3 = 1.225;

struct LiquidEngineProfile {
  double fuel_flow_lbs_s;
  double oxidizer_flow_lbs_s;
  double mixture_ratio;
};

static LiquidEngineProfile build_liquid_engine_profile(const AloeJsbsimConfig& cfg) {
  if (cfg.thrust_newtons <= 1e-6 || cfg.isp_s <= 1e-6 || cfg.fuel_mass_kg <= 0.0 || cfg.oxidizer_mass_kg <= 0.0) {
    return {0.0, 0.0, 1.0};
  }

  const double fuel_lbs = std::max(cfg.fuel_mass_kg * kPoundsPerKilogram, 1e-6);
  const double oxidizer_lbs = std::max(cfg.oxidizer_mass_kg * kPoundsPerKilogram, 1e-6);
  const double mixture_ratio = oxidizer_lbs / fuel_lbs;
  const double max_thrust_lbf = std::max(cfg.thrust_newtons * kPoundsForcePerNewton, 0.0);
  const double total_flow_lbs_s = cfg.isp_s > 1e-6 ? max_thrust_lbf / cfg.isp_s : 0.0;
  const double fuel_flow_lbs_s = total_flow_lbs_s / (1.0 + mixture_ratio);
  const double oxidizer_flow_lbs_s = total_flow_lbs_s - fuel_flow_lbs_s;

  return {fuel_flow_lbs_s, oxidizer_flow_lbs_s, mixture_ratio};
}

static double compute_sea_level_pressure_psf(const AloeJsbsimConfig& cfg) {
  if (!std::isfinite(cfg.air_density_sea_level_kg_m3) || cfg.air_density_sea_level_kg_m3 <= 0.0) {
    return kStandardSeaLevelPressurePsf;
  }

  return kStandardSeaLevelPressurePsf * (cfg.air_density_sea_level_kg_m3 / kStandardSeaLevelDensityKgPerM3);
}

class AloeJsbsimQuietLogger final : public JSBSim::FGLogger {
 public:
  void Message(const std::string&) override {}
  void Flush(void) override {}
};

static double interpolate_thrust_newtons(const AloeJsbsimConfig& cfg, double t_s) {
  if (!std::isfinite(t_s) || t_s < 0.0) {
    return 0.0;
  }

  if (cfg.thrust_curve_len > 0 && cfg.thrust_curve_time_s != nullptr && cfg.thrust_curve_thrust_n != nullptr) {
    if (t_s <= cfg.thrust_curve_time_s[0]) {
      return std::max(0.0, cfg.thrust_curve_thrust_n[0]);
    }

    for (size_t index = 1; index < cfg.thrust_curve_len; ++index) {
      const double t0 = cfg.thrust_curve_time_s[index - 1];
      const double t1 = cfg.thrust_curve_time_s[index];
      const double f0 = cfg.thrust_curve_thrust_n[index - 1];
      const double f1 = cfg.thrust_curve_thrust_n[index];

      if (t_s > t1) {
        continue;
      }

      const double span = t1 - t0;
      if (span <= 1e-9) {
        return std::max(0.0, f1);
      }

      const double frac = (t_s - t0) / span;
      return std::max(0.0, f0 + frac * (f1 - f0));
    }

    return std::max(0.0, cfg.thrust_curve_thrust_n[cfg.thrust_curve_len - 1]);
  }

  return t_s <= cfg.burn_time_s ? std::max(0.0, cfg.thrust_newtons) : 0.0;
}

static bool landed_after_flight(double sim_time_s,
                                double launch_delay_s,
                                double launch_rod_length_m,
                                double altitude_m,
                                double max_altitude_m,
                                double vertical_speed_down_mps,
                                bool motor_enabled) {
  const double ground_altitude_tolerance_m = std::max(launch_rod_length_m, 10.0);
  const double min_airborne_altitude_m = std::max(launch_rod_length_m * 0.5, 5.0);
  const double max_upward_speed_near_ground_mps = 5.0;

  if (!motor_enabled) {
    return sim_time_s >= launch_delay_s + 0.5;
  }

  if (sim_time_s < launch_delay_s + 2.0) {
    return false;
  }

  return max_altitude_m >= min_airborne_altitude_m && altitude_m <= ground_altitude_tolerance_m &&
         vertical_speed_down_mps >= -max_upward_speed_near_ground_mps;
}

static bool constrained_on_launch_rail(double sim_time_s,
                                       double launch_delay_s,
                                       double launch_rod_length_m,
                                       bool motor_enabled,
                                       const JSBSim::FGPropagate& propagate) {
  return motor_enabled && sim_time_s >= launch_delay_s &&
         propagate.GetDistanceAGL() * kMetersPerFoot < launch_rod_length_m;
}

static void constrain_to_launch_rail(JSBSim::FGPropagate& propagate,
                                     const JSBSim::FGQuaternion& launch_rail_attitude_eci) {
  const double down_fps = propagate.GetVel(3);
  propagate.SetInertialOrientation(launch_rail_attitude_eci);
  propagate.SetInertialRates(JSBSim::FGColumnVector3(0.0, 0.0, 0.0));
  propagate.SetInertialVelocity(propagate.GetTl2i() * JSBSim::FGColumnVector3(0.0, 0.0, down_fps));
}

static std::string write_model_files(const AloeJsbsimConfig& cfg) {
  namespace fs = std::filesystem;

  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path root = fs::temp_directory_path() /
                        fs::path("aloe-jsbsim-runtime-" + std::to_string(now) + "-" +
                                 std::to_string(static_cast<std::uint64_t>(std::llround(cfg.thrust_newtons * 1000.0))));
  const fs::path aircraft_dir = root / fs::path("aircraft") / fs::path("aloe_rocket");
  const fs::path engine_dir = root / fs::path("engine");
  const fs::path systems_dir = root / fs::path("systems");

  fs::create_directories(aircraft_dir);
  fs::create_directories(engine_dir);
  fs::create_directories(systems_dir);

  const double emptywt_lbs = cfg.dry_mass_kg * kPoundsPerKilogram;
  const double fuel_lbs = std::max(cfg.fuel_mass_kg * kPoundsPerKilogram, 1e-6);
  const double oxidizer_lbs = std::max(cfg.oxidizer_mass_kg * kPoundsPerKilogram, 1e-6);
  const double total_reactant_kg = cfg.fuel_mass_kg + cfg.oxidizer_mass_kg;
  const double ixx = cfg.inertia_xx_kg_m2 * kSlugFt2PerKgM2;
  const double iyy = cfg.inertia_yy_kg_m2 * kSlugFt2PerKgM2;
  const double izz = cfg.inertia_zz_kg_m2 * kSlugFt2PerKgM2;
  const double dry_cg_in = cfg.cg_empty_m * kInchesPerMeter;
  double propellant_cg_m = cfg.cg_full_m;
  if (total_reactant_kg > 1e-6) {
    propellant_cg_m = ((cfg.cg_full_m * (cfg.dry_mass_kg + total_reactant_kg)) -
                       (cfg.cg_empty_m * cfg.dry_mass_kg)) /
                       total_reactant_kg;
  }
  const double propellant_cg_in = propellant_cg_m * kInchesPerMeter;
  const double cp_in = cfg.cp_location_m * kInchesPerMeter;
  const double nozzle_in = cfg.nozzle_location_m * kInchesPerMeter;
  const double area_ft2 = cfg.ref_area_m2 * kSquareFeetPerSquareMeter;
  const double body_diameter_ft = 2.0 * std::sqrt(std::max(area_ft2, 1e-6) / M_PI);
  const double span_ft = std::max(body_diameter_ft, 1e-3);
  const double chord_ft = span_ft;
  const double gravity_ft_s2 = std::max(cfg.gravity_mps2 * kFeetPerMeter, 1e-6);
  const double planet_gm_ft3_s2 = gravity_ft_s2 * kEarthRadiusFt * kEarthRadiusFt;
  const auto engine_profile = build_liquid_engine_profile(cfg);

  std::ostringstream aircraft;
  aircraft << "<?xml version=\"1.0\"?>\n";
  aircraft << "<fdm_config name=\"Aloe Rocket\" version=\"2.0\" release=\"Aloe\">\n";
  aircraft << "  <planet name=\"Aloe World\">\n";
  aircraft << "    <equatorial_radius unit=\"FT\">" << kEarthRadiusFt << "</equatorial_radius>\n";
  aircraft << "    <polar_radius unit=\"FT\">" << kEarthRadiusFt << "</polar_radius>\n";
  aircraft << "    <rotation_rate unit=\"RAD/SEC\">0.0</rotation_rate>\n";
  aircraft << "    <GM unit=\"FT3/SEC2\">" << planet_gm_ft3_s2 << "</GM>\n";
  aircraft << "    <J2>0</J2>\n";
  aircraft << "  </planet>\n";
  aircraft << "  <metrics>\n";
  aircraft << "    <wingarea unit=\"FT2\">" << area_ft2 << "</wingarea>\n";
  aircraft << "    <wingspan unit=\"FT\">" << span_ft << "</wingspan>\n";
  aircraft << "    <chord unit=\"FT\">" << chord_ft << "</chord>\n";
  aircraft << "    <htailarea unit=\"FT2\">0</htailarea>\n";
  aircraft << "    <htailarm unit=\"FT\">0</htailarm>\n";
  aircraft << "    <vtailarea unit=\"FT2\">0</vtailarea>\n";
  aircraft << "    <vtailarm unit=\"FT\">0</vtailarm>\n";
  aircraft << "    <location name=\"AERORP\" unit=\"IN\"><x>" << cp_in << "</x><y>0</y><z>0</z></location>\n";
  aircraft << "    <location name=\"EYEPOINT\" unit=\"IN\"><x>0</x><y>0</y><z>0</z></location>\n";
  aircraft << "    <location name=\"VRP\" unit=\"IN\"><x>0</x><y>0</y><z>0</z></location>\n";
  aircraft << "  </metrics>\n";
  aircraft << "  <mass_balance>\n";
  aircraft << "    <ixx unit=\"SLUG*FT2\">" << ixx << "</ixx>\n";
  aircraft << "    <iyy unit=\"SLUG*FT2\">" << iyy << "</iyy>\n";
  aircraft << "    <izz unit=\"SLUG*FT2\">" << izz << "</izz>\n";
  aircraft << "    <emptywt unit=\"LBS\">" << emptywt_lbs << "</emptywt>\n";
  aircraft << "    <location name=\"CG\" unit=\"IN\"><x>" << dry_cg_in << "</x><y>0</y><z>0</z></location>\n";
  aircraft << "  </mass_balance>\n";
  aircraft << "  <ground_reactions>\n";
  aircraft << "    <contact type=\"STRUCTURE\" name=\"PAD\">\n";
  aircraft << "      <location unit=\"IN\"><x>0</x><y>0</y><z>0</z></location>\n";
  aircraft << "      <static_friction>" << cfg.pad_static_friction << "</static_friction>\n";
  aircraft << "      <dynamic_friction>" << cfg.pad_dynamic_friction << "</dynamic_friction>\n";
  aircraft << "      <rolling_friction>0</rolling_friction>\n";
  aircraft << "      <spring_coeff unit=\"LBS/FT\">" << cfg.pad_spring_coeff_lbs_ft << "</spring_coeff>\n";
  aircraft << "      <damping_coeff unit=\"LBS/FT/SEC\">" << cfg.pad_damping_coeff_lbs_ft_s << "</damping_coeff>\n";
  aircraft << "      <max_steer unit=\"DEG\">0</max_steer>\n";
  aircraft << "      <brake_group>NONE</brake_group>\n";
  aircraft << "      <retractable>0</retractable>\n";
  aircraft << "    </contact>\n";
  aircraft << "  </ground_reactions>\n";
  aircraft << "  <propulsion>\n";
  aircraft << "    <engine file=\"aloe_rocket_engine\">\n";
  aircraft << "      <feed>0</feed>\n";
  aircraft << "      <feed>1</feed>\n";
  aircraft << "      <thruster file=\"aloe_rocket_nozzle\">\n";
  aircraft << "        <location unit=\"IN\"><x>" << nozzle_in << "</x><y>0</y><z>0</z></location>\n";
  aircraft << "        <orient unit=\"DEG\"><roll>0</roll><pitch>" << cfg.thrust_cant_deg << "</pitch><yaw>0</yaw></orient>\n";
  aircraft << "      </thruster>\n";
  aircraft << "    </engine>\n";
  aircraft << "    <tank type=\"FUEL\"><location unit=\"IN\"><x>" << propellant_cg_in << "</x><y>0</y><z>0</z></location><capacity unit=\"LBS\">" << fuel_lbs << "</capacity><contents unit=\"LBS\">" << fuel_lbs << "</contents><priority>1</priority></tank>\n";
  aircraft << "    <tank type=\"OXIDIZER\"><location unit=\"IN\"><x>" << propellant_cg_in << "</x><y>0</y><z>0</z></location><capacity unit=\"LBS\">" << oxidizer_lbs << "</capacity><contents unit=\"LBS\">" << oxidizer_lbs << "</contents><priority>1</priority></tank>\n";
  aircraft << "  </propulsion>\n";
  aircraft << "  <flight_control name=\"FCS\"></flight_control>\n";
  aircraft << "  <aerodynamics>\n";
  aircraft << "    <axis name=\"DRAG\">\n";
  aircraft << "      <function name=\"aero/drag\"><product><property>aero/qbar-psf</property><property>metrics/Sw-sqft</property><value>" << cfg.drag_coeff_axial << "</value></product></function>\n";
  aircraft << "    </axis>\n";
  aircraft << "    <axis name=\"SIDE\">\n";
  aircraft << "      <function name=\"aero/side\"><product><property>aero/qbar-psf</property><property>metrics/Sw-sqft</property><property>aero/beta-rad</property><value>" << cfg.normal_force_coeff << "</value></product></function>\n";
  aircraft << "    </axis>\n";
  aircraft << "    <axis name=\"LIFT\">\n";
  aircraft << "      <function name=\"aero/lift\"><product><property>aero/qbar-psf</property><property>metrics/Sw-sqft</property><property>aero/alpha-rad</property><value>" << cfg.normal_force_coeff << "</value></product></function>\n";
  aircraft << "    </axis>\n";
  aircraft << "  </aerodynamics>\n";
  aircraft << "</fdm_config>\n";

  std::ostringstream engine;
  engine << std::fixed << std::setprecision(6);
  engine << "<?xml version=\"1.0\"?>\n";
  engine << "<rocket_engine name=\"Aloe Rocket Engine\">\n";
  engine << "  <isp>" << cfg.isp_s << "</isp>\n";
  engine << "  <maxthrottle>1.0</maxthrottle>\n";
  engine << "  <minthrottle>0.000001</minthrottle>\n";
  engine << "  <slfuelflowmax unit=\"LBS/SEC\">" << engine_profile.fuel_flow_lbs_s << "</slfuelflowmax>\n";
  engine << "  <sloxiflowmax unit=\"LBS/SEC\">" << engine_profile.oxidizer_flow_lbs_s << "</sloxiflowmax>\n";
  engine << "  <mixtureratio>" << engine_profile.mixture_ratio << "</mixtureratio>\n";
  engine << "</rocket_engine>\n";

  std::ostringstream nozzle;
  nozzle << "<?xml version=\"1.0\"?>\n";
  nozzle << "<nozzle name=\"Aloe Nozzle\">\n";
  nozzle << "  <pe unit=\"PSF\">" << cfg.nozzle_exit_pressure_psf << "</pe>\n";
  nozzle << "  <area unit=\"FT2\">" << cfg.nozzle_area_ft2 << "</area>\n";
  nozzle << "</nozzle>\n";

  {
    std::ofstream aircraft_file(aircraft_dir / "aloe_rocket.xml");
    aircraft_file << aircraft.str();
  }
  {
    std::ofstream engine_file(engine_dir / "aloe_rocket_engine.xml");
    engine_file << engine.str();
  }
  {
    std::ofstream nozzle_file(engine_dir / "aloe_rocket_nozzle.xml");
    nozzle_file << nozzle.str();
  }

  return root.string();
}

template <typename T>
static bool output_ready(const T* ptr, size_t len) {
  return ptr != nullptr && len > 0;
}

}  // namespace

struct AloeJsbsimOpaque {
  std::string last_error;
};

extern "C" {

::AloeJsbsimOpaque* aloe_jsbsim_create(void) {
  return new ::AloeJsbsimOpaque();
}

void aloe_jsbsim_destroy(::AloeJsbsimOpaque* handle) {
  delete handle;
}

const char* aloe_jsbsim_last_error(const ::AloeJsbsimOpaque* handle) {
  if (!handle) {
    return "invalid handle";
  }
  return handle->last_error.c_str();
}

int aloe_jsbsim_run(::AloeJsbsimOpaque* handle, const AloeJsbsimConfig* config, AloeJsbsimOutput* output) {
  if (!handle || !config || !output) {
    return 1;
  }

  try {
    handle->last_error.clear();
    const bool motor_enabled = config->thrust_newtons > 1e-6 && config->isp_s > 1e-6 &&
                               config->fuel_mass_kg > 0.0 && config->oxidizer_mass_kg > 0.0;
    const std::string root_dir = write_model_files(*config);

    static auto quiet_logger = std::make_shared<AloeJsbsimQuietLogger>();
    JSBSim::SetLogger(quiet_logger);

    JSBSim::FGFDMExec exec;
    exec.SetDebugLevel(0);
    exec.DisableOutput();
    exec.SetRootDir(SGPath(root_dir));
    exec.SetAircraftPath(SGPath("aircraft"));
    exec.SetEnginePath(SGPath("engine"));
    exec.SetSystemsPath(SGPath("systems"));
    exec.Setdt(config->dt_s);

    if (!exec.LoadModel("aloe_rocket")) {
      handle->last_error = "failed to load generated JSBSim rocket model";
      return 2;
    }

    exec.GetAtmosphere()->SetPressureSL(
        JSBSim::FGAtmosphere::ePSF,
        compute_sea_level_pressure_psf(*config));

    auto ic = exec.GetIC();
    ic->InitializeIC();
    ic->SetTerrainElevationFtIC(0.0);
    ic->SetAltitudeAGLFtIC(0.0);
    ic->SetLatitudeDegIC(0.0);
    ic->SetLongitudeDegIC(0.0);
    ic->SetPhiRadIC(0.0);
    ic->SetThetaRadIC(M_PI_2);
    ic->SetPsiRadIC(0.0);
    ic->SetVNorthFpsIC(0.0);
    ic->SetVEastFpsIC(0.0);
    ic->SetVDownFpsIC(0.0);
    ic->SetPRadpsIC(config->spin_rate_deg_per_s * M_PI / 180.0);
    ic->SetQRadpsIC(0.0);
    ic->SetRRadpsIC(0.0);
    ic->SetWindNEDFpsIC(
        config->wind_north_mps * kFeetPerMeter,
        config->wind_east_mps * kFeetPerMeter,
        config->wind_down_mps * kFeetPerMeter);

    if (!exec.RunIC()) {
      handle->last_error = "JSBSim initial conditions failed";
      return 2;
    }

    const auto launch_rail_attitude_eci = exec.GetPropagate()->GetQuaternionECI();

    if (!output_ready(output->time_s, output->len) || !output_ready(output->pos_n_m, output->len) ||
        !output_ready(output->pos_e_m, output->len) || !output_ready(output->pos_d_m, output->len) ||
        !output_ready(output->vel_n_mps, output->len) || !output_ready(output->vel_e_mps, output->len) ||
        !output_ready(output->vel_d_mps, output->len) || !output_ready(output->accel_bx_mps2, output->len) ||
        !output_ready(output->accel_by_mps2, output->len) || !output_ready(output->accel_bz_mps2, output->len) ||
        !output_ready(output->p_rad_s, output->len) || !output_ready(output->q_rad_s, output->len) ||
        !output_ready(output->r_rad_s, output->len) || !output_ready(output->quat_w, output->len) ||
        !output_ready(output->quat_x, output->len) || !output_ready(output->quat_y, output->len) ||
        !output_ready(output->quat_z, output->len)) {
      return 3;
    }

    double pos_n_m = 0.0;
    double pos_e_m = 0.0;
    double pos_d_m = 0.0;
    double max_altitude_m = 0.0;
    const double max_thrust_n = std::max(config->thrust_newtons, 1e-6);
    size_t index = 0;
    while (index < output->len && exec.GetSimTime() <= config->max_time_s) {
      const double hold_down = (!motor_enabled || exec.GetSimTime() < config->launch_delay_s) ? 1.0 : 0.0;
      const double engine_time_s = exec.GetSimTime() - config->launch_delay_s;
      const double target_thrust_n = motor_enabled ? interpolate_thrust_newtons(*config, engine_time_s) : 0.0;
      const double throttle = std::clamp(target_thrust_n / max_thrust_n, 0.0, 1.0);
      exec.SetPropertyValue("forces/hold-down", hold_down);
      exec.SetPropertyValue("fcs/throttle-cmd-norm[0]", throttle);

      const auto propagate = exec.GetPropagate();
      if (constrained_on_launch_rail(exec.GetSimTime(),
                                     config->launch_delay_s,
                                     config->launch_rod_length_m,
                                     motor_enabled,
                                     *propagate)) {
        constrain_to_launch_rail(*propagate, launch_rail_attitude_eci);
      }
      const auto accelerations = exec.GetAccelerations();
      const auto quat = propagate->GetQuaternion();
      if (index > 0) {
        pos_n_m += propagate->GetVel(1) * kMetersPerFoot * config->dt_s;
        pos_e_m += propagate->GetVel(2) * kMetersPerFoot * config->dt_s;
        pos_d_m += propagate->GetVel(3) * kMetersPerFoot * config->dt_s;
      }

      const double sim_time_s = exec.GetSimTime();
      const double vel_n_mps = propagate->GetVel(1) * kMetersPerFoot;
      const double vel_e_mps = propagate->GetVel(2) * kMetersPerFoot;
      const double vel_d_mps = propagate->GetVel(3) * kMetersPerFoot;
      const double accel_bx_mps2 = accelerations->GetBodyAccel(1) * kMetersPerFoot;
      const double accel_by_mps2 = accelerations->GetBodyAccel(2) * kMetersPerFoot;
      const double accel_bz_mps2 = accelerations->GetBodyAccel(3) * kMetersPerFoot;
      const double p_rad_s = propagate->GetPQR(1);
      const double q_rad_s = propagate->GetPQR(2);
      const double r_rad_s = propagate->GetPQR(3);
      const double quat_w = quat(1);
      const double quat_x = quat(2);
      const double quat_y = quat(3);
      const double quat_z = quat(4);

      if (!std::isfinite(sim_time_s) || !std::isfinite(pos_n_m) || !std::isfinite(pos_e_m) ||
          !std::isfinite(pos_d_m) || !std::isfinite(vel_n_mps) || !std::isfinite(vel_e_mps) ||
          !std::isfinite(vel_d_mps) || !std::isfinite(accel_bx_mps2) ||
          !std::isfinite(accel_by_mps2) || !std::isfinite(accel_bz_mps2) ||
          !std::isfinite(p_rad_s) || !std::isfinite(q_rad_s) || !std::isfinite(r_rad_s) ||
          !std::isfinite(quat_w) || !std::isfinite(quat_x) || !std::isfinite(quat_y) ||
          !std::isfinite(quat_z)) {
        handle->last_error = "non-finite JSBSim state encountered";
        break;
      }

      const double altitude_m = -pos_d_m;
      max_altitude_m = std::max(max_altitude_m, altitude_m);

      output->time_s[index] = sim_time_s;
      output->pos_n_m[index] = pos_n_m;
      output->pos_e_m[index] = pos_e_m;
      output->pos_d_m[index] = pos_d_m;
      output->vel_n_mps[index] = vel_n_mps;
      output->vel_e_mps[index] = vel_e_mps;
      output->vel_d_mps[index] = vel_d_mps;
      output->accel_bx_mps2[index] = accel_bx_mps2;
      output->accel_by_mps2[index] = accel_by_mps2;
      output->accel_bz_mps2[index] = accel_bz_mps2;
      output->p_rad_s[index] = p_rad_s;
      output->q_rad_s[index] = q_rad_s;
      output->r_rad_s[index] = r_rad_s;
      output->quat_w[index] = quat_w;
      output->quat_x[index] = quat_x;
      output->quat_y[index] = quat_y;
      output->quat_z[index] = quat_z;

      ++index;

      if (landed_after_flight(exec.GetSimTime(),
                              config->launch_delay_s,
                              config->launch_rod_length_m,
                              altitude_m,
                              max_altitude_m,
                              propagate->GetVel(3) * kMetersPerFoot,
                              motor_enabled)) {
        break;
      }

      if (!exec.Run()) {
        handle->last_error = "JSBSim simulation step failed";
        break;
      }
    }

    output->len = index;
    if (index == 0) {
      handle->last_error = "JSBSim produced no samples";
      return 2;
    }
    return 0;
  } catch (const std::exception& ex) {
    handle->last_error = ex.what();
    return 4;
  } catch (...) {
    handle->last_error = "unknown C++ exception";
    return 5;
  }
}

}  // extern "C"
