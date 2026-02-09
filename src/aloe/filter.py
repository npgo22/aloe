"""Kalman-filter + quantization pipeline.

This module takes a simulation DataFrame (with sensor columns), runs the
Rust ES-EKF, quantizes the output to the on-wire telemetry format, then
dequantizes and computes error statistics against the ground-truth trajectory.

All heavy computation happens in Rust via the ``aloe_core`` native extension
(built with maturin).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Try importing the native module — fall back to a stub for import-time safety
# (e.g. so that the GUI can still start if the Rust lib hasn't been compiled).
# ---------------------------------------------------------------------------
try:
    from aloe.aloe_core import (
        run_eskf_on_arrays,
        quantize_flight_array,
        dequantize_flight_array,
        quantize_recovery_array,
        dequantize_recovery_array,
        detect_flight_states,
        detect_truth_states,
    )

    _HAS_NATIVE = True
except ImportError:
    _HAS_NATIVE = False


def _require_native():
    if not _HAS_NATIVE:
        raise RuntimeError(
            "aloe_core native extension not available. "
            "Build it with: maturin develop --release  (from the project root)"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class FilterConfig:
    """Knobs for the ESKF + quantization pipeline."""

    ground_pressure_mbar: float = 1013.25
    mag_declination_deg: float = 0.0
    # Home location (for recovery-packet GPS quantization)
    home_lat_deg: float = 35.0
    home_lon_deg: float = -106.0
    home_alt_m: float = 1500.0


# ---------------------------------------------------------------------------
# NED ↔ Geodetic helpers (Python side, for generating lat/lon from sim)
# ---------------------------------------------------------------------------
_EARTH_R = 6_371_000.0


def _ned_to_geodetic(n: float, e: float, d: float, lat0: float, lon0: float, alt0: float) -> tuple[float, float, float]:
    """Convert NED offset (m) to lat/lon/alt (deg, deg, m MSL)."""
    lat = lat0 + math.degrees(n / _EARTH_R)
    lon = lon0 + math.degrees(e / (_EARTH_R * math.cos(math.radians(lat0))))
    alt = alt0 - d
    return lat, lon, alt


# State label lookup
_STATE_LABELS = ["pad", "ignition", "burn", "coasting", "apogee", "recovery"]


def _state_label(s: int) -> str:
    return _STATE_LABELS[s] if 0 <= s < len(_STATE_LABELS) else "unknown"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_filter_pipeline(
    df: pl.DataFrame,
    cfg: FilterConfig | None = None,
) -> pl.DataFrame:
    """Run ES-EKF on a sim+sensor DataFrame and return augmented results.

    New columns added:
      - eskf_pos_{n,e,d}      — filtered position (NED, m)
      - eskf_vel_{n,e,d}      — filtered velocity (NED, m/s)
      - eskf_q{w,x,y,z}       — filtered orientation quaternion

      - q_flight_pos_{n,e}_m   — quantized flight position (i16 → f32)
      - q_flight_alt_m         — quantized flight altitude  (i32 cm → f32 m)
      - q_flight_vel_{n,e,d}_ms — quantized flight velocity
      - q_flight_{roll,pitch,yaw}_deg — quantized attitude

      - q_recovery_lat_deg, q_recovery_lon_deg, q_recovery_alt_m — recovery packet

    Parameters
    ----------
    df : pl.DataFrame
        Output of ``simulate_rocket`` + ``add_sensor_data``.
    cfg : FilterConfig, optional
        Filter configuration.  Defaults to standard sea-level / mid-latitude.
    """
    _require_native()
    if cfg is None:
        cfg = FilterConfig()

    n = len(df)
    times = df["time_s"].to_numpy().astype(np.float32)

    # --- Prepare sensor columns (replace null → NaN for Rust) ---
    def _col_f32(name: str) -> np.ndarray:
        if name in df.columns:
            s = df[name].to_numpy().astype(np.float64)
            return np.where(np.isnan(s), np.nan, s).astype(np.float32)
        return np.full(n, np.nan, dtype=np.float32)

    # ── Coordinate mapping ────────────────────────────────────────
    # Sim uses XYZ where Y = up (altitude).
    # ESKF uses NED:  N = sim_X,  E = sim_Z,  D = -sim_Y.
    #
    # Accelerometer specific-force conversion:
    #   The sim stores *coordinate* acceleration (includes gravity).
    #   A real IMU measures *specific force* = accel - gravity.
    #   In NED, gravity is [0, 0, +g], so:
    #       a_specific = coord_accel_NED - [0, 0, g]
    #   coord_accel_NED = [ax, az, -ay]  (from sim XYZ)
    #   a_specific_D = -ay - g
    # ──────────────────────────────────────────────────────────────

    GRAVITY = np.float32(9.80665)

    # Gyro (convert °/s → rad/s, remap axes X→N, Z→E, Y→-D)
    deg2rad = np.float32(math.pi / 180.0)
    gyro_raw_x = _col_f32("bmi088_gyro_x_dps")  # sim pitch rate → N axis
    gyro_raw_y = _col_f32("bmi088_gyro_y_dps")  # sim yaw rate → E axis
    gyro_raw_z = _col_f32("bmi088_gyro_z_dps")  # sim roll rate → -D axis
    gyro_n = gyro_raw_x * deg2rad
    gyro_e = gyro_raw_z * deg2rad
    gyro_d = -gyro_raw_y * deg2rad

    # Low-g accel: convert coord accel → specific force in NED
    al_raw_x = _col_f32("bmi088_accel_x_ms2")
    al_raw_y = _col_f32("bmi088_accel_y_ms2")
    al_raw_z = _col_f32("bmi088_accel_z_ms2")
    al_n = al_raw_x  # N = sim_X
    al_e = al_raw_z  # E = sim_Z
    al_d = np.where(np.isnan(al_raw_y), np.nan, -al_raw_y - GRAVITY)  # D = -(sim_Y) - g

    # High-g accel: same transform
    ah_raw_x = _col_f32("adxl375_accel_x_ms2")
    ah_raw_y = _col_f32("adxl375_accel_y_ms2")
    ah_raw_z = _col_f32("adxl375_accel_z_ms2")
    ah_n = ah_raw_x
    ah_e = ah_raw_z
    ah_d = np.where(np.isnan(ah_raw_y), np.nan, -ah_raw_y - GRAVITY)

    # Baro (mbar) — no coordinate transform needed
    baro = _col_f32("ms5611_pressure_mbar")

    # Mag (gauss — remap to NED body frame, then normalised in Rust)
    mx_raw = _col_f32("lis3mdl_mag_x_gauss")
    my_raw = _col_f32("lis3mdl_mag_y_gauss")
    mz_raw = _col_f32("lis3mdl_mag_z_gauss")
    mx = mx_raw  # N = sim_X
    my = mz_raw  # E = sim_Z
    mz = np.where(np.isnan(my_raw), np.nan, -my_raw)  # D = -sim_Y

    # GPS position (remap: sim_X→N, sim_Z→E, -sim_Y→D)
    gps_raw_x = _col_f32("gps_pos_x_m")
    gps_raw_y = _col_f32("gps_pos_y_m")  # altitude
    gps_raw_z = _col_f32("gps_pos_z_m")
    gps_pn = gps_raw_x
    gps_pe = gps_raw_z
    gps_pd = np.where(np.isnan(gps_raw_y), np.nan, -gps_raw_y)

    # GPS velocity (same remap)
    gps_raw_vx = _col_f32("gps_vel_x_ms")
    gps_raw_vy = _col_f32("gps_vel_y_ms")
    gps_raw_vz = _col_f32("gps_vel_z_ms")
    gps_vn = gps_raw_vx
    gps_ve = gps_raw_vz
    gps_vd = np.where(np.isnan(gps_raw_vy), np.nan, -gps_raw_vy)

    # --- Run the Rust ES-EKF ---
    (
        _t,
        pn,
        pe,
        pd,
        vn,
        ve,
        vd,
        qw,
        qx,
        qy,
        qk,
    ) = run_eskf_on_arrays(
        times.tolist(),
        gyro_n.tolist(),
        gyro_e.tolist(),
        gyro_d.tolist(),
        al_n.tolist(),
        al_e.tolist(),
        al_d.tolist(),
        ah_n.tolist(),
        ah_e.tolist(),
        ah_d.tolist(),
        baro.tolist(),
        mx.tolist(),
        my.tolist(),
        mz.tolist(),
        gps_pn.tolist(),
        gps_pe.tolist(),
        gps_pd.tolist(),
        gps_vn.tolist(),
        gps_ve.tolist(),
        gps_vd.tolist(),
        cfg.ground_pressure_mbar,
        cfg.mag_declination_deg,
    )

    # Add ESKF columns
    df = df.with_columns(
        [
            pl.Series("eskf_pos_n", pn, dtype=pl.Float32),
            pl.Series("eskf_pos_e", pe, dtype=pl.Float32),
            pl.Series("eskf_pos_d", pd, dtype=pl.Float32),
            pl.Series("eskf_vel_n", vn, dtype=pl.Float32),
            pl.Series("eskf_vel_e", ve, dtype=pl.Float32),
            pl.Series("eskf_vel_d", vd, dtype=pl.Float32),
            pl.Series("eskf_qw", qw, dtype=pl.Float32),
            pl.Series("eskf_qx", qx, dtype=pl.Float32),
            pl.Series("eskf_qy", qy, dtype=pl.Float32),
            pl.Series("eskf_qz", qk, dtype=pl.Float32),
        ]
    )

    # --- Compute Euler angles from quaternion (for quantization) ---
    qw_a = np.array(qw, dtype=np.float32)
    qx_a = np.array(qx, dtype=np.float32)
    qy_a = np.array(qy, dtype=np.float32)
    qz_a = np.array(qk, dtype=np.float32)

    # Roll (x), Pitch (y), Yaw (z)  — standard aerospace ZYX convention
    sinr_cosp = 2.0 * (qw_a * qx_a + qy_a * qz_a)
    cosr_cosp = 1.0 - 2.0 * (qx_a * qx_a + qy_a * qy_a)
    roll_deg = np.degrees(np.arctan2(sinr_cosp, cosr_cosp)).astype(np.float32)

    sinp = 2.0 * (qw_a * qy_a - qz_a * qx_a)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch_deg = np.degrees(np.arcsin(sinp)).astype(np.float32)

    siny_cosp = 2.0 * (qw_a * qz_a + qx_a * qy_a)
    cosy_cosp = 1.0 - 2.0 * (qy_a * qy_a + qz_a * qz_a)
    yaw_deg = np.degrees(np.arctan2(siny_cosp, cosy_cosp)).astype(np.float32)

    # --- Quantize Flight Data ---
    pn_a = np.array(pn, dtype=np.float32)
    pe_a = np.array(pe, dtype=np.float32)
    pd_a = np.array(pd, dtype=np.float32)
    vn_a = np.array(vn, dtype=np.float32)
    ve_a = np.array(ve, dtype=np.float32)
    vd_a = np.array(vd, dtype=np.float32)

    # Altitude AGL = -pos_d (NED down → AGL up)
    alt_agl = -pd_a

    q_pn, q_pe, q_alt_cm, q_vn, q_ve, q_vd, q_roll, q_pitch, q_yaw = quantize_flight_array(
        pn_a.tolist(),
        pe_a.tolist(),
        alt_agl.tolist(),
        vn_a.tolist(),
        ve_a.tolist(),
        vd_a.tolist(),
        roll_deg.tolist(),
        pitch_deg.tolist(),
        yaw_deg.tolist(),
    )

    # Dequantize to measure round-trip error
    dq_pn, dq_pe, dq_alt, dq_vn, dq_ve, dq_vd, dq_roll, dq_pitch, dq_yaw = dequantize_flight_array(
        q_pn,
        q_pe,
        q_alt_cm,
        q_vn,
        q_ve,
        q_vd,
        q_roll,
        q_pitch,
        q_yaw,
    )

    df = df.with_columns(
        [
            pl.Series("q_flight_pos_n_m", dq_pn, dtype=pl.Float32),
            pl.Series("q_flight_pos_e_m", dq_pe, dtype=pl.Float32),
            pl.Series("q_flight_alt_m", dq_alt, dtype=pl.Float32),
            pl.Series("q_flight_vel_n_ms", dq_vn, dtype=pl.Float32),
            pl.Series("q_flight_vel_e_ms", dq_ve, dtype=pl.Float32),
            pl.Series("q_flight_vel_d_ms", dq_vd, dtype=pl.Float32),
            pl.Series("q_flight_roll_deg", dq_roll, dtype=pl.Float32),
            pl.Series("q_flight_pitch_deg", dq_pitch, dtype=pl.Float32),
            pl.Series("q_flight_yaw_deg", dq_yaw, dtype=pl.Float32),
        ]
    )

    # --- Quantize Recovery Data (lat/lon/alt from NED) ---
    lat_arr = np.empty(n, dtype=np.float32)
    lon_arr = np.empty(n, dtype=np.float32)
    alt_msl_arr = np.empty(n, dtype=np.float32)
    for i in range(n):
        lat, lon, alt = _ned_to_geodetic(
            pn_a[i],
            pe_a[i],
            pd_a[i],
            cfg.home_lat_deg,
            cfg.home_lon_deg,
            cfg.home_alt_m,
        )
        lat_arr[i] = lat
        lon_arr[i] = lon
        alt_msl_arr[i] = alt

    qlat, qlon, qalt_m = quantize_recovery_array(
        lat_arr.tolist(),
        lon_arr.tolist(),
        alt_msl_arr.tolist(),
    )
    dqlat, dqlon, dqalt = dequantize_recovery_array(qlat, qlon, qalt_m)

    df = df.with_columns(
        [
            pl.Series("eskf_lat_deg", lat_arr.tolist(), dtype=pl.Float32),
            pl.Series("eskf_lon_deg", lon_arr.tolist(), dtype=pl.Float32),
            pl.Series("eskf_alt_msl_m", alt_msl_arr.tolist(), dtype=pl.Float32),
            pl.Series("q_recovery_lat_deg", dqlat, dtype=pl.Float32),
            pl.Series("q_recovery_lon_deg", dqlon, dtype=pl.Float32),
            pl.Series("q_recovery_alt_m", dqalt, dtype=pl.Float32),
        ]
    )

    # --- Detect flight states (Rust state machine) ---
    # ESKF state detection (from filter velocity estimates)
    eskf_state_arr, eskf_trans = detect_flight_states(
        times.tolist(),
        list(vn),
        list(ve),
        list(vd),
    )
    # Truth state detection (from sim ground-truth data)
    truth_accel_y = df["acceleration_y_ms2"].to_numpy().astype(np.float32)
    truth_vel_y = df["velocity_y_ms"].to_numpy().astype(np.float32)
    truth_thrust = df["thrust_N"].to_numpy().astype(np.float32)
    truth_state_arr, truth_trans = detect_truth_states(
        times.tolist(),
        truth_accel_y.tolist(),
        truth_vel_y.tolist(),
        truth_thrust.tolist(),
    )

    # Per-sample state columns
    df = df.with_columns(
        [
            pl.Series("truth_state", [_state_label(s) for s in truth_state_arr], dtype=pl.Utf8),
            pl.Series("truth_state_id", truth_state_arr, dtype=pl.UInt8),
            pl.Series("eskf_state", [_state_label(s) for s in eskf_state_arr], dtype=pl.Utf8),
            pl.Series("eskf_state_id", eskf_state_arr, dtype=pl.UInt8),
        ]
    )

    # Transition time columns (scalar broadcast to all rows)
    state_names = ["pad", "ignition", "burn", "coasting", "apogee", "recovery"]
    for i, name in enumerate(state_names):
        truth_t = float(truth_trans[i])
        eskf_t = float(eskf_trans[i])
        df = df.with_columns(
            [
                pl.lit(truth_t).cast(pl.Float32).alias(f"truth_{name}_time"),
                pl.lit(eskf_t).cast(pl.Float32).alias(f"eskf_{name}_time"),
            ]
        )

    return df


# ---------------------------------------------------------------------------
# Error statistics
# ---------------------------------------------------------------------------


@dataclass
class ErrorStats:
    """Error statistics for a single axis / quantity."""

    name: str
    mean_err: float = 0.0
    std_err: float = 0.0
    max_abs_err: float = 0.0
    rmse: float = 0.0
    p95_abs_err: float = 0.0
    n_samples: int = 0


def _compute_stats(name: str, truth: np.ndarray, estimate: np.ndarray) -> ErrorStats:
    err = estimate - truth
    abs_err = np.abs(err)
    return ErrorStats(
        name=name,
        mean_err=float(np.mean(err)),
        std_err=float(np.std(err)),
        max_abs_err=float(np.max(abs_err)),
        rmse=float(np.sqrt(np.mean(err**2))),
        p95_abs_err=float(np.percentile(abs_err, 95)),
        n_samples=len(err),
    )


def compute_error_report(df: pl.DataFrame) -> pl.DataFrame:
    """Compute error statistics comparing truth, ESKF, and quantized outputs.

    Returns a Polars DataFrame with one row per metric per stage.
    Stages: ``eskf`` (filter vs truth), ``quantized`` (quant vs truth).
    """
    truth_pos_x = df["position_x_m"].to_numpy().astype(np.float32)
    truth_alt = df["altitude_m"].to_numpy().astype(np.float32)
    truth_pos_z = df["position_z_m"].to_numpy().astype(np.float32)
    truth_vel_x = df["velocity_x_ms"].to_numpy().astype(np.float32)
    truth_vel_y = df["velocity_y_ms"].to_numpy().astype(np.float32)
    truth_vel_z = df["velocity_z_ms"].to_numpy().astype(np.float32)

    # The sim uses XYZ where Y=up; the ESKF uses NED.
    # Mapping: sim_x → N, sim_y (alt) → -D (up), sim_z → E
    # So: truth_N = truth_pos_x, truth_E = truth_pos_z, truth_D = -truth_alt
    truth_n = truth_pos_x
    truth_e = truth_pos_z
    truth_d = -truth_alt
    truth_vn = truth_vel_x
    truth_ve = truth_vel_z
    truth_vd = -truth_vel_y

    rows: list[dict] = []

    def _add(stage: str, stats: ErrorStats):
        rows.append(
            {
                "stage": stage,
                "quantity": stats.name,
                "mean_error": stats.mean_err,
                "std_error": stats.std_err,
                "max_abs_error": stats.max_abs_err,
                "rmse": stats.rmse,
                "p95_abs_error": stats.p95_abs_err,
                "n_samples": stats.n_samples,
            }
        )

    # ── ESKF vs Truth ─────────────────────────────────────────────
    if "eskf_pos_n" in df.columns:
        ekf_pn = df["eskf_pos_n"].to_numpy().astype(np.float32)
        ekf_pe = df["eskf_pos_e"].to_numpy().astype(np.float32)
        ekf_pd = df["eskf_pos_d"].to_numpy().astype(np.float32)
        ekf_vn = df["eskf_vel_n"].to_numpy().astype(np.float32)
        ekf_ve = df["eskf_vel_e"].to_numpy().astype(np.float32)
        ekf_vd = df["eskf_vel_d"].to_numpy().astype(np.float32)

        _add("eskf", _compute_stats("pos_n (m)", truth_n, ekf_pn))
        _add("eskf", _compute_stats("pos_e (m)", truth_e, ekf_pe))
        _add("eskf", _compute_stats("pos_d (m)", truth_d, ekf_pd))
        _add("eskf", _compute_stats("vel_n (m/s)", truth_vn, ekf_vn))
        _add("eskf", _compute_stats("vel_e (m/s)", truth_ve, ekf_ve))
        _add("eskf", _compute_stats("vel_d (m/s)", truth_vd, ekf_vd))

        # 3D position error
        pos_err_3d = np.sqrt((ekf_pn - truth_n) ** 2 + (ekf_pe - truth_e) ** 2 + (ekf_pd - truth_d) ** 2)
        _add(
            "eskf",
            ErrorStats(
                name="pos_3d (m)",
                mean_err=float(np.mean(pos_err_3d)),
                std_err=float(np.std(pos_err_3d)),
                max_abs_err=float(np.max(pos_err_3d)),
                rmse=float(np.sqrt(np.mean(pos_err_3d**2))),
                p95_abs_err=float(np.percentile(pos_err_3d, 95)),
                n_samples=len(pos_err_3d),
            ),
        )

    # ── Quantized Flight vs Truth ──────────────────────────────────
    if "q_flight_pos_n_m" in df.columns:
        qpn = df["q_flight_pos_n_m"].to_numpy().astype(np.float32)
        qpe = df["q_flight_pos_e_m"].to_numpy().astype(np.float32)
        qalt = df["q_flight_alt_m"].to_numpy().astype(np.float32)
        qvn = df["q_flight_vel_n_ms"].to_numpy().astype(np.float32)
        qve = df["q_flight_vel_e_ms"].to_numpy().astype(np.float32)
        qvd = df["q_flight_vel_d_ms"].to_numpy().astype(np.float32)

        _add("quantized_flight", _compute_stats("pos_n (m)", truth_n, qpn))
        _add("quantized_flight", _compute_stats("pos_e (m)", truth_e, qpe))
        _add("quantized_flight", _compute_stats("alt_agl (m)", truth_alt, qalt))
        _add("quantized_flight", _compute_stats("vel_n (m/s)", truth_vn, qvn))
        _add("quantized_flight", _compute_stats("vel_e (m/s)", truth_ve, qve))
        _add("quantized_flight", _compute_stats("vel_d (m/s)", truth_vd, qvd))

        # 3D position error (using alt_agl = -pos_d)
        q_pd = -qalt
        qpos_err_3d = np.sqrt((qpn - truth_n) ** 2 + (qpe - truth_e) ** 2 + (q_pd - truth_d) ** 2)
        _add(
            "quantized_flight",
            ErrorStats(
                name="pos_3d (m)",
                mean_err=float(np.mean(qpos_err_3d)),
                std_err=float(np.std(qpos_err_3d)),
                max_abs_err=float(np.max(qpos_err_3d)),
                rmse=float(np.sqrt(np.mean(qpos_err_3d**2))),
                p95_abs_err=float(np.percentile(qpos_err_3d, 95)),
                n_samples=len(qpos_err_3d),
            ),
        )

    # ── Quantization-only error (filter output vs quant round-trip) ─
    if "eskf_pos_n" in df.columns and "q_flight_pos_n_m" in df.columns:
        ekf_pn_rt = df["eskf_pos_n"].to_numpy().astype(np.float32)
        ekf_pe_rt = df["eskf_pos_e"].to_numpy().astype(np.float32)
        ekf_pd_rt = df["eskf_pos_d"].to_numpy().astype(np.float32)
        ekf_vn_rt = df["eskf_vel_n"].to_numpy().astype(np.float32)
        ekf_ve_rt = df["eskf_vel_e"].to_numpy().astype(np.float32)
        ekf_vd_rt = df["eskf_vel_d"].to_numpy().astype(np.float32)
        qpn_rt = df["q_flight_pos_n_m"].to_numpy().astype(np.float32)
        qpe_rt = df["q_flight_pos_e_m"].to_numpy().astype(np.float32)
        qalt_rt = df["q_flight_alt_m"].to_numpy().astype(np.float32)
        qvn_rt = df["q_flight_vel_n_ms"].to_numpy().astype(np.float32)
        qve_rt = df["q_flight_vel_e_ms"].to_numpy().astype(np.float32)
        qvd_rt = df["q_flight_vel_d_ms"].to_numpy().astype(np.float32)
        _add("quant_roundtrip", _compute_stats("pos_n (m)", ekf_pn_rt, qpn_rt))
        _add("quant_roundtrip", _compute_stats("pos_e (m)", ekf_pe_rt, qpe_rt))
        _add("quant_roundtrip", _compute_stats("alt_agl (m)", -ekf_pd_rt, qalt_rt))
        _add("quant_roundtrip", _compute_stats("vel_n (m/s)", ekf_vn_rt, qvn_rt))
        _add("quant_roundtrip", _compute_stats("vel_e (m/s)", ekf_ve_rt, qve_rt))
        _add("quant_roundtrip", _compute_stats("vel_d (m/s)", ekf_vd_rt, qvd_rt))

    # ── Recovery packet quantization error ───────────────────────
    if "eskf_lat_deg" in df.columns and "q_recovery_lat_deg" in df.columns:
        eskf_lat = df["eskf_lat_deg"].to_numpy().astype(np.float32)
        eskf_lon = df["eskf_lon_deg"].to_numpy().astype(np.float32)
        eskf_alt_msl = df["eskf_alt_msl_m"].to_numpy().astype(np.float32)
        qrlat = df["q_recovery_lat_deg"].to_numpy().astype(np.float32)
        qrlon = df["q_recovery_lon_deg"].to_numpy().astype(np.float32)
        qralt = df["q_recovery_alt_m"].to_numpy().astype(np.float32)

        _add("quant_recovery", _compute_stats("lat (deg)", eskf_lat, qrlat))
        _add("quant_recovery", _compute_stats("lon (deg)", eskf_lon, qrlon))
        _add("quant_recovery", _compute_stats("alt_msl (m)", eskf_alt_msl, qralt))

        # Convert lat/lon error to metres for interpretability
        lat_err_m = (qrlat - eskf_lat) * (np.pi / 180.0) * _EARTH_R
        lon_err_m = (qrlon - eskf_lon) * (np.pi / 180.0) * _EARTH_R * np.cos(np.radians(eskf_lat))
        horiz_err_m = np.sqrt(lat_err_m**2 + lon_err_m**2)
        _add(
            "quant_recovery",
            ErrorStats(
                name="horiz_pos (m)",
                mean_err=float(np.mean(horiz_err_m)),
                std_err=float(np.std(horiz_err_m)),
                max_abs_err=float(np.max(horiz_err_m)),
                rmse=float(np.sqrt(np.mean(horiz_err_m**2))),
                p95_abs_err=float(np.percentile(horiz_err_m, 95)),
                n_samples=len(horiz_err_m),
            ),
        )

    # ── State detection timing errors ──────────────────────────────
    state_names_detect = ["ignition", "burn", "coasting", "apogee", "recovery"]
    for name in state_names_detect:
        truth_col = f"truth_{name}_time"
        eskf_col = f"eskf_{name}_time"
        if truth_col in df.columns and eskf_col in df.columns:
            truth_t = float(df[truth_col][0])
            eskf_t = float(df[eskf_col][0])
            if not np.isnan(truth_t) and not np.isnan(eskf_t):
                err = eskf_t - truth_t
                rows.append(
                    {
                        "stage": "state_detection",
                        "quantity": f"{name}_time_error (s)",
                        "mean_error": err,
                        "std_error": 0.0,
                        "max_abs_error": abs(err),
                        "rmse": abs(err),
                        "p95_abs_error": abs(err),
                        "n_samples": 1,
                    }
                )
            else:
                rows.append(
                    {
                        "stage": "state_detection",
                        "quantity": f"{name}_time_error (s)",
                        "mean_error": None,
                        "std_error": None,
                        "max_abs_error": None,
                        "rmse": None,
                        "p95_abs_error": None,
                        "n_samples": 0,
                    }
                )

    return pl.DataFrame(rows)


def write_error_report_xlsx(
    error_df: pl.DataFrame,
    flight_df: pl.DataFrame,
    path: str,
) -> None:
    """Write a multi-sheet XLSX workbook with flight data + error stats."""
    import xlsxwriter

    with xlsxwriter.Workbook(path, {"nan_inf_to_errors": True}) as wb:
        # Sheet 1: Error summary
        ws_err = wb.add_worksheet("Error Statistics")
        bold = wb.add_format({"bold": True})
        cols = error_df.columns
        for c, col in enumerate(cols):
            ws_err.write(0, c, col, bold)
        for r, row in enumerate(error_df.iter_rows()):
            for c, val in enumerate(row):
                ws_err.write(r + 1, c, val)
        ws_err.autofit()

        # Sheet 2: Positions (truth, ESKF, quantized)
        pos_cols = [
            "time_s",
            "position_x_m",
            "altitude_m",
            "position_z_m",
        ]
        eskf_pos_cols = ["eskf_pos_n", "eskf_pos_e", "eskf_pos_d"]
        q_pos_cols = ["q_flight_pos_n_m", "q_flight_pos_e_m", "q_flight_alt_m"]
        recovery_cols = [
            "eskf_lat_deg",
            "eskf_lon_deg",
            "eskf_alt_msl_m",
            "q_recovery_lat_deg",
            "q_recovery_lon_deg",
            "q_recovery_alt_m",
        ]
        all_pos = pos_cols + [c for c in eskf_pos_cols + q_pos_cols + recovery_cols if c in flight_df.columns]
        pos_df = flight_df.select([c for c in all_pos if c in flight_df.columns])

        ws_pos = wb.add_worksheet("Positions")
        for c, col in enumerate(pos_df.columns):
            ws_pos.write(0, c, col, bold)
        for r, row in enumerate(pos_df.iter_rows()):
            for c, val in enumerate(row):
                if val is not None:
                    ws_pos.write(r + 1, c, val)
        ws_pos.autofit()

        # Sheet 3: Velocities
        vel_cols = ["time_s", "velocity_x_ms", "velocity_y_ms", "velocity_z_ms"]
        eskf_vel_cols = ["eskf_vel_n", "eskf_vel_e", "eskf_vel_d"]
        q_vel_cols = ["q_flight_vel_n_ms", "q_flight_vel_e_ms", "q_flight_vel_d_ms"]
        all_vel = vel_cols + [c for c in eskf_vel_cols + q_vel_cols if c in flight_df.columns]
        vel_df = flight_df.select([c for c in all_vel if c in flight_df.columns])

        ws_vel = wb.add_worksheet("Velocities")
        for c, col in enumerate(vel_df.columns):
            ws_vel.write(0, c, col, bold)
        for r, row in enumerate(vel_df.iter_rows()):
            for c, val in enumerate(row):
                if val is not None:
                    ws_vel.write(r + 1, c, val)
        ws_vel.autofit()

        # Sheet 4: Attitude (quaternion + quantized Euler)
        att_cols = ["time_s"]
        att_cols += [c for c in ["eskf_qw", "eskf_qx", "eskf_qy", "eskf_qz"] if c in flight_df.columns]
        att_cols += [
            c for c in ["q_flight_roll_deg", "q_flight_pitch_deg", "q_flight_yaw_deg"] if c in flight_df.columns
        ]
        att_df = flight_df.select([c for c in att_cols if c in flight_df.columns])

        ws_att = wb.add_worksheet("Attitude")
        for c, col in enumerate(att_df.columns):
            ws_att.write(0, c, col, bold)
        for r, row in enumerate(att_df.iter_rows()):
            for c, val in enumerate(row):
                if val is not None:
                    ws_att.write(r + 1, c, val)
        ws_att.autofit()
