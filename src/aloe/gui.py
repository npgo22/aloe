import io
import hashlib
from dataclasses import fields
from typing import Any, Union
from nicegui import run, ui
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

from aloe.sim import (
    RocketParams,
    SensorConfig,
    PRESETS,
    simulate_rocket,
    add_sensor_data,
)
from aloe.params import (
    get_rocket_sliders,
    get_env_sliders,
    get_sensor_rate_sliders,
    get_sensor_latency_sliders,
)

try:
    from aloe.filter import run_filter_pipeline, compute_error_report, _HAS_NATIVE

    _CAN_FILTER = _HAS_NATIVE
except Exception:
    _CAN_FILTER = False

# Slider config: (label, attr, min, max, step)
ROCKET_SLIDERS = get_rocket_sliders()
ENV_SLIDERS = get_env_sliders()
SENSOR_RATE_SLIDERS = get_sensor_rate_sliders()
SENSOR_LATENCY_SLIDERS = get_sensor_latency_sliders()


def create_3d_figure(df, sensor_cfg: SensorConfig | None = None, max_time=None):
    """Create 3D flight path with truth, ESKF, and quantized overlays."""
    import math as _math

    if max_time is not None:
        df = df.filter(pl.col("time_s") <= max_time)

    max_alt = df["altitude_m"].max()
    traces = [
        go.Scatter3d(
            x=df["position_x_m"].to_list(),
            y=df["position_z_m"].to_list(),
            z=df["altitude_m"].to_list(),
            mode="lines",
            name="Truth",
            line=dict(color="blue", width=4),
        )
    ]

    # ESKF estimates (if available)
    if "eskf_pos_n" in df.columns:
        traces.append(
            go.Scatter3d(
                x=df["eskf_pos_n"].to_list(),
                y=df["eskf_pos_e"].to_list(),
                z=(-df["eskf_pos_d"]).to_list(),
                mode="lines",
                name="ESKF",
                line=dict(color="red", width=3, dash="dash"),
            )
        )

    # Quantized telemetry (if available)
    if "q_flight_pos_n_m" in df.columns:
        traces.append(
            go.Scatter3d(
                x=df["q_flight_pos_n_m"].to_list(),
                y=df["q_flight_pos_e_m"].to_list(),
                z=df["q_flight_alt_m"].to_list(),
                mode="lines",
                name="Quantized",
                line=dict(color="green", width=2, dash="dot"),
            )
        )

    # State transition markers (truth)
    _state_marker_cfg = [
        ("truth_burn_time", "Burn (truth)", "red", "diamond"),
        ("truth_coasting_time", "Coasting (truth)", "cyan", "diamond"),
        ("truth_recovery_time", "Recovery (truth)", "purple", "diamond"),
    ]
    for col, label, color, symbol in _state_marker_cfg:
        if col in df.columns:
            t_val = float(df[col][0])
            if not _math.isnan(t_val):
                idx = (df["time_s"] - t_val).abs().arg_min()
                if idx is not None:
                    traces.append(
                        go.Scatter3d(
                            x=[df["position_x_m"][idx]],
                            y=[df["position_z_m"][idx]],
                            z=[df["altitude_m"][idx]],
                            mode="markers+text",
                            name=label,
                            marker=dict(size=7, color=color, symbol=symbol),
                            text=[label.split(" (")[0]],
                            textposition="top center",
                            showlegend=True,
                            hovertemplate=f"{label}<br>t={t_val:.3f}s<extra></extra>",
                        )
                    )

    # State transition markers (ESKF)
    _eskf_marker_cfg = [
        ("eskf_burn_time", "Burn (ESKF)", "darkred", "square"),
        ("eskf_coasting_time", "Coasting (ESKF)", "darkcyan", "square"),
        ("eskf_recovery_time", "Recovery (ESKF)", "indigo", "square"),
    ]
    for col, label, color, symbol in _eskf_marker_cfg:
        if col in df.columns and "eskf_pos_n" in df.columns:
            t_val = float(df[col][0])
            if not _math.isnan(t_val):
                idx = (df["time_s"] - t_val).abs().arg_min()
                if idx is not None:
                    traces.append(
                        go.Scatter3d(
                            x=[df["eskf_pos_n"][idx]],
                            y=[df["eskf_pos_e"][idx]],
                            z=[float(-df["eskf_pos_d"][idx])],
                            mode="markers",
                            name=label,
                            marker=dict(size=7, color=color, symbol=symbol),
                            showlegend=True,
                            hovertemplate=f"{label}<br>t={t_val:.3f}s<extra></extra>",
                        )
                    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="North / X (m)",
            yaxis_title="East / Z (m)",
            zaxis_title="Altitude (m)",
        ),
        height=700,
        title_text=f"Flight Path: Truth vs ESKF vs Quantized (Max Alt: {max_alt:.1f} m)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def create_2d_figures(df, max_time=None, launch_delay: float = 0.0):
    """Create 2D subplot figure (altitude, velocity, accel, forces, mass).

    Note: Each trace includes its own x (time_s) array in the JSON payload.
    Plotly.js doesn't support shared data arrays across traces, so we can't
    deduplicate time_s in the JSON. However, with LTTB downsampling to ~2000
    points per trace, the overhead is now ~16KB per trace vs ~160KB unsampled.
    """
    if max_time is not None:
        df = df.filter(pl.col("time_s") <= max_time)

    # Find apogee (maximum altitude) for vertical line markers
    max_alt_idx = df["altitude_m"].arg_max()
    apogee_time = df["time_s"][max_alt_idx] if max_alt_idx is not None else 0
    apogee_alt = df["altitude_m"].max()

    # Find ignition time (= launch_delay, when thrust first appears)
    ignition_time = launch_delay if launch_delay > 0 else None

    # Find thrust cutoff time (when thrust becomes zero after firing)
    thrust_cutoff_idx = None
    for i, thrust in enumerate(df["thrust_N"]):
        if i > 0 and df["thrust_N"][i - 1] > 0 and thrust == 0:
            thrust_cutoff_idx = i
            break
    thrust_cutoff_time = df["time_s"][thrust_cutoff_idx] if thrust_cutoff_idx is not None else None

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Altitude vs Time",
            "Velocity vs Time",
            "Acceleration vs Time",
            "Forces vs Time",
            "Mass vs Time",
            "",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # Altitude
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["altitude_m"],
            mode="lines",
            name="Altitude",
            line=dict(color="green", width=2),
        ),
        row=1,
        col=1,
    )

    # rip
    # if df["time_s"].max() > 0:
    #     fig.add_hline(y=30000, line_dash="dash", line_color="red", annotation_text="30km Target")

    # Velocity
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["velocity_total_ms"],
            mode="lines",
            name="Total Velocity",
            line=dict(color="red", width=2),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["velocity_y_ms"],
            mode="lines",
            name="Vertical Velocity",
            line=dict(color="orange", width=1, dash="dash"),
        ),
        row=1,
        col=2,
    )

    # Add apogee line to velocity plot
    fig.add_shape(
        type="line",
        x0=apogee_time,
        x1=apogee_time,
        y0=0,
        y1=1,
        xref="x2",
        yref="y2 domain",
        line=dict(color="green", width=2, dash="dot"),
        row=1,
        col=2,
    )
    fig.add_annotation(
        x=apogee_time,
        y=0.85,
        xref="x2",
        yref="y2 domain",
        text=f"Apogee: {apogee_time:.1f}s",
        showarrow=False,
        yshift=0,
        font=dict(color="green", size=9),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="green",
        borderwidth=1,
        row=1,
        col=2,
    )

    # Add ignition line to velocity plot
    if ignition_time is not None:
        fig.add_shape(
            type="line",
            x0=ignition_time,
            x1=ignition_time,
            y0=0,
            y1=1,
            xref="x2",
            yref="y2 domain",
            line=dict(color="orange", width=2, dash="dashdot"),
            row=1,
            col=2,
        )
        fig.add_annotation(
            x=ignition_time,
            y=0.55,
            xref="x2",
            yref="y2 domain",
            text=f"Ignition: {ignition_time:.1f}s",
            showarrow=False,
            yshift=0,
            font=dict(color="orange", size=9),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="orange",
            borderwidth=1,
            row=1,
            col=2,
        )

    # Acceleration
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["acceleration_total_ms2"],
            mode="lines",
            name="Total Acceleration",
            line=dict(color="purple", width=2),
        ),
        row=2,
        col=1,
    )

    # Add vertical acceleration component to understand the spike
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["acceleration_y_ms2"],
            mode="lines",
            name="Vertical Acceleration",
            line=dict(color="magenta", width=1, dash="dash"),
        ),
        row=2,
        col=1,
    )

    # Add apogee line to acceleration plot
    fig.add_shape(
        type="line",
        x0=apogee_time,
        x1=apogee_time,
        y0=0,
        y1=1,
        xref="x3",
        yref="y3 domain",
        line=dict(color="green", width=2, dash="dot"),
        row=2,
        col=1,
    )
    fig.add_annotation(
        x=apogee_time,
        y=0.85,
        xref="x3",
        yref="y3 domain",
        text=f"Apogee: {apogee_alt:.0f}m",
        showarrow=False,
        yshift=0,
        font=dict(color="green", size=9),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="green",
        borderwidth=1,
        row=2,
        col=1,
    )

    # Add ignition line to acceleration plot
    if ignition_time is not None:
        fig.add_shape(
            type="line",
            x0=ignition_time,
            x1=ignition_time,
            y0=0,
            y1=1,
            xref="x3",
            yref="y3 domain",
            line=dict(color="orange", width=2, dash="dashdot"),
            row=2,
            col=1,
        )
        fig.add_annotation(
            x=ignition_time,
            y=0.55,
            xref="x3",
            yref="y3 domain",
            text=f"Ignition: {ignition_time:.1f}s",
            showarrow=False,
            yshift=0,
            font=dict(color="orange", size=9),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="orange",
            borderwidth=1,
            row=2,
            col=1,
        )

    # Add thrust cutoff line to acceleration plot
    if thrust_cutoff_time is not None:
        fig.add_shape(
            type="line",
            x0=thrust_cutoff_time,
            x1=thrust_cutoff_time,
            y0=0,
            y1=1,
            xref="x3",
            yref="y3 domain",
            line=dict(color="red", width=2, dash="dash"),
            row=2,
            col=1,
        )
        fig.add_annotation(
            x=thrust_cutoff_time,
            y=0.15,
            xref="x3",
            yref="y3 domain",
            text=f"Engine Cutoff: {thrust_cutoff_time:.1f}s",
            showarrow=False,
            yshift=0,
            font=dict(color="red", size=9),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
            row=2,
            col=1,
        )

    # Forces
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["thrust_N"],
            mode="lines",
            name="Thrust",
            line=dict(color="red", width=2),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["drag_force_N"],
            mode="lines",
            name="Drag",
            line=dict(color="blue", width=2),
        ),
        row=2,
        col=2,
    )

    # Add thrust cutoff line to forces plot
    if thrust_cutoff_time is not None:
        fig.add_shape(
            type="line",
            x0=thrust_cutoff_time,
            x1=thrust_cutoff_time,
            y0=0,
            y1=1,
            xref="x4",
            yref="y4 domain",
            line=dict(color="red", width=2, dash="dash"),
            row=2,
            col=2,
        )
        fig.add_annotation(
            x=thrust_cutoff_time,
            y=0.85,
            xref="x4",
            yref="y4 domain",
            text="Engine Cutoff",
            showarrow=False,
            yshift=0,
            font=dict(color="red", size=9),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red",
            borderwidth=1,
            row=2,
            col=2,
        )

    # Mass
    fig.add_trace(
        go.Scatter(
            x=df["time_s"],
            y=df["mass_kg"],
            mode="lines",
            name="Mass",
            line=dict(color="brown", width=2),
        ),
        row=3,
        col=1,
    )

    # Axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Altitude (m)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Force (N)", row=2, col=2)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Mass (kg)", row=3, col=1)

    fig.update_layout(
        height=800,
        showlegend=True,
        title_font_size=16,
    )
    return fig


def create_sensor_figures(df, sensor_cfg: SensorConfig, max_time=None):
    """Create sensor data subplot figure showing readings from enabled sensors."""
    if max_time is not None:
        df = df.filter(pl.col("time_s") <= max_time)

    # Count enabled sensor groups to size the grid
    panels: list[tuple[str, list[tuple[str, str, str]]]] = []
    # Each panel: (title, [(col_name, trace_name, color), ...])

    if sensor_cfg.enabled.get("bmi088_accel") and "bmi088_accel_x_ms2" in df.columns:
        panels.append(
            (
                "BMI088 Low-G Accel (m/s²)",
                [
                    ("bmi088_accel_x_ms2", "X", "red"),
                    ("bmi088_accel_y_ms2", "Y", "green"),
                    ("bmi088_accel_z_ms2", "Z", "blue"),
                ],
            )
        )

    if sensor_cfg.enabled.get("bmi088_gyro") and "bmi088_gyro_x_dps" in df.columns:
        panels.append(
            (
                "BMI088 Gyro (°/s)",
                [
                    ("bmi088_gyro_x_dps", "X (pitch)", "red"),
                    ("bmi088_gyro_y_dps", "Y (yaw)", "green"),
                    ("bmi088_gyro_z_dps", "Z (roll)", "blue"),
                ],
            )
        )

    if sensor_cfg.enabled.get("adxl375") and "adxl375_accel_x_ms2" in df.columns:
        panels.append(
            (
                "ADXL375 High-G Accel (m/s²)",
                [
                    ("adxl375_accel_x_ms2", "X", "red"),
                    ("adxl375_accel_y_ms2", "Y", "green"),
                    ("adxl375_accel_z_ms2", "Z", "blue"),
                ],
            )
        )

    if sensor_cfg.enabled.get("ms5611") and "ms5611_pressure_mbar" in df.columns:
        panels.append(
            (
                "MS5611 Pressure (mbar)",
                [
                    ("ms5611_pressure_mbar", "Pressure", "purple"),
                ],
            )
        )

    if sensor_cfg.enabled.get("lis3mdl") and "lis3mdl_mag_x_gauss" in df.columns:
        panels.append(
            (
                "LIS3MDL Magnetometer (gauss)",
                [
                    ("lis3mdl_mag_x_gauss", "X", "red"),
                    ("lis3mdl_mag_y_gauss", "Y", "green"),
                    ("lis3mdl_mag_z_gauss", "Z", "blue"),
                ],
            )
        )

    if sensor_cfg.enabled.get("lc29h") and "gps_pos_x_m" in df.columns:
        panels.append(
            (
                "LC29H GPS X/Z Position (m)",
                [
                    ("gps_pos_x_m", "X", "red"),
                    ("gps_pos_z_m", "Z", "blue"),
                ],
            )
        )
        panels.append(
            (
                "LC29H GPS Altitude (m)",
                [
                    ("gps_pos_y_m", "Alt", "green"),
                ],
            )
        )
        panels.append(
            (
                "LC29H GPS Velocity (m/s)",
                [
                    ("gps_vel_x_ms", "Vx", "red"),
                    ("gps_vel_y_ms", "Vy", "green"),
                    ("gps_vel_z_ms", "Vz", "blue"),
                ],
            )
        )

    if not panels:
        fig = go.Figure()
        fig.update_layout(height=200, title_text="No sensors enabled")
        return fig

    n_panels = len(panels)
    rows = (n_panels + 1) // 2
    cols = min(n_panels, 2)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[p[0] for p in panels],
        vertical_spacing=0.08,
        horizontal_spacing=0.10,
    )

    for idx, (title, traces) in enumerate(panels):
        r = idx // 2 + 1
        c = idx % 2 + 1
        for col_name, trace_name, color in traces:
            if col_name in df.columns:
                series = df[col_name]
                # Drop nulls for scatter — sensors have sparse data
                mask = series.is_not_null()
                t_vals = df.filter(mask)["time_s"]
                s_vals = df.filter(mask)[col_name]
                fig.add_trace(
                    go.Scattergl(
                        x=t_vals,
                        y=s_vals,
                        mode="markers" if len(t_vals) < 5000 else "lines",
                        name=f"{title.split()[0]} {trace_name}",
                        marker=dict(color=color, size=2),
                        line=dict(color=color, width=1),
                        showlegend=False,
                    ),
                    row=r,
                    col=c,
                )
        fig.update_xaxes(title_text="Time (s)", row=r, col=c)

    fig.update_layout(
        height=300 * rows,
        showlegend=False,
        title_text="Sensor Readings",
        title_font_size=16,
        margin=dict(t=60, b=30),
    )
    return fig


def create_filter_error_figure(df, max_time=None, launch_delay: float = 1.0):
    """Create 2D position/velocity error plots."""
    if max_time is not None:
        df = df.filter(pl.col("time_s") <= max_time)

    # ---------- 2D error time-series ----------

    fig_2d = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Position North Error",
            "Position East Error",
            "Altitude Error",
            "Velocity North Error",
            "Velocity East Error",
            "Velocity Down Error",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.12,
    )

    time = df["time_s"]
    truth_n = df["position_x_m"]
    truth_e = df["position_z_m"]
    truth_alt = df["altitude_m"]
    truth_vn = df["velocity_x_ms"]
    truth_ve = df["velocity_z_ms"]
    truth_vd = -df["velocity_y_ms"]

    if "eskf_pos_n" in df.columns:
        ekf_n = df["eskf_pos_n"]
        ekf_e = df["eskf_pos_e"]
        ekf_alt = -df["eskf_pos_d"]
        ekf_vn = df["eskf_vel_n"]
        ekf_ve = df["eskf_vel_e"]
        ekf_vd = df["eskf_vel_d"]

        err_pairs = [
            (truth_n - ekf_n, "ESKF", "red", 1, 1),
            (truth_e - ekf_e, "ESKF", "red", 1, 2),
            (truth_alt - ekf_alt, "ESKF", "red", 2, 1),
            (truth_vn - ekf_vn, "ESKF", "red", 2, 2),
            (truth_ve - ekf_ve, "ESKF", "red", 3, 1),
            (truth_vd - ekf_vd, "ESKF", "red", 3, 2),
        ]
        for err, name, color, r, c in err_pairs:
            fig_2d.add_trace(
                go.Scattergl(
                    x=time,
                    y=err,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=1),
                    showlegend=(r == 1 and c == 1),
                ),
                row=r,
                col=c,
            )

    if "q_flight_pos_n_m" in df.columns:
        q_n = df["q_flight_pos_n_m"]
        q_e = df["q_flight_pos_e_m"]
        q_alt = df["q_flight_alt_m"]
        q_vn = df["q_flight_vel_n_ms"]
        q_ve = df["q_flight_vel_e_ms"]
        q_vd = df["q_flight_vel_d_ms"]

        qerr_pairs = [
            (truth_n - q_n, "Quantized", "green", 1, 1),
            (truth_e - q_e, "Quantized", "green", 1, 2),
            (truth_alt - q_alt, "Quantized", "green", 2, 1),
            (truth_vn - q_vn, "Quantized", "green", 2, 2),
            (truth_ve - q_ve, "Quantized", "green", 3, 1),
            (truth_vd - q_vd, "Quantized", "green", 3, 2),
        ]
        for err, name, color, r, c in qerr_pairs:
            fig_2d.add_trace(
                go.Scattergl(
                    x=time,
                    y=err,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=1),
                    showlegend=(r == 1 and c == 1),
                ),
                row=r,
                col=c,
            )

    # ---------- Annotations: apogee, ignition, engine cutoff ----------
    max_alt_idx = df["altitude_m"].arg_max()
    apogee_time = df["time_s"][max_alt_idx] if max_alt_idx is not None else 0
    apogee_alt = df["altitude_m"].max()

    ignition_time = launch_delay if launch_delay > 0 else None

    thrust_cutoff_time = None
    if "thrust_N" in df.columns:
        for i, thrust in enumerate(df["thrust_N"]):
            if i > 0 and df["thrust_N"][i - 1] > 0 and thrust == 0:
                thrust_cutoff_time = df["time_s"][i]
                break

    # Map (row, col) → axis index: (1,1)→1, (1,2)→2, (2,1)→3, (2,2)→4, (3,1)→5, (3,2)→6
    for r in range(1, 4):
        for c in range(1, 3):
            ax_idx = (r - 1) * 2 + c
            xref = f"x{ax_idx}" if ax_idx > 1 else "x"
            yref = f"y{ax_idx} domain" if ax_idx > 1 else "y domain"

            # Apogee
            fig_2d.add_shape(
                type="line",
                x0=apogee_time,
                x1=apogee_time,
                y0=0,
                y1=1,
                xref=xref,
                yref=yref,
                line=dict(color="green", width=1.5, dash="dot"),
                row=r,
                col=c,
            )
            if r == 1 and c == 1:
                fig_2d.add_annotation(
                    x=apogee_time,
                    y=0.92,
                    xref=xref,
                    yref=yref,
                    text=f"Apogee: {apogee_time:.1f}s ({apogee_alt:.0f}m)",
                    showarrow=False,
                    font=dict(color="green", size=8),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="green",
                    borderwidth=1,
                    row=r,
                    col=c,
                )

            # Ignition
            if ignition_time is not None:
                fig_2d.add_shape(
                    type="line",
                    x0=ignition_time,
                    x1=ignition_time,
                    y0=0,
                    y1=1,
                    xref=xref,
                    yref=yref,
                    line=dict(color="orange", width=1.5, dash="dashdot"),
                    row=r,
                    col=c,
                )
                if r == 1 and c == 1:
                    fig_2d.add_annotation(
                        x=ignition_time,
                        y=0.72,
                        xref=xref,
                        yref=yref,
                        text=f"Ignition: {ignition_time:.1f}s",
                        showarrow=False,
                        font=dict(color="orange", size=8),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="orange",
                        borderwidth=1,
                        row=r,
                        col=c,
                    )

            # Engine cutoff
            if thrust_cutoff_time is not None:
                fig_2d.add_shape(
                    type="line",
                    x0=thrust_cutoff_time,
                    x1=thrust_cutoff_time,
                    y0=0,
                    y1=1,
                    xref=xref,
                    yref=yref,
                    line=dict(color="red", width=1.5, dash="dash"),
                    row=r,
                    col=c,
                )
                if r == 1 and c == 1:
                    fig_2d.add_annotation(
                        x=thrust_cutoff_time,
                        y=0.52,
                        xref=xref,
                        yref=yref,
                        text=f"Cutoff: {thrust_cutoff_time:.1f}s",
                        showarrow=False,
                        font=dict(color="red", size=8),
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="red",
                        borderwidth=1,
                        row=r,
                        col=c,
                    )

    # Add axis labels — only show x-axis title on bottom row to avoid overlap
    for r in range(1, 4):
        for c in range(1, 3):
            if r == 3:
                fig_2d.update_xaxes(title_text="Time (s)", row=r, col=c)
            else:
                fig_2d.update_xaxes(title_text="", row=r, col=c)
    fig_2d.update_yaxes(title_text="Error (m)", row=1, col=1)
    fig_2d.update_yaxes(title_text="Error (m)", row=1, col=2)
    fig_2d.update_yaxes(title_text="Error (m)", row=2, col=1)
    fig_2d.update_yaxes(title_text="Error (m/s)", row=2, col=2)
    fig_2d.update_yaxes(title_text="Error (m/s)", row=3, col=1)
    fig_2d.update_yaxes(title_text="Error (m/s)", row=3, col=2)

    fig_2d.update_layout(
        height=850,
        showlegend=True,
        title_text="Filter Error vs Truth",
        title_font_size=16,
        margin=dict(t=60, b=40),
    )

    return fig_2d


# Figure cache for diffing (avoids sending identical JSON)
_figure_cache: dict[str, str] = {}


def _fig_hash(fig) -> str:
    """Fast hash of figure JSON for diffing."""
    # plotly_json() is expensive, but necessary for accurate comparison
    # We could also use fig.to_plotly_json() which is slightly faster
    fig_json = fig.to_json()
    return hashlib.md5(fig_json.encode()).hexdigest()


def _update_if_changed(plot_element, fig, cache_key: str):
    """Only call update_figure if the figure data has changed."""
    fig_hash = _fig_hash(fig)
    if cache_key not in _figure_cache or _figure_cache[cache_key] != fig_hash:
        _figure_cache[cache_key] = fig_hash
        plot_element.update_figure(fig)


@ui.page("/")
def index_page():
    params = RocketParams()
    sensor_cfg = SensorConfig()
    timer_ref: dict[str, ui.timer | None] = {"timer": None}
    full_df: dict[str, Any] = {"df": None}
    debounce_timer: dict[str, ui.timer | None] = {"t": None}
    slider_refs: dict[str, ui.slider] = {}
    value_labels: dict[str, Union[ui.label, ui.number]] = {}

    # UI elements to be initialized later
    plot_3d = None
    plot_2d = None
    plot_sensor = None
    plot_filter_error = None
    filter_stats_container = None
    time_slider = None
    animate_checkbox = None
    play_button = None

    def _run_sim_sync():
        """Synchronous sim + sensor + filter — called inside a thread."""
        df = simulate_rocket(params)
        df = add_sensor_data(df, sensor_cfg)
        # Always run filter if available
        if _CAN_FILTER:
            df = run_filter_pipeline(df)
        full_df["df"] = df
        return df

    async def _run_sim():
        """Run the simulation off the event loop to avoid WebSocket timeouts."""
        return await run.io_bound(_run_sim_sync)

    async def _render(df=None):
        if df is None:
            df = full_df["df"] if full_df["df"] is not None else await _run_sim()
        t_max = None
        if animate_checkbox is not None and animate_checkbox.value and time_slider is not None:
            t_max = time_slider.value

        # Build figures in a thread so the event loop stays responsive
        def _build_figures():
            figs: dict[str, Any] = {}
            figs["3d"] = create_3d_figure(df, sensor_cfg=sensor_cfg, max_time=t_max)
            figs["2d"] = create_2d_figures(df, max_time=t_max, launch_delay=params.launch_delay)
            figs["sensor"] = create_sensor_figures(df, sensor_cfg, max_time=t_max)
            # Always include filter data if available
            if _CAN_FILTER and "eskf_pos_n" in df.columns:
                figs["filter_error"] = create_filter_error_figure(df, max_time=t_max, launch_delay=params.launch_delay)
                figs["filter_stats"] = compute_error_report(df)
            return figs

        figs = await run.io_bound(_build_figures)

        # Figure diffing: only update if data changed (reduces WebSocket traffic)
        if plot_3d is not None:
            _update_if_changed(plot_3d, figs["3d"], "3d")
        if plot_2d is not None:
            _update_if_changed(plot_2d, figs["2d"], "2d")
        if plot_sensor is not None:
            _update_if_changed(plot_sensor, figs["sensor"], "sensor")
        if "filter_error" in figs:
            if plot_filter_error is not None:
                _update_if_changed(plot_filter_error, figs["filter_error"], "filter_error")
            if filter_stats_container is not None:
                filter_stats_container.clear()
                with filter_stats_container:
                    _render_error_table(figs["filter_stats"])

    def _render_error_table(err_df):
        """Render error statistics as a NiceGUI table."""
        columns = [
            {
                "name": c,
                "label": c.replace("_", " ").title(),
                "field": c,
                "sortable": True,
            }
            for c in err_df.columns
        ]
        rows_data = []
        for row in err_df.iter_rows(named=True):
            formatted = {}
            for k, v in row.items():
                if isinstance(v, float):
                    formatted[k] = f"{v:.6g}"
                else:
                    formatted[k] = v
            rows_data.append(formatted)
        ui.table(columns=columns, rows=rows_data, title="Filter Error Statistics").classes("w-full").props("dense")

    def _debounced_update():
        """Re-run sim + render, debounced so dragging a slider doesn't flood."""
        if debounce_timer["t"] is not None:
            debounce_timer["t"].cancel()

        async def go():
            debounce_timer["t"] = None
            await _run_sim()
            await _render()
            if time_slider is not None and full_df["df"] is not None:
                try:
                    max_time = full_df["df"]["time_s"].max()
                    if max_time is not None:
                        time_slider._props["max"] = round(float(max_time), 2)  # type: ignore
                        time_slider.update()
                except (TypeError, ValueError):
                    pass  # Skip if conversion fails

        debounce_timer["t"] = ui.timer(0.15, go, once=True)

    def _debounced_sensor_update():
        """Re-generate sensor data + re-run filter (faster than full sim re-run)."""
        if debounce_timer["t"] is not None:
            debounce_timer["t"].cancel()

        async def go():
            debounce_timer["t"] = None
            # Keep the existing base simulation, just regenerate sensor overlay
            if full_df["df"] is not None:

                def _regen_sensors():
                    # Strip sensor, filter, and state columns — keep only raw sim data
                    base_df = full_df["df"].select(
                        [
                            c
                            for c in full_df["df"].columns
                            if not c.startswith(
                                (
                                    "bmi088_",
                                    "adxl375_",
                                    "ms5611_",
                                    "lis3mdl_",
                                    "gps_",
                                    "eskf_",
                                    "q_flight_",
                                    "q_recovery_",
                                    "truth_state",
                                    "eskf_state",
                                    "truth_pad_",
                                    "truth_ignition_",
                                    "truth_burn_",
                                    "truth_coasting_",
                                    "truth_apogee_",
                                    "truth_recovery_",
                                    "eskf_pad_",
                                    "eskf_ignition_",
                                    "eskf_burn_",
                                    "eskf_coasting_",
                                    "eskf_apogee_",
                                    "eskf_recovery_",
                                )
                            )
                        ]
                    )
                    df = add_sensor_data(base_df, sensor_cfg)
                    # Re-run filter so ESKF/state columns reflect new sensor config
                    if _CAN_FILTER:
                        df = run_filter_pipeline(df)
                    return df

                full_df["df"] = await run.io_bound(_regen_sensors)
            await _render()

        debounce_timer["t"] = ui.timer(0.05, go, once=True)

    def _make_param_setter(attr, target_obj=None, target_attr=None, use_sensor_update=False):
        """Return a handler that sets an attribute and triggers debounced update."""
        obj = target_obj or params
        real_attr = target_attr or attr

        def handler(e):
            val = float(e.args)
            setattr(obj, real_attr, val)
            key = f"sensor_{target_attr}" if target_obj is not None else attr
            if key in value_labels and hasattr(value_labels[key], "value"):
                value_labels[key].value = val  # type: ignore
            if use_sensor_update:
                _debounced_sensor_update()
            else:
                _debounced_update()

        return handler

    def _make_input_setter(
        attr,
        min_val,
        max_val,
        target_obj=None,
        target_attr=None,
        use_sensor_update=False,
    ):
        """Return a handler for manual input that validates bounds."""
        obj = target_obj or params
        real_attr = target_attr or attr

        def handler(e):
            try:
                val = float(e.args if hasattr(e, "args") else e.value)
                val = max(min_val, min(max_val, val))
                setattr(obj, real_attr, val)
                key = f"sensor_{target_attr}" if target_obj is not None else attr
                if key in slider_refs:
                    slider_refs[key].value = val
                if key in value_labels and hasattr(value_labels[key], "value"):
                    value_labels[key].value = val  # type: ignore
                if use_sensor_update:
                    _debounced_sensor_update()
                else:
                    _debounced_update()
            except ValueError:
                pass

        return handler

    def _labeled_slider(
        label_text,
        attr,
        target_obj=None,
        target_attr=None,
        use_sensor_update=False,
        **kwargs,
    ):
        """Label on the left, slider in middle, manual input on right."""
        min_val = kwargs.get("min", 0)
        max_val = kwargs.get("max", 100)
        current_val = kwargs.get("value", 0)

        with ui.row().classes("w-full items-center gap-2 mt-1"):
            ui.label(label_text).classes("text-xs font-semibold whitespace-nowrap min-w-[7rem]")
            s = (
                ui.slider(**kwargs)
                .classes("flex-grow")
                .on(
                    "update:model-value",
                    _make_param_setter(
                        attr,
                        target_obj=target_obj,
                        target_attr=target_attr,
                        use_sensor_update=use_sensor_update,
                    ),
                )
            )
            slider_refs[attr] = s

            input_field = (
                ui.number(
                    value=current_val,
                    format="%.3g",
                    min=min_val,
                    max=max_val,
                    step=kwargs.get("step", 0.1),
                )
                .props("dense outlined")
                .classes("w-20")
                .on(
                    "update:model-value",
                    _make_input_setter(
                        attr,
                        min_val,
                        max_val,
                        target_obj=target_obj,
                        target_attr=target_attr,
                        use_sensor_update=use_sensor_update,
                    ),
                )
            )

            value_labels[attr] = input_field

    def _apply_preset(name: str):
        preset = PRESETS[name]
        for f in fields(RocketParams):
            val = getattr(preset, f.name)
            setattr(params, f.name, val)
            if f.name in slider_refs:
                slider_refs[f.name].value = val
            if f.name in value_labels and hasattr(value_labels[f.name], "value"):
                value_labels[f.name].value = val  # type: ignore
        _debounced_update()

    render_debounce: dict[str, ui.timer | None] = {"t": None}

    def on_time_change(e):
        if render_debounce["t"] is not None:
            render_debounce["t"].cancel()

        async def _do():
            render_debounce["t"] = None
            await _render()

        render_debounce["t"] = ui.timer(0.1, _do, once=True)

    def on_animate_toggle(e):
        if render_debounce["t"] is not None:
            render_debounce["t"].cancel()

        async def _do():
            render_debounce["t"] = None
            await _render()

        render_debounce["t"] = ui.timer(0.1, _do, once=True)

    def on_play():
        if timer_ref["timer"] is not None:
            timer_ref["timer"].cancel()
            timer_ref["timer"] = None
            if play_button is not None:
                play_button.set_text("▶ Play")
            return
        if play_button is not None:
            play_button.set_text("⏸ Pause")
        if animate_checkbox is not None:
            animate_checkbox.value = True
        if time_slider is not None:
            time_slider.value = 0.0
        step = 0.5

        async def tick():
            if time_slider is None or play_button is None:
                return
            t_max = time_slider._props.get("max", 10)
            if time_slider.value >= t_max:
                if timer_ref["timer"] is not None:
                    timer_ref["timer"].cancel()
                timer_ref["timer"] = None
                play_button.set_text("▶ Play")
                return
            time_slider.value = round(time_slider.value + step, 2)
            await _render()

        timer_ref["timer"] = ui.timer(0.1, tick)

    async def export_xlsx():
        df = full_df["df"] if full_df["df"] is not None else await _run_sim()

        def _write():
            buf = io.BytesIO()
            df.write_excel(buf, worksheet="Flight Data")
            return buf.getvalue()

        data = await run.io_bound(_write)
        ui.download(data, "rocket_simulation.xlsx")

    async def export_csv():
        df = full_df["df"] if full_df["df"] is not None else await _run_sim()

        def _write():
            buf = io.BytesIO()
            df.write_csv(buf)
            return buf.getvalue()

        data = await run.io_bound(_write)
        ui.download(data, "rocket_simulation.csv")

    # ── Build UI ──────────────────────────────────────────────────────
    ui.page_title("Aloe")

    sidebar_visible = {"open": True}
    sidebar_column = None
    toggle_button = None

    def toggle_sidebar():
        sidebar_visible["open"] = not sidebar_visible["open"]
        if sidebar_column is not None:
            if sidebar_visible["open"]:
                sidebar_column.style("display: flex; width: 24rem; min-width: 22rem")
            else:
                sidebar_column.style("display: none; width: 0; min-width: 0")
        if toggle_button is not None:
            toggle_button.props(f"icon={'menu_open' if sidebar_visible['open'] else 'menu'}")

    with ui.row().classes("w-full h-screen items-stretch"):
        # ── Left column — tabbed settings sidebar ─────────────────
        with ui.column().classes("w-96 min-w-[22rem] overflow-y-auto p-4 border-r") as sidebar_column:
            ui.label("Aloe").classes("text-2xl font-bold text-center w-full mb-2")

            # Preset buttons (always visible at top)
            with ui.card().classes("w-full mb-2"):
                ui.label("Presets").classes("text-sm font-bold mb-1")
                with ui.row().classes("w-full flex-wrap gap-1"):
                    for preset_name in PRESETS:
                        ui.button(
                            preset_name,
                            on_click=lambda e, _n=preset_name: _apply_preset(_n),
                        ).props(
                            "dense outline size=sm"
                        ).classes("text-xs")

            # Tabbed panels for different setting categories
            with ui.tabs().classes("w-full") as tabs:
                rocket_tab = ui.tab("Rocket")
                env_tab = ui.tab("Env")
                sensors_tab = ui.tab("Sensors")
                playback_tab = ui.tab("Playback")

            with ui.tab_panels(tabs, value=rocket_tab).classes("w-full"):
                with ui.tab_panel(rocket_tab):
                    ui.label("Rocket Parameters").classes("text-sm font-bold mb-2")
                    for label, attr, mn, mx, step in ROCKET_SLIDERS:
                        _labeled_slider(
                            label,
                            attr,
                            min=mn,
                            max=mx,
                            step=step,
                            value=getattr(params, attr),
                        )

                with ui.tab_panel(env_tab):
                    ui.label("Environment").classes("text-sm font-bold mb-2")
                    for label, attr, mn, mx, step in ENV_SLIDERS:
                        _labeled_slider(
                            label,
                            attr,
                            min=mn,
                            max=mx,
                            step=step,
                            value=getattr(params, attr),
                        )
                    ui.label("Tip: Set Crosswind Z for lateral drift").classes("text-xs text-gray-500 mt-2 italic")

                with ui.tab_panel(sensors_tab):
                    ui.label("Sensor Configuration").classes("text-sm font-bold mb-2")

                    with ui.row().classes("w-full items-center gap-2 mb-2"):
                        ui.label("Noise Scale").classes("text-xs font-semibold min-w-[7rem]")
                        noise_slider = ui.slider(min=0, max=5, step=0.1, value=sensor_cfg.noise_scale).classes(
                            "flex-grow"
                        )
                        noise_input = (
                            ui.number(
                                value=sensor_cfg.noise_scale,
                                format="%.1f",
                                min=0,
                                max=5,
                                step=0.1,
                            )
                            .props("dense outlined")
                            .classes("w-20")
                        )

                        def _set_noise(e):
                            sensor_cfg.noise_scale = float(e.args if hasattr(e, "args") else e.value)
                            noise_input.value = sensor_cfg.noise_scale  # type: ignore
                            _debounced_sensor_update()

                        noise_slider.on("update:model-value", _set_noise)
                        noise_input.on("update:model-value", _set_noise)

                    with ui.row().classes("w-full items-center gap-2 mb-2"):
                        ui.label("RNG Seed").classes("text-xs font-semibold min-w-[7rem]")
                        seed_input = (
                            ui.number(
                                value=sensor_cfg.seed,
                                format="%d",
                                min=0,
                                max=2**31,
                                step=1,
                            )
                            .props("dense outlined")
                            .classes("w-32")
                        )

                        def _set_seed(e):
                            sensor_cfg.seed = int(e.args if hasattr(e, "args") else e.value)
                            _debounced_sensor_update()

                        seed_input.on("update:model-value", _set_seed)

                    ui.label("Enabled Sensors").classes("text-xs font-semibold mt-2 mb-1")
                    sensor_toggles: dict[str, ui.checkbox] = {}
                    sensor_labels = {
                        "bmi088_accel": "BMI088 Accel (±24g)",
                        "bmi088_gyro": "BMI088 Gyro (±2000°/s)",
                        "adxl375": "ADXL375 High-G (±200g)",
                        "ms5611": "MS5611 Baro",
                        "lis3mdl": "LIS3MDL Mag",
                        "lc29h": "LC29H GPS",
                    }
                    for key, label in sensor_labels.items():

                        def _toggle_sensor(e, _k=key):
                            sensor_cfg.enabled[_k] = e.value
                            _debounced_sensor_update()

                        sensor_toggles[key] = ui.checkbox(
                            label,
                            value=sensor_cfg.enabled.get(key, True),
                            on_change=_toggle_sensor,
                        ).classes("text-xs")

                    ui.separator().classes("my-3")
                    ui.label("Sample Rates (Hz)").classes("text-xs font-semibold mb-1")
                    for label, attr, mn, mx, step in SENSOR_RATE_SLIDERS:
                        _labeled_slider(
                            label,
                            f"sensor_{attr}",
                            min=mn,
                            max=mx,
                            step=step,
                            value=getattr(sensor_cfg, attr),
                            target_obj=sensor_cfg,
                            target_attr=attr,
                            use_sensor_update=True,
                        )

                    ui.separator().classes("my-3")
                    ui.label("Latency (ms)").classes("text-xs font-semibold mb-1")
                    for label, attr, mn, mx, step in SENSOR_LATENCY_SLIDERS:
                        _labeled_slider(
                            label,
                            f"sensor_{attr}",
                            min=mn,
                            max=mx,
                            step=step,
                            value=getattr(sensor_cfg, attr),
                            target_obj=sensor_cfg,
                            target_attr=attr,
                            use_sensor_update=True,
                        )
                    ui.label("Note: Noise is independent per axis").classes("text-xs text-gray-500 mt-2 italic")

                with ui.tab_panel(playback_tab):
                    ui.label("Playback Controls").classes("text-sm font-bold mb-2")
                    animate_checkbox = ui.checkbox("Animate over time", value=False, on_change=on_animate_toggle)
                    time_slider = ui.slider(min=0, max=10, step=0.1, value=10).props("label")
                    time_slider.on("update:model-value", on_time_change)
                    play_button = ui.button("▶ Play", on_click=on_play).props("color=secondary").classes("w-full mt-2")

                    ui.separator().classes("my-3")
                    ui.label("Export Data").classes("text-sm font-bold mb-2")
                    with ui.row().classes("w-full gap-2"):
                        ui.button("Excel", on_click=export_xlsx).props("color=primary dense").classes("flex-grow")
                        ui.button("CSV", on_click=export_csv).props("color=primary outline dense").classes("flex-grow")

        # ── Right column — charts fill remaining space ────────────
        with ui.column().classes("flex-grow overflow-y-auto p-2 relative"):
            # Floating sidebar toggle button
            toggle_button = (
                ui.button(
                    icon="menu_open",
                    on_click=toggle_sidebar,
                )
                .props("fab")
                .classes("absolute top-4 left-4 z-50")
                .style("box-shadow: 0 2px 8px rgba(0,0,0,0.3)")
            )

            df = _run_sim_sync()
            plot_3d = ui.plotly(create_3d_figure(df, sensor_cfg=sensor_cfg)).classes("w-full")
            plot_2d = ui.plotly(create_2d_figures(df, launch_delay=params.launch_delay)).classes("w-full")
            plot_sensor = ui.plotly(create_sensor_figures(df, sensor_cfg)).classes("w-full")

            # Filter error plots and stats (always shown if filter ran successfully)
            if _CAN_FILTER and "eskf_pos_n" in df.columns:
                ui.separator().classes("my-4")
                ui.label("Kalman Filter Performance").classes("text-lg font-bold mb-2")
                ui.label(
                    "ESKF (Error-State Extended Kalman Filter) fuses sensor data. "
                    "Quantized shows telemetry after radio transmission encoding."
                ).classes("text-xs text-gray-600 mb-4")

                plot_filter_error = ui.plotly(create_filter_error_figure(df, launch_delay=params.launch_delay)).classes(
                    "w-full"
                )

                filter_stats_container = ui.column().classes("w-full mt-4")
                with filter_stats_container:
                    err_df = compute_error_report(df)
                    _render_error_table(err_df)
            else:
                plot_filter_error = None
                filter_stats_container = None
                if not _CAN_FILTER:
                    ui.label("⚠ Kalman filter unavailable. Build native extension: maturin develop --release").classes(
                        "text-sm text-orange-600 mt-4"
                    )

            if time_slider is not None:
                try:
                    max_time = df["time_s"].max()
                    if max_time is not None:
                        time_slider._props["max"] = round(float(max_time), 2)  # type: ignore
                        time_slider._props["value"] = time_slider._props["max"]
                        time_slider.update()
                except (TypeError, ValueError):
                    pass  # Skip if conversion fails
