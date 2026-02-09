import io
from dataclasses import fields
from typing import Any, Union
from nicegui import ui
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

from aloe.sim import RocketParams, SensorConfig, PRESETS, simulate_rocket, add_sensor_data
from aloe.params import get_rocket_sliders, get_env_sliders, get_sensor_rate_sliders, get_sensor_latency_sliders

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
    """Create standalone 3D flight path figure with optional sensor overlays."""
    if max_time is not None:
        df = df.filter(pl.col("time_s") <= max_time)

    max_alt = df["altitude_m"].max()
    traces = [
        go.Scatter3d(
            x=df["position_x_m"].to_list(),
            y=df["position_z_m"].to_list(),
            z=df["altitude_m"].to_list(),
            mode="lines",
            name="Flight Path",
            line=dict(color="blue", width=4),
        )
    ]

    # Overlay 3-axis sensor data as scatter points
    if sensor_cfg is not None:
        # GPS position (same coordinate space as flight path)
        if sensor_cfg.enabled.get("lc29h") and "gps_pos_x_m" in df.columns:
            mask = df["gps_pos_x_m"].is_not_null()
            gps_df = df.filter(mask)
            traces.append(
                go.Scatter3d(
                    x=gps_df["gps_pos_x_m"].to_list(),
                    y=gps_df["gps_pos_z_m"].to_list(),
                    z=gps_df["gps_pos_y_m"].to_list(),
                    mode="markers",
                    name="GPS (LC29H)",
                    marker=dict(color="red", size=3, symbol="circle", opacity=0.7),
                )
            )

        # BMI088 low-g accel — plot as 3D scatter of accel vector
        if sensor_cfg.enabled.get("bmi088_accel") and "bmi088_accel_x_ms2" in df.columns:
            mask = df["bmi088_accel_x_ms2"].is_not_null()
            s_df = df.filter(mask)
            traces.append(
                go.Scatter3d(
                    x=s_df["bmi088_accel_x_ms2"].to_list(),
                    y=s_df["bmi088_accel_z_ms2"].to_list(),
                    z=s_df["bmi088_accel_y_ms2"].to_list(),
                    mode="markers",
                    name="BMI088 Accel",
                    marker=dict(color="orange", size=2, symbol="diamond", opacity=0.5),
                    visible="legendonly",
                )
            )

        # ADXL375 high-g accel
        if sensor_cfg.enabled.get("adxl375") and "adxl375_accel_x_ms2" in df.columns:
            mask = df["adxl375_accel_x_ms2"].is_not_null()
            s_df = df.filter(mask)
            traces.append(
                go.Scatter3d(
                    x=s_df["adxl375_accel_x_ms2"].to_list(),
                    y=s_df["adxl375_accel_z_ms2"].to_list(),
                    z=s_df["adxl375_accel_y_ms2"].to_list(),
                    mode="markers",
                    name="ADXL375 Accel",
                    marker=dict(color="green", size=2, symbol="square", opacity=0.5),
                    visible="legendonly",
                )
            )

        # BMI088 gyro
        if sensor_cfg.enabled.get("bmi088_gyro") and "bmi088_gyro_x_dps" in df.columns:
            mask = df["bmi088_gyro_x_dps"].is_not_null()
            s_df = df.filter(mask)
            traces.append(
                go.Scatter3d(
                    x=s_df["bmi088_gyro_x_dps"].to_list(),
                    y=s_df["bmi088_gyro_z_dps"].to_list(),
                    z=s_df["bmi088_gyro_y_dps"].to_list(),
                    mode="markers",
                    name="BMI088 Gyro",
                    marker=dict(color="purple", size=2, symbol="cross", opacity=0.5),
                    visible="legendonly",
                )
            )

        # LIS3MDL magnetometer
        if sensor_cfg.enabled.get("lis3mdl") and "lis3mdl_mag_x_gauss" in df.columns:
            mask = df["lis3mdl_mag_x_gauss"].is_not_null()
            s_df = df.filter(mask)
            traces.append(
                go.Scatter3d(
                    x=s_df["lis3mdl_mag_x_gauss"].to_list(),
                    y=s_df["lis3mdl_mag_z_gauss"].to_list(),
                    z=s_df["lis3mdl_mag_y_gauss"].to_list(),
                    mode="markers",
                    name="LIS3MDL Mag",
                    marker=dict(color="cyan", size=2, symbol="x", opacity=0.5),
                    visible="legendonly",
                )
            )

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="Downrange X (m)",
            yaxis_title="Crosswind Z (m)",
            zaxis_title="Altitude (m)",
        ),
        height=700,
        title_text=f"3D Flight Path (Max Alt: {max_alt:.1f} m)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def create_2d_figures(df, max_time=None, launch_delay: float = 0.0):
    """Create 2D subplot figure (altitude, velocity, accel, forces, mass)."""
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
            x=df["time_s"], y=df["altitude_m"], mode="lines", name="Altitude", line=dict(color="green", width=2)
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
        go.Scatter(x=df["time_s"], y=df["thrust_N"], mode="lines", name="Thrust", line=dict(color="red", width=2)),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["drag_force_N"], mode="lines", name="Drag", line=dict(color="blue", width=2)),
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
        go.Scatter(x=df["time_s"], y=df["mass_kg"], mode="lines", name="Mass", line=dict(color="brown", width=2)),
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


def create_filter_figures(df, max_time=None):
    """Create filter output plots: 3D path comparison + position/velocity error over time."""
    if max_time is not None:
        df = df.filter(pl.col("time_s") <= max_time)

    # ---------- 3D path comparison: truth vs ESKF vs quantized ----------
    traces_3d = [
        go.Scatter3d(
            x=df["position_x_m"].to_list(),
            y=df["position_z_m"].to_list(),
            z=df["altitude_m"].to_list(),
            mode="lines",
            name="Truth",
            line=dict(color="blue", width=4),
        ),
    ]
    if "eskf_pos_n" in df.columns:
        traces_3d.append(
            go.Scatter3d(
                x=df["eskf_pos_n"].to_list(),
                y=df["eskf_pos_e"].to_list(),
                z=(-df["eskf_pos_d"]).to_list(),
                mode="lines",
                name="ESKF",
                line=dict(color="red", width=3, dash="dash"),
            )
        )
    if "q_flight_pos_n_m" in df.columns:
        traces_3d.append(
            go.Scatter3d(
                x=df["q_flight_pos_n_m"].to_list(),
                y=df["q_flight_pos_e_m"].to_list(),
                z=df["q_flight_alt_m"].to_list(),
                mode="lines",
                name="Quantized",
                line=dict(color="green", width=2, dash="dot"),
            )
        )

    fig_3d = go.Figure(data=traces_3d)
    max_alt = df["altitude_m"].max()
    fig_3d.update_layout(
        scene=dict(
            xaxis_title="North / X (m)",
            yaxis_title="East / Z (m)",
            zaxis_title="Altitude (m)",
        ),
        height=650,
        title_text=f"Flight Path: Truth vs ESKF vs Quantized (Max Alt: {max_alt:.1f} m)",
        margin=dict(l=0, r=0, t=40, b=0),
    )

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
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
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

    # Add zero lines and axis labels
    for r in range(1, 4):
        for c in range(1, 3):
            fig_2d.update_xaxes(title_text="Time (s)", row=r, col=c)
    fig_2d.update_yaxes(title_text="Error (m)", row=1, col=1)
    fig_2d.update_yaxes(title_text="Error (m)", row=1, col=2)
    fig_2d.update_yaxes(title_text="Error (m)", row=2, col=1)
    fig_2d.update_yaxes(title_text="Error (m/s)", row=2, col=2)
    fig_2d.update_yaxes(title_text="Error (m/s)", row=3, col=1)
    fig_2d.update_yaxes(title_text="Error (m/s)", row=3, col=2)

    fig_2d.update_layout(
        height=750,
        showlegend=True,
        title_text="Filter Error vs Truth",
        title_font_size=16,
    )

    return fig_3d, fig_2d


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
    plot_filter_3d = None
    plot_filter_2d = None
    filter_stats_container = None
    time_slider = None
    animate_checkbox = None
    play_button = None
    filter_state: dict[str, Any] = {"enabled": False}

    def _run_sim():
        df = simulate_rocket(params)
        df = add_sensor_data(df, sensor_cfg)
        if filter_state["enabled"] and _CAN_FILTER:
            df = run_filter_pipeline(df)
        full_df["df"] = df
        return df

    def _render(df=None):
        if df is None:
            df = full_df["df"] if full_df["df"] is not None else _run_sim()
        t_max = None
        if animate_checkbox is not None and animate_checkbox.value and time_slider is not None:
            t_max = time_slider.value
        if plot_3d is not None:
            plot_3d.update_figure(create_3d_figure(df, sensor_cfg=sensor_cfg, max_time=t_max))
        if plot_2d is not None:
            plot_2d.update_figure(create_2d_figures(df, max_time=t_max, launch_delay=params.launch_delay))
        if plot_sensor is not None:
            plot_sensor.update_figure(create_sensor_figures(df, sensor_cfg, max_time=t_max))
        if filter_state["enabled"] and _CAN_FILTER and "eskf_pos_n" in df.columns:
            f3, f2 = create_filter_figures(df, max_time=t_max)
            if plot_filter_3d is not None:
                plot_filter_3d.update_figure(f3)
            if plot_filter_2d is not None:
                plot_filter_2d.update_figure(f2)
            if filter_stats_container is not None:
                filter_stats_container.clear()
                err_df = compute_error_report(df)
                with filter_stats_container:
                    _render_error_table(err_df)

    def _render_error_table(err_df):
        """Render error statistics as a NiceGUI table."""
        columns = [
            {"name": c, "label": c.replace("_", " ").title(), "field": c, "sortable": True} for c in err_df.columns
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

        def go():
            debounce_timer["t"] = None
            _run_sim()
            _render()
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
        """Re-generate sensor data only (faster than full sim re-run)."""
        if debounce_timer["t"] is not None:
            debounce_timer["t"].cancel()

        def go():
            debounce_timer["t"] = None
            # Keep the existing base simulation, just regenerate sensor overlay
            if full_df["df"] is not None:
                base_df = full_df["df"].select(
                    [
                        c
                        for c in full_df["df"].columns
                        if not c.startswith(("bmi088_", "adxl375_", "ms5611_", "lis3mdl_", "gps_"))
                    ]
                )
                full_df["df"] = add_sensor_data(base_df, sensor_cfg)
            _render()

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

    def _make_input_setter(attr, min_val, max_val, target_obj=None, target_attr=None, use_sensor_update=False):
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

    def _labeled_slider(label_text, attr, target_obj=None, target_attr=None, use_sensor_update=False, **kwargs):
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
                        attr, target_obj=target_obj, target_attr=target_attr, use_sensor_update=use_sensor_update
                    ),
                )
            )
            slider_refs[attr] = s

            input_field = (
                ui.number(value=current_val, format="%.3g", min=min_val, max=max_val, step=kwargs.get("step", 0.1))
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

    def on_time_change(e):
        _render()

    def on_animate_toggle(e):
        _render()

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
            _render()

        timer_ref["timer"] = ui.timer(0.1, tick)

    def export_xlsx():
        df = full_df["df"] if full_df["df"] is not None else _run_sim()
        buf = io.BytesIO()
        df.write_excel(buf, worksheet="Flight Data")
        ui.download(buf.getvalue(), "rocket_simulation.xlsx")

    def export_csv():
        df = full_df["df"] if full_df["df"] is not None else _run_sim()
        buf = io.BytesIO()
        df.write_csv(buf)
        ui.download(buf.getvalue(), "rocket_simulation.csv")

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
                filter_tab = ui.tab("Filter")
                playback_tab = ui.tab("Playback")

            with ui.tab_panels(tabs, value=rocket_tab).classes("w-full"):
                with ui.tab_panel(rocket_tab):
                    ui.label("Rocket Parameters").classes("text-sm font-bold mb-2")
                    for label, attr, mn, mx, step in ROCKET_SLIDERS:
                        _labeled_slider(label, attr, min=mn, max=mx, step=step, value=getattr(params, attr))

                with ui.tab_panel(env_tab):
                    ui.label("Environment").classes("text-sm font-bold mb-2")
                    for label, attr, mn, mx, step in ENV_SLIDERS:
                        _labeled_slider(label, attr, min=mn, max=mx, step=step, value=getattr(params, attr))
                    ui.label("Tip: Set Crosswind Z for lateral drift").classes("text-xs text-gray-500 mt-2 italic")

                with ui.tab_panel(sensors_tab):
                    ui.label("Sensor Configuration").classes("text-sm font-bold mb-2")

                    with ui.row().classes("w-full items-center gap-2 mb-2"):
                        ui.label("Noise Scale").classes("text-xs font-semibold min-w-[7rem]")
                        noise_slider = ui.slider(min=0, max=5, step=0.1, value=sensor_cfg.noise_scale).classes(
                            "flex-grow"
                        )
                        noise_input = (
                            ui.number(value=sensor_cfg.noise_scale, format="%.1f", min=0, max=5, step=0.1)
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
                            ui.number(value=sensor_cfg.seed, format="%d", min=0, max=2**31, step=1)
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

                with ui.tab_panel(filter_tab):
                    ui.label("ES-EKF Sensor Fusion").classes("text-sm font-bold mb-2")
                    if _CAN_FILTER:

                        def _toggle_filter(e):
                            filter_state["enabled"] = e.value
                            _debounced_update()

                        ui.checkbox(
                            "Enable Kalman Filter",
                            value=False,
                            on_change=_toggle_filter,
                        ).classes("text-sm")
                        ui.label(
                            "Runs an Error-State EKF on sensor data to produce "
                            "fused position/velocity estimates, then quantizes to "
                            "the on-wire telemetry format."
                        ).classes("text-xs text-gray-500 mt-2")
                    else:
                        ui.label("⚠ Native extension not available. " "Build with: maturin develop --release").classes(
                            "text-xs text-red-500"
                        )

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

            df = _run_sim()
            plot_3d = ui.plotly(create_3d_figure(df, sensor_cfg=sensor_cfg)).classes("w-full")
            plot_2d = ui.plotly(create_2d_figures(df, launch_delay=params.launch_delay)).classes("w-full")
            plot_sensor = ui.plotly(create_sensor_figures(df, sensor_cfg)).classes("w-full")

            # Filter plots (initially empty, populated when filter is enabled)
            with ui.expansion("Kalman Filter Output", icon="filter_alt").classes("w-full").props("default-closed"):
                if _CAN_FILTER and "eskf_pos_n" in df.columns:
                    f3d, f2d = create_filter_figures(df)
                    plot_filter_3d = ui.plotly(f3d).classes("w-full")
                    plot_filter_2d = ui.plotly(f2d).classes("w-full")
                    filter_stats_container = ui.column().classes("w-full")
                    with filter_stats_container:
                        err_df = compute_error_report(df)
                        _render_error_table(err_df)
                else:
                    empty_fig = go.Figure()
                    empty_fig.update_layout(height=200, title_text="Enable filter in the Filter tab")
                    plot_filter_3d = ui.plotly(empty_fig).classes("w-full")
                    plot_filter_2d = ui.plotly(empty_fig).classes("w-full")
                    filter_stats_container = ui.column().classes("w-full")

            if time_slider is not None:
                try:
                    max_time = df["time_s"].max()
                    if max_time is not None:
                        time_slider._props["max"] = round(float(max_time), 2)  # type: ignore
                        time_slider._props["value"] = time_slider._props["max"]
                        time_slider.update()
                except (TypeError, ValueError):
                    pass  # Skip if conversion fails
