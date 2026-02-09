import io
from dataclasses import fields
from typing import Any, Union
from nicegui import ui
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

from aloe.sim import RocketParams, PRESETS, simulate_rocket

# Slider config: (label, attr, min, max, step)
ROCKET_SLIDERS = [
    ("Dry Mass (kg)", "dry_mass", 5, 100, 1),
    ("Propellant (kg)", "propellant_mass", 10, 300, 5),
    ("Thrust (N)", "thrust", 500, 25000, 250),
    ("Burn Time (s)", "burn_time", 2, 45, 0.5),
    ("Drag Coeff.", "drag_coeff", 0.15, 1.5, 0.05),
    ("Ref. Area (m²)", "ref_area", 0.005, 0.08, 0.002),
]
ENV_SLIDERS = [
    ("Gravity (m/s²)", "gravity", 1.0, 15, 0.1),
    ("Wind X (m/s)", "wind_speed", 0, 25, 1),
    ("Crosswind Z (m/s)", "wind_speed_z", -25, 25, 1),
    ("Air Density (kg/m³)", "air_density", 0.3, 1.8, 0.02),
]


def create_3d_figure(df, max_time=None):
    """Create standalone 3D flight path figure."""
    if max_time is not None:
        df = df.filter(pl.col("time_s") <= max_time)

    max_alt = df["altitude_m"].max()
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df["position_x_m"].to_list(),
                y=df["position_z_m"].to_list(),
                z=df["altitude_m"].to_list(),
                mode="lines",
                name="Flight Path",
                line=dict(color="blue", width=4),
            )
        ],
    )
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


def create_2d_figures(df, max_time=None):
    """Create 2D subplot figure (altitude, velocity, accel, forces, mass)."""
    if max_time is not None:
        df = df.filter(pl.col("time_s") <= max_time)

    # Find apogee (maximum altitude) for vertical line markers
    max_alt_idx = df["altitude_m"].arg_max()
    apogee_time = df["time_s"][max_alt_idx] if max_alt_idx is not None else 0
    apogee_alt = df["altitude_m"].max()
    
    # Find thrust cutoff time (when thrust becomes zero)
    thrust_cutoff_idx = None
    for i, thrust in enumerate(df["thrust_N"]):
        if i > 0 and df["thrust_N"][i-1] > 0 and thrust == 0:
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
        x0=apogee_time, x1=apogee_time, y0=0, y1=1,
        xref="x2", yref="y2 domain",
        line=dict(color="green", width=2, dash="dot"),
        row=1, col=2
    )
    fig.add_annotation(
        x=apogee_time, y=0.85, xref="x2", yref="y2 domain",
        text=f"Apogee: {apogee_time:.1f}s", showarrow=False,
        yshift=0, font=dict(color="green", size=9),
        bgcolor="rgba(255,255,255,0.8)", bordercolor="green", borderwidth=1,
        row=1, col=2
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
        x0=apogee_time, x1=apogee_time, y0=0, y1=1,
        xref="x3", yref="y3 domain",
        line=dict(color="green", width=2, dash="dot"),
        row=2, col=1
    )
    fig.add_annotation(
        x=apogee_time, y=0.85, xref="x3", yref="y3 domain",
        text=f"Apogee: {apogee_alt:.0f}m", showarrow=False,
        yshift=0, font=dict(color="green", size=9),
        bgcolor="rgba(255,255,255,0.8)", bordercolor="green", borderwidth=1,
        row=2, col=1
    )
                  
    # Add thrust cutoff line to acceleration plot
    if thrust_cutoff_time is not None:
        fig.add_shape(
            type="line",
            x0=thrust_cutoff_time, x1=thrust_cutoff_time, y0=0, y1=1,
            xref="x3", yref="y3 domain",
            line=dict(color="red", width=2, dash="dash"),
            row=2, col=1
        )
        fig.add_annotation(
            x=thrust_cutoff_time, y=0.15, xref="x3", yref="y3 domain",
            text=f"Engine Cutoff: {thrust_cutoff_time:.1f}s", showarrow=False,
            yshift=0, font=dict(color="red", size=9),
            bgcolor="rgba(255,255,255,0.8)", bordercolor="red", borderwidth=1,
            row=2, col=1
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
            x0=thrust_cutoff_time, x1=thrust_cutoff_time, y0=0, y1=1,
            xref="x4", yref="y4 domain",
            line=dict(color="red", width=2, dash="dash"),
            row=2, col=2
        )
        fig.add_annotation(
            x=thrust_cutoff_time, y=0.85, xref="x4", yref="y4 domain",
            text="Engine Cutoff", showarrow=False,
            yshift=0, font=dict(color="red", size=9),
            bgcolor="rgba(255,255,255,0.8)", bordercolor="red", borderwidth=1,
            row=2, col=2
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


@ui.page("/")
def index_page():
    params = RocketParams()
    timer_ref: dict[str, ui.timer | None] = {"timer": None}
    full_df: dict[str, Any] = {"df": None}
    debounce_timer: dict[str, ui.timer | None] = {"t": None}
    slider_refs: dict[str, ui.slider] = {}
    value_labels: dict[str, Union[ui.label, ui.number]] = {}

    # UI elements to be initialized later
    plot_3d = None
    plot_2d = None
    time_slider = None
    animate_checkbox = None
    play_button = None

    def _run_sim():
        full_df["df"] = simulate_rocket(params)
        return full_df["df"]

    def _render(df=None):
        if df is None:
            df = full_df["df"] if full_df["df"] is not None else _run_sim()
        t_max = None
        if animate_checkbox is not None and animate_checkbox.value and time_slider is not None:
            t_max = time_slider.value
        if plot_3d is not None:
            plot_3d.update_figure(create_3d_figure(df, max_time=t_max))
        if plot_2d is not None:
            plot_2d.update_figure(create_2d_figures(df, max_time=t_max))

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

        debounce_timer["t"] = ui.timer(0.25, go, once=True)

    def _make_param_setter(attr):
        """Return a handler that sets params.<attr> and triggers debounced update."""
        def handler(e):
            val = float(e.args)
            setattr(params, attr, val)
            if attr in value_labels and hasattr(value_labels[attr], 'value'):
                value_labels[attr].value = val  # type: ignore
            _debounced_update()
        return handler

    def _make_input_setter(attr, min_val, max_val):
        """Return a handler for manual input that validates bounds."""
        def handler(e):
            try:
                val = float(e.value)
                # Clamp to bounds
                val = max(min_val, min(max_val, val))
                setattr(params, attr, val)
                if attr in slider_refs:
                    slider_refs[attr].value = val
                if attr in value_labels and hasattr(value_labels[attr], 'value'):
                    value_labels[attr].value = val  # type: ignore
                _debounced_update()
            except ValueError:
                pass  # Invalid input, ignore
        return handler

    def _labeled_slider(label_text, attr, **kwargs):
        """Label on the left, slider in middle, manual input on right."""
        min_val = kwargs.get('min', 0)
        max_val = kwargs.get('max', 100)
        current_val = kwargs.get('value', 0)
        
        with ui.row().classes("w-full items-center gap-2 mt-1"):
            ui.label(label_text).classes("text-xs font-semibold whitespace-nowrap min-w-[7rem]")
            s = ui.slider(**kwargs).classes("flex-grow").on("update:model-value", _make_param_setter(attr))
            slider_refs[attr] = s
            
            # Manual input field
            input_field = ui.number(
                value=current_val, 
                format="%.3g",
                min=min_val,
                max=max_val,
                step=kwargs.get('step', 0.1)
            ).props("dense outlined").classes("w-20").on('update:model-value', _make_input_setter(attr, min_val, max_val))
            
            value_labels[attr] = input_field  # Store reference for updates

    def _apply_preset(name: str):
        preset = PRESETS[name]
        for f in fields(RocketParams):
            val = getattr(preset, f.name)
            setattr(params, f.name, val)
            if f.name in slider_refs:
                slider_refs[f.name].value = val
            if f.name in value_labels and hasattr(value_labels[f.name], 'value'):
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

    with ui.row().classes("w-full h-screen items-stretch"):
        # ── Left column — scrollable settings sidebar ─────────────
        with ui.column().classes("w-96 min-w-[22rem] overflow-y-auto p-4 border-r"):
            ui.label("Aloe").classes("text-2xl font-bold text-center w-full")

            # Preset buttons
            with ui.card().classes("w-full mb-2"):
                ui.label("Presets").classes("text-lg font-bold mb-1")
                with ui.row().classes("w-full flex-wrap gap-1"):
                    for preset_name in PRESETS:
                        ui.button(
                            preset_name,
                            on_click=lambda e, _n=preset_name: _apply_preset(_n),
                        ).props(
                            "dense outline size=sm"
                        ).classes("text-xs")

            with ui.card().classes("w-full mb-2"):
                ui.label("Rocket Parameters").classes("text-lg font-bold mb-1")
                for label, attr, mn, mx, step in ROCKET_SLIDERS:
                    _labeled_slider(label, attr, min=mn, max=mx, step=step, value=getattr(params, attr))

            with ui.card().classes("w-full mb-2"):
                ui.label("Environment").classes("text-lg font-bold mb-1")
                for label, attr, mn, mx, step in ENV_SLIDERS:
                    _labeled_slider(label, attr, min=mn, max=mx, step=step, value=getattr(params, attr))

            with ui.card().classes("w-full mb-2"):
                ui.label("Playback").classes("text-lg font-bold mb-1")
                animate_checkbox = ui.checkbox("Animate over time", value=False, on_change=on_animate_toggle)
                time_slider = ui.slider(min=0, max=10, step=0.1, value=10).props("label")
                time_slider.on("update:model-value", on_time_change)
                play_button = ui.button("▶ Play", on_click=on_play).props("color=secondary").classes("w-full")

            with ui.row().classes("w-full gap-2"):
                ui.button("Excel", on_click=export_xlsx).props("color=primary dense").classes("flex-grow")
                ui.button("CSV", on_click=export_csv).props("color=primary outline dense").classes("flex-grow")

        # ── Right column — charts fill remaining space ────────────
        with ui.column().classes("flex-grow overflow-y-auto p-2"):
            df = _run_sim()
            plot_3d = ui.plotly(create_3d_figure(df)).classes("w-full")
            plot_2d = ui.plotly(create_2d_figures(df)).classes("w-full")
            if time_slider is not None:
                try:
                    max_time = df["time_s"].max()
                    if max_time is not None:
                        time_slider._props["max"] = round(float(max_time), 2)  # type: ignore
                        time_slider._props["value"] = time_slider._props["max"]
                        time_slider.update()
                except (TypeError, ValueError):
                    pass  # Skip if conversion fails
