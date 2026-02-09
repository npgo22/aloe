import polars as pl
import math
from dataclasses import dataclass


@dataclass
class RocketParams:
    """All inputs to the rocket flight simulation."""

    dry_mass: float = 50.0  # Mass of the rocket without propellant (kg)
    propellant_mass: float = 150.0  # Mass of the propellant (kg)
    thrust: float = 15000.0  # Engine thrust force (N)
    burn_time: float = 25.0  # Duration the engine fires (s)
    drag_coeff: float = 0.40  # Aerodynamic drag coefficient (dimensionless)
    ref_area: float = 0.03  # Reference cross-sectional area for drag (m²)
    gravity: float = 9.81  # Gravitational acceleration (m/s²)
    wind_speed: float = 3.0  # Wind speed along X axis (m/s)
    wind_speed_z: float = 0.0  # Crosswind speed along Z axis (m/s)
    air_density: float = 1.225  # Ambient air density (kg/m³)


# ── Preset configurations ────────────────────────────────────────────
PRESETS: dict[str, RocketParams] = {
    # 8 in rocket diameter -> .0324m^2
    # No idea what to set anything else to
    # (still need prop/vehicle data)
    "30 km (rip)": RocketParams(
        dry_mass=50,
        propellant_mass=150,
        thrust=15000,
        burn_time=25,
        drag_coeff=0.40,
        ref_area=0.0324,
        wind_speed=3,
        wind_speed_z=0,
    ),
    "12 km": RocketParams(
        dry_mass=30,
        propellant_mass=60,
        thrust=6000,
        burn_time=12,
        drag_coeff=0.45,
        ref_area=0.0324,
        wind_speed=5,
        wind_speed_z=2,
    ),
    "High-drag test": RocketParams(
        dry_mass=50,
        propellant_mass=150,
        thrust=15000,
        burn_time=25,
        drag_coeff=1.8,
        ref_area=0.324,
        wind_speed=0,
        wind_speed_z=0,
    ),
}


def simulate_rocket(p: RocketParams) -> pl.DataFrame:
    """Simulate rocket flight and return a Polars DataFrame of the trajectory.

    Parameters
    ----------
    p : RocketParams
        All physical and environmental parameters for the simulation.
    """
    time_step = 0.01  # Integration time step (s)
    max_time = 300.0  # Maximum simulation duration (s)
    time_steps = int(max_time / time_step)

    # Arrays to store data
    times = []
    positions_x = []
    positions_y = []
    positions_z = []
    velocities_x = []
    velocities_y = []
    velocities_z = []
    accelerations_x = []
    accelerations_y = []
    accelerations_z = []
    masses = []
    drags = []
    thrusts = []

    # Initial conditions
    t = 0.0
    x = 0.0
    y = 0.0
    z = 0.0
    vx = 0.0
    vy = 0.0
    vz = 0.0

    for step in range(time_steps):
        # Current mass
        if t <= p.burn_time:
            mass_flow_rate = p.propellant_mass / p.burn_time
            current_mass = p.dry_mass + p.propellant_mass - (mass_flow_rate * t)
        else:
            current_mass = p.dry_mass

        # Thrust
        thrust_force = p.thrust if t <= p.burn_time else 0.0

        # Velocity magnitude
        v_total = math.sqrt(vx**2 + vy**2 + vz**2)

        # Drag force
        if v_total > 0:
            drag_force = 0.5 * p.air_density * v_total**2 * p.drag_coeff * p.ref_area
            drag_x = -drag_force * (vx / v_total)
            drag_y = -drag_force * (vy / v_total)
            drag_z = -drag_force * (vz / v_total)
        else:
            drag_force = 0.0
            drag_x = 0.0
            drag_y = 0.0
            drag_z = 0.0

        # Forces
        thrust_x = 0.0
        thrust_y = thrust_force
        thrust_z = 0.0
        wind_force_x = p.wind_speed * 0.5
        wind_force_y = 0.0
        wind_force_z = p.wind_speed_z * 0.5
        gravity_x = 0.0
        gravity_y = -current_mass * p.gravity
        gravity_z = 0.0

        # Total forces
        total_force_x = thrust_x + drag_x + wind_force_x + gravity_x
        total_force_y = thrust_y + drag_y + wind_force_y + gravity_y
        total_force_z = thrust_z + drag_z + wind_force_z + gravity_z

        # Accelerations
        ax = total_force_x / current_mass
        ay = total_force_y / current_mass
        az = total_force_z / current_mass

        # Store data
        times.append(t)
        positions_x.append(x)
        positions_y.append(y)
        positions_z.append(z)
        velocities_x.append(vx)
        velocities_y.append(vy)
        velocities_z.append(vz)
        accelerations_x.append(ax)
        accelerations_y.append(ay)
        accelerations_z.append(az)
        masses.append(current_mass)
        drags.append(drag_force)
        thrusts.append(thrust_force)

        # Update state
        vx += ax * time_step
        vy += ay * time_step
        vz += az * time_step
        x += vx * time_step
        y += vy * time_step
        z += vz * time_step

        # Stop if rocket hits ground
        if y < 0 and t > 0:
            break

        t += time_step

    # Create DataFrame
    df = pl.DataFrame(
        {
            "time_s": times,
            "position_x_m": positions_x,
            "altitude_m": positions_y,
            "position_z_m": positions_z,
            "velocity_x_ms": velocities_x,
            "velocity_y_ms": velocities_y,
            "velocity_z_ms": velocities_z,
            "acceleration_x_ms2": accelerations_x,
            "acceleration_y_ms2": accelerations_y,
            "acceleration_z_ms2": accelerations_z,
            "mass_kg": masses,
            "drag_force_N": drags,
            "thrust_N": thrusts,
        },
        strict=False,
    )

    df = df.with_columns(
        [
            (pl.col("velocity_x_ms") ** 2 + pl.col("velocity_y_ms") ** 2 + pl.col("velocity_z_ms") ** 2)
            .sqrt()
            .alias("velocity_total_ms"),
            (pl.col("acceleration_x_ms2") ** 2 + pl.col("acceleration_y_ms2") ** 2 + pl.col("acceleration_z_ms2") ** 2)
            .sqrt()
            .alias("acceleration_total_ms2"),
        ]
    )

    return df
