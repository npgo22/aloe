# JSBSim Integration Plan

## Goal

Replace Aloe's old native simulator implementation with a JSBSim-backed backend while preserving the existing Aloe simulation boundary and downstream sensor/filter consumers.

## Scope

- Build JSBSim from source via a git submodule.
- Use the JSBSim C++ API through a small Aloe-owned C++ wrapper.
- Expose a narrow Rust FFI surface through a new sys crate.
- Add a safe Rust wrapper crate that converts Aloe inputs and outputs.
- Preserve the existing sensor and filter pipeline by continuing to emit Aloe `SimResult` values.
- Expose meaningful JSBSim-backed rocket parameters through the CLI and GUI instead of hardcoding them where practical.

## Non-Goals

- Replacing `aloe-core`.
- Reintroducing the old native backend as the active simulation path.
- Porting all Aloe-specific physics into JSBSim internals.
- Matching a retired native backend numerically for every case.

## Architecture

### Workspace additions

- `third_party/jsbsim` git submodule pinned to a known commit/tag.
- `crates/aloe-jsbsim-sys`
  - `build.rs` compiles JSBSim from the vendored submodule using CMake.
  - Compiles Aloe's C++ wrapper.
  - Exposes raw FFI declarations.
- `crates/aloe-jsbsim`
  - Safe Rust wrapper over `aloe-jsbsim-sys`.
  - Maps `RocketParams` to a generated JSBSim model/input set.
  - Runs JSBSim and converts output back into Aloe `SimResult`.

### Backend seam

Preserve Aloe's existing public sim boundary.

- Keep `simulate_6dof(&RocketParams) -> SimResult`.
- Implement that boundary internally with JSBSim.

### CLI and GUI

- CLI and GUI continue to target the same Aloe `RocketParams` boundary.
- No backend selector is needed in the current direction; JSBSim is the active backend.
- CLI/GUI expose additional rocket-model parameters that now materially affect the generated JSBSim model.

## C++ Wrapper Design

The wrapper should hide JSBSim internals and expose only the operations Aloe needs.

### C ABI surface

- `aloe_jsbsim_create()`
- `aloe_jsbsim_destroy()`
- `aloe_jsbsim_load_model(config_dir, aircraft_name)`
- `aloe_jsbsim_run_simulation(config_json, output_struct)`
- `aloe_jsbsim_last_error()`

The first implementation may use a single-shot run entrypoint to minimize lifetime complexity.

### Why a wrapper

- Avoid binding raw JSBSim C++ types into Rust.
- Keep exceptions, STL, and property-tree logic on the C++ side.
- Reduce breakage if JSBSim internals change.

## JSBSim Model Strategy

Generate temporary JSBSim model files from Aloe parameters.

### First-pass generated assets

- Aircraft definition XML.
- Mass balance XML.
- Propulsion XML using rocket engine / thrust table.
- Initial conditions script.

### First-pass model fidelity

- Rigid body mass/inertia.
- CG shift approximation.
- Thrust curve.
- Gravity and atmosphere.
- Simple wind vector.

## Exposed Parameters

The following simulator-relevant parameters are now exposed through Aloe instead of being buried in the wrapper:

- `normal_force_coeff`
- `cg_full`
- `cg_empty`
- `cp_location`
- `inertia_x`, `inertia_y`, `inertia_z`
- `isp`
- `nozzle_location`
- `launch_rod_length`
- explicit thrust curves via `--thrust-curve`

Current UI exposure is narrower than CLI exposure. The GUI currently exposes:

- `normal_force_coeff`
- `isp`
- `nozzle_location`

## Remaining Hardcoded Wrapper Assumptions

Some generated-model details are still fixed in the wrapper and should be treated as current implementation assumptions rather than user-tunable rocket parameters:

- simple aerodynamic force model structure using axial drag plus linear lift/side-force terms
- nozzle exit area and pressure
- ground contact spring/damping/friction values
- a generated `builduptime` heuristic when the thrust curve starts immediately above zero
- landing termination heuristic used to stop long simulations once the vehicle has clearly flown and returned to ground

### Deferred fidelity work

- Wind shear and turbulence.
- Recovery/parachute phases.
- Rail/launcher behavior parity.
- Higher-fidelity aero coefficient generation.

## Output Mapping

The safe wrapper must convert JSBSim output into Aloe conventions.

- Position: NED meters.
- Velocity: NED m/s.
- Orientation: body-to-NED quaternion.
- Angular rates: body rad/s.
- Proper acceleration: body m/s^2 if available, otherwise derived carefully.
- Event extraction:
  - ascent
  - coast
  - descent

## Build Strategy

### Submodule

- Add `third_party/jsbsim` as a git submodule.
- Pin to a stable release tag or commit.

### `build.rs`

- Configure JSBSim CMake build in an out-of-tree build directory.
- Build static library if practical.
- Build Aloe wrapper against JSBSim headers and library.
- Emit `cargo:rustc-link-search` and `cargo:rustc-link-lib` directives.
- Rebuild if wrapper sources or submodule revision changes.

### Cargo feature gating

- `aloe-sim` feature: `jsbsim`
- `aloe-cli`, `aloe-gui`, and `aloe` forward the feature.
- Keep JSBSim optional to preserve current developer workflow.

## Verification Plan

### Unit level

- Wrapper loads a minimal model and runs a short simulation.
- Output arrays have consistent lengths.
- Coordinate conversion sanity checks.

### Integration level

- CLI single-run simulation works through JSBSim.
- CLI tune-sweep / greedy tuning works through JSBSim.
- GUI builds and consumes the same JSBSim-backed `SimResult` flow.
- Sensor generation and filter execution continue to work from JSBSim `SimResult`.

### Comparison cases

- Vertical no-wind launch.
- Constant crosswind drift.
- Spin + thrust cant spiral case.
- Timing checks: launch, burnout, apogee, landing.

## Risks

- JSBSim model authoring may dominate the effort.
- Launch rail and proper-accel semantics may differ from Aloe native sim.
- Unit mismatches between JSBSim conventions and Aloe SI/NED conventions.
- LGPL review required.

## Status

Implemented:

- JSBSim submodule and source build from `build.rs`
- Aloe-owned C++ wrapper plus Rust FFI crates
- `simulate_6dof(&RocketParams) -> SimResult` backed by JSBSim
- CLI and GUI parameter plumbing for the exposed fields above
- full thrust-curve plumbing from Aloe into generated JSBSim rocket-engine tables
- release-mode greedy tuning smoke passing on the JSBSim backend

Still worth improving:

- suppress remaining JSBSim startup logging noise during batch/tuning runs
- improve wrapper/model fidelity beyond the current simplified aero/contact assumptions
