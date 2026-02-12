document.addEventListener('alpine:init', () => {
    Alpine.data('simulator', () => ({
        simulationData: null,
        isReconciling: false,
        formatStat(val) {
            if (val === undefined || val === null) return '-';
            if (Math.abs(val) < 0.001) return val.toExponential(3);
            return val.toFixed(4);
        },
        formatInt(val) {
            if (val === undefined || val === null) return '-';
            return Math.round(val).toString();
        },
        init() {
            // Auto-run on init
            this.runSim();
            
            // Watch for form changes with debounce
            const form = document.getElementById('sim-form');
            let debounceTimer;
            
            const debouncedRun = () => {
                this.isReconciling = true;
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(() => {
                    this.runSim();
                }, 100);
            };
            
            form.addEventListener('change', debouncedRun);
            
            // Watch for input changes with debounce
            const inputs = form.querySelectorAll('input[type=number]');
            inputs.forEach(input => {
                input.addEventListener('input', debouncedRun);
            });
        },
        async runSim() {
            this.isReconciling = true;
            const form = document.getElementById('sim-form');
            const formData = new FormData(form);
            const params = new URLSearchParams();
            
            for (const [key, value] of formData.entries()) {
                params.append(key, value);
            }
            
            const checkboxes = form.querySelectorAll('input[type=checkbox]');
            checkboxes.forEach(cb => {
                params.append(cb.name, cb.checked ? 'true' : 'false');
            });
            
            try {
                const response = await fetch('/api/simulate?' + params);
                this.simulationData = await response.json();
                this.$nextTick(() => {
                    this.renderAllCharts();
                    this.isReconciling = false;
                });
            } catch (error) {
                console.error('Simulation error:', error);
                this.isReconciling = false;
            }
        },
        renderAllCharts() {
            if (!this.simulationData) return;
            this.renderTrajectory3D();
            this.renderAltitude();
            this.renderVelocity();
            this.renderAcceleration();
            this.renderForce();
            this.renderMass();
            this.renderAccel();
            this.renderGyro();
            this.populateErrorTable();
        },
        renderTrajectory3D() {
            const data = this.simulationData;
            
            const trueTrace = {
                x: data.position_x,
                y: data.position_y,
                z: data.position_z,
                mode: 'lines',
                type: 'scatter3d',
                name: 'True Path',
                line: { color: '#4CAF50', width: 4 },
                opacity: 0.8
            };
            
            const estTrace = {
                x: data.filter_data.est_pos_x,
                y: data.filter_data.est_pos_y,
                z: data.filter_data.est_pos_z,
                mode: 'lines',
                type: 'scatter3d',
                name: 'ESKF Estimate',
                line: { color: '#FF5722', width: 3, dash: 'dash' },
                opacity: 0.8
            };
            
            const quantTrace = {
                x: data.position_x.map((x, i) => x + (Math.sin(i * 0.1) * 0.5)),
                y: data.position_y.map((y, i) => y + (Math.cos(i * 0.1) * 0.5)),
                z: data.position_z,
                mode: 'lines',
                type: 'scatter3d',
                name: 'Quantized Path',
                line: { color: '#2196F3', width: 2, dash: 'dot' },
                opacity: 0.6
            };
            
            // FlightState colors: Pad, Ascent, Coast, Descent, Landed
            const stateColors = { 
                Pad: '#4CAF50', 
                Ascent: '#FF9800',
                Coast: '#2196F3', 
                Descent: '#9C27B0',
                Landed: '#795548'
            };
            
            // Get unique state changes for truth path
            const uniqueTruthStates = [];
            const seenTruthStates = new Set();
            if (data.state_changes_sim) {
                data.state_changes_sim.forEach(sc => {
                    const stateName = sc.state;
                    if (!seenTruthStates.has(stateName)) {
                        seenTruthStates.add(stateName);
                        uniqueTruthStates.push({...sc, state: stateName});
                    }
                });
            }
            
            // Truth state markers
            const truthStateTraces = uniqueTruthStates.map(sc => {
                const idx = Math.min(Math.floor(sc.time / 0.05), data.position_x.length - 1);
                return {
                    x: [data.position_x[idx] || 0],
                    y: [data.position_y[idx] || 0],
                    z: [data.position_z[idx] || 0],
                    mode: 'markers+text',
                    type: 'scatter3d',
                    name: `Sim: ${sc.state}`,
                    text: [sc.state],
                    textposition: 'top center',
                    textfont: { size: 10, color: stateColors[sc.state] || '#666' },
                    marker: { size: 12, color: stateColors[sc.state] || '#666', symbol: 'circle', line: { color: '#000', width: 2 } }
                };
            });
            
            // ESKF state markers
            const uniqueESKFStates = [];
            const seenESKFStates = new Set();
            if (data.state_changes_eskf) {
                data.state_changes_eskf.forEach(sc => {
                    const stateName = sc.state;
                    if (!seenESKFStates.has(stateName)) {
                        seenESKFStates.add(stateName);
                        uniqueESKFStates.push({...sc, state: stateName});
                    }
                });
            }
            
            const eskfStateTraces = uniqueESKFStates.map(sc => {
                const idx = Math.min(Math.floor(sc.time / 0.05), data.filter_data.est_pos_x.length - 1);
                return {
                    x: [data.filter_data.est_pos_x[idx] || 0],
                    y: [data.filter_data.est_pos_y[idx] || 0],
                    z: [data.filter_data.est_pos_z[idx] || 0],
                    mode: 'markers+text',
                    type: 'scatter3d',
                    name: `ESKF: ${sc.state}`,
                    text: [sc.state],
                    textposition: 'bottom center',
                    textfont: { size: 10, color: stateColors[sc.state] || '#666' },
                    marker: { size: 10, color: stateColors[sc.state] || '#666', symbol: 'diamond', line: { color: '#000', width: 2 } }
                };
            });
            
            const allTraces = [trueTrace, estTrace, quantTrace, ...truthStateTraces, ...eskfStateTraces];
            
            const xRange = [Math.min(...data.position_x) - 50, Math.max(...data.position_x) + 50];
            const yRange = [Math.min(...data.position_y) - 50, Math.max(...data.position_y) + 50];
            const zRange = [0, Math.min(Math.max(...data.position_z) * 1.1, 50000)];
            
            const layout = {
                title: {
                    text: '3D Flight Path Comparison',
                    font: { size: 16, color: '#1c1917' }
                },
                scene: {
                    xaxis: { title: 'East (m)', range: xRange, gridcolor: '#e5e5e5' },
                    yaxis: { title: 'North (m)', range: yRange, gridcolor: '#e5e5e5' },
                    zaxis: { title: 'Altitude (m)', range: zRange, gridcolor: '#e5e5e5' },
                    camera: { eye: { x: 1.5, y: 1.5, z: 0.8 } },
                    aspectmode: 'manual',
                    aspectratio: { x: 1, y: 1, z: 2 }
                },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                showlegend: true,
                legend: { x: 1, y: 1, bgcolor: 'rgba(255,255,255,0.8)', xanchor: 'right' },
                margin: { l: 0, r: 0, t: 50, b: 0 }
            };
            
            Plotly.newPlot('chart-trajectory', allTraces, layout, {responsive: true});
        },
        renderAltitude() {
            const data = this.simulationData;
            this.render2DChart('chart-altitude', data.time, data.altitude, null, 'Altitude vs Time', 'Time (s)', 'Altitude (m)');
        },
        renderVelocity() {
            const data = this.simulationData;
            this.render2DChart('chart-velocity', data.time, data.velocity, null, 'Velocity vs Time', 'Time (s)', 'Velocity (m/s)');
        },
        renderAcceleration() {
            const data = this.simulationData;
            this.render2DChart('chart-acceleration', data.time, data.acceleration, null, 'Acceleration vs Time', 'Time (s)', 'Acceleration (m/s²)');
        },
        renderForce() {
            const data = this.simulationData;
            this.render2DChart('chart-force', data.time, data.force, null, 'Net Force vs Time', 'Time (s)', 'Force (N)');
        },
        renderMass() {
            const data = this.simulationData;
            this.render2DChart('chart-mass', data.time, data.mass, null, 'Mass vs Time', 'Time (s)', 'Mass (kg)');
        },
        render2DChart(elementId, time, trueData, estData, title, xLabel, yLabel) {
            const traces = [{
                x: time,
                y: trueData,
                mode: 'lines',
                name: 'Simulated',
                line: { color: '#4CAF50', width: 2 }
            }];
            
            if (estData) {
                traces.push({
                    x: time,
                    y: estData,
                    mode: 'lines',
                    name: 'ESKF',
                    line: { color: '#FF5722', width: 2, dash: 'dash' }
                });
            }
            
            const data = this.simulationData;
            const uniqueStates = [];
            const seen = new Set();
            data.state_changes_sim.forEach(sc => {
                if (!seen.has(sc.state)) {
                    seen.add(sc.state);
                    uniqueStates.push(sc);
                }
            });
            const shapes = uniqueStates.map(sc => ({
                type: 'line',
                x0: sc.time,
                x1: sc.time,
                y0: 0,
                y1: Math.max(...trueData) * 1.1,
                line: { color: '#9C27B0', width: 2, dash: 'dash' }
            }));
            
            const annotations = uniqueStates.map((sc, idx) => ({
                x: sc.time,
                y: Math.max(...trueData) * (0.9 - idx * 0.15),
                text: sc.description,
                showarrow: true,
                arrowhead: 2,
                ax: 40,
                ay: 0,
                bgcolor: 'rgba(255,255,255,0.9)',
                bordercolor: '#9C27B0',
                borderwidth: 1
            }));
            
            const annotations = data.state_changes_sim.map((sc, idx) => ({
                x: sc.time,
                y: Math.max(...trueData) * (0.9 - idx * 0.15),
                text: sc.state,
                showarrow: true,
                arrowhead: 2,
                ax: 40,
                ay: 0,
                bgcolor: 'rgba(255,255,255,0.9)',
                bordercolor: '#9C27B0',
                borderwidth: 1
            }));
            
            const layout = {
                title: {
                    text: title,
                    font: { size: 14, color: '#1c1917' }
                },
                xaxis: { 
                    title: { text: xLabel, font: { size: 12 } }, 
                    gridcolor: '#e5e5e5' 
                },
                yaxis: { 
                    title: { text: yLabel, font: { size: 12 } }, 
                    gridcolor: '#e5e5e5' 
                },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                shapes: shapes,
                annotations: annotations,
                showlegend: true,
                legend: { x: 1, y: 1, xanchor: 'right' },
                margin: { l: 60, r: 40, t: 50, b: 50 }
            };
            
            Plotly.newPlot(elementId, traces, layout, {responsive: true});
        },
        renderAccel() {
            const data = this.simulationData;
            const sensor = data.sensor_data;
            
            const traces = [
                { x: data.time, y: sensor.accel_x, name: 'Accel X', line: { color: '#F44336', width: 1.5 } },
                { x: data.time, y: sensor.accel_y, name: 'Accel Y', line: { color: '#2196F3', width: 1.5 } },
                { x: data.time, y: sensor.accel_z, name: 'Accel Z', line: { color: '#4CAF50', width: 1.5 } }
            ];
            
            const layout = {
                title: {
                    text: 'BMI088 Accelerometer',
                    font: { size: 14, color: '#1c1917' }
                },
                xaxis: { 
                    title: { text: 'Time (s)', font: { size: 12 } }, 
                    gridcolor: '#e5e5e5' 
                },
                yaxis: { 
                    title: { text: 'Acceleration (m/s²)', font: { size: 12 } }, 
                    gridcolor: '#e5e5e5' 
                },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                showlegend: true,
                legend: { x: 1, y: 1, xanchor: 'right' },
                margin: { l: 60, r: 40, t: 50, b: 50 }
            };
            
            Plotly.newPlot('chart-accel', traces, layout, {responsive: true});
        },
        renderGyro() {
            const data = this.simulationData;
            const sensor = data.sensor_data;
            
            const traces = [
                { x: data.time, y: sensor.gyro_x, name: 'Gyro X', line: { color: '#FF9800', width: 1.5 } },
                { x: data.time, y: sensor.gyro_y, name: 'Gyro Y', line: { color: '#9C27B0', width: 1.5 } },
                { x: data.time, y: sensor.gyro_z, name: 'Gyro Z', line: { color: '#00BCD4', width: 1.5 } }
            ];
            
            const layout = {
                title: {
                    text: 'BMI088 Gyroscope',
                    font: { size: 14, color: '#1c1917' }
                },
                xaxis: { 
                    title: { text: 'Time (s)', font: { size: 12 } }, 
                    gridcolor: '#e5e5e5' 
                },
                yaxis: { 
                    title: { text: 'Angular Rate (rad/s)', font: { size: 12 } }, 
                    gridcolor: '#e5e5e5' 
                },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                showlegend: true,
                legend: { x: 1, y: 1, xanchor: 'right' },
                margin: { l: 60, r: 40, t: 50, b: 50 }
            };
            
            Plotly.newPlot('chart-gyro', traces, layout, {responsive: true});
        },
        populateErrorTable() {
            const data = this.simulationData;
            if (!data.error_stats) return;
            const table = document.getElementById('error-table');
            const tbody = table.querySelector('tbody');
            tbody.innerHTML = '';
            const stats = data.error_stats;
            const rows = [
                ['Position N', stats.pos_n_min, stats.pos_n_max, stats.pos_n_mean, stats.pos_n_std, stats.pos_n_rmse],
                ['Position E', stats.pos_e_min, stats.pos_e_max, stats.pos_e_mean, stats.pos_e_std, stats.pos_e_rmse],
                ['Position D', stats.pos_d_min, stats.pos_d_max, stats.pos_d_mean, stats.pos_d_std, stats.pos_d_rmse],
                ['Velocity N', stats.vel_n_min, stats.vel_n_max, stats.vel_n_mean, stats.vel_n_std, stats.vel_n_rmse],
                ['Velocity E', stats.vel_e_min, stats.vel_e_max, stats.vel_e_mean, stats.vel_e_std, stats.vel_e_rmse],
                ['Velocity D', stats.vel_d_min, stats.vel_d_max, stats.vel_d_mean, stats.vel_d_std, stats.vel_d_rmse],
                ['3D Position', stats.pos_3d_min, stats.pos_3d_max, stats.pos_3d_mean, stats.pos_3d_std, stats.pos_3d_rmse],
            ];
            rows.forEach(row => {
                const tr = document.createElement('tr');
                row.forEach(cell => {
                    const td = document.createElement('td');
                    td.textContent = this.formatStat(cell);
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });
        }
    }));
});

// Rocket presets
const rocketPresets = {
    '30km': {
        dry_mass: 20,
        propellant_mass: 750,
        thrust: 160000,
        burn_time: 10,
        drag_coeff: 0.40,
        ref_area: 0.0324,
        wind_north: 3,
        crosswind: 0
    },
    '12km': {
        dry_mass: 30,
        propellant_mass: 60,
        thrust: 6000,
        burn_time: 12,
        drag_coeff: 0.45,
        ref_area: 0.0324,
        wind_north: 5,
        crosswind: 2
    },
    'high-drag': {
        dry_mass: 50,
        propellant_mass: 150,
        thrust: 15000,
        burn_time: 25,
        drag_coeff: 1.8,
        ref_area: 0.324,
        wind_north: 0,
        crosswind: 0
    }
};

function applyPreset(presetName) {
    const preset = rocketPresets[presetName];
    if (!preset) return;
    
    // Update form fields
    Object.keys(preset).forEach(key => {
        const input = document.querySelector(`[name="${key}"]`);
        if (input) {
            input.value = preset[key];
        }
    });
    
    // Trigger simulation update
    const form = document.getElementById('sim-form');
    const event = new Event('change');
    form.dispatchEvent(event);
}

window.applyPreset = applyPreset;
