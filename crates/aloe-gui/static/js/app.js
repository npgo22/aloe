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
                if (!cb.checked) {
                    params.append(cb.name, 'false');
                }
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
            this.renderMag();
            this.renderGps();
            this.renderAdxl();
            this.renderBaro();
        },
        renderTrajectory3D() {
            const data = this.simulationData;
            
            const trueTrace = {
                x: data.position_y,
                y: data.position_x,
                z: data.position_z,
                mode: 'lines',
                type: 'scatter3d',
                name: 'True Path',
                line: { color: '#4CAF50', width: 4 },
                opacity: 0.8
            };
            
            const estTrace = {
                x: data.filter_data.est_pos_y,
                y: data.filter_data.est_pos_x,
                z: data.filter_data.est_pos_z.map(z => -z),
                mode: 'lines',
                type: 'scatter3d',
                name: 'ESKF Estimate',
                line: { color: '#FF5722', width: 3, dash: 'dash' },
                opacity: 0.8
            };
            
            const quantTrace = {
                x: data.filter_data.quantized_est_pos_y,
                y: data.filter_data.quantized_est_pos_x,
                z: data.filter_data.quantized_est_pos_z,
                mode: 'lines',
                type: 'scatter3d',
                name: 'Quantized ESKF',
                line: { color: '#2196F3', width: 2, dash: 'dot' },
                opacity: 0.6
            };
            
            // FlightState colors: Pad, Ascent, Coast, Descent, Landed
            const stateColors = { 
                Pad: '#4CAF50', 
                Ascent: '#FF9800',
                Burn: '#FF5722',
                Coast: '#2196F3', 
                Descent: '#9C27B0',
                Recovery: '#9C27B0',
                Landed: '#795548'
            };
            
            // Get unique state changes for truth path
            const uniqueTruthStates = [];
            const seenTruthStates = new Set();
            if (data.state_changes_sim) {
                data.state_changes_sim.forEach(sc => {
                    const stateName = sc.state || sc.description;
                    if (!seenTruthStates.has(stateName)) {
                        seenTruthStates.add(stateName);
                        uniqueTruthStates.push({...sc, state: stateName});
                    }
                });
            }
            
            // Truth state markers
            const truthStateTraces = uniqueTruthStates
                .filter(sc => sc.state !== 'Recovery')
                .map(sc => {
                    const idx = data.time.findIndex(t => t >= sc.time) !== -1 ? data.time.findIndex(t => t >= sc.time) : data.time.length - 1;
                    return {
                        x: [data.position_y[idx] || 0],
                        y: [data.position_x[idx] || 0],
                        z: [data.position_z[idx] || 0],
                        mode: 'markers+text',
                        type: 'scatter3d',
                        name: `Truth: ${sc.state}`,
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
                    const stateName = sc.state || sc.description;
                    if (!seenESKFStates.has(stateName)) {
                        seenESKFStates.add(stateName);
                        uniqueESKFStates.push({...sc, state: stateName});
                    }
                });
            }
            
            const eskfStateTraces = uniqueESKFStates
                .filter(sc => sc.state !== 'Recovery')
                .map(sc => {
                    const idx = data.time.findIndex(t => t >= sc.time) !== -1 ? data.time.findIndex(t => t >= sc.time) : data.time.length - 1;
                    return {
                        x: [data.filter_data.est_pos_y[idx] || 0],
                        y: [data.filter_data.est_pos_x[idx] || 0],
                        z: [-(data.filter_data.est_pos_z[idx] || 0)],
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
            
            const xRange = [Math.min(...data.position_y) - 50, Math.max(...data.position_y) + 50];
            const yRange = [Math.min(...data.position_x) - 50, Math.max(...data.position_x) + 50];
            const zRange = [0, Math.max(...data.position_z) * 1.1];
            
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
                    aspectmode: 'cube'
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
            console.log("Altitude data:", {
                length: data.altitude?.length,
                max: Math.max(...(data.altitude || [0])),
                last: data.altitude?.[data.altitude.length - 1]
            });
            this.render2DChart('chart-altitude', data.time, data.altitude, data.filter_data.est_pos_z, 'Altitude vs Time', 'Time (s)', 'Altitude (m)');
        },
        renderVelocity() {
            const data = this.simulationData;
            this.render2DChart('chart-velocity', data.time, data.velocity, data.filter_data.est_vel_mag, 'Velocity vs Time', 'Time (s)', 'Velocity (m/s)');
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
            const shapes = data.state_changes_sim.map(sc => ({
                type: 'line',
                x0: sc.time,
                x1: sc.time,
                y0: 0,
                y1: Math.max(...trueData) * 1.1,
                line: { color: '#9C27B0', width: 2, dash: 'dash' }
            }));
            
            const annotations = data.state_changes_sim.map((sc, idx) => ({
                x: sc.time,
                y: Math.max(...trueData) * (0.9 - idx * 0.15),
                text: sc.state || sc.description,
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
        renderMag() {
            const data = this.simulationData;
            const sensor = data.sensor_data;
            
            const traces = [
                { x: data.time, y: sensor.mag_x, name: 'Mag X', line: { color: '#E91E63', width: 1.5 } },
                { x: data.time, y: sensor.mag_y, name: 'Mag Y', line: { color: '#673AB7', width: 1.5 } },
                { x: data.time, y: sensor.mag_z, name: 'Mag Z', line: { color: '#3F51B5', width: 1.5 } }
            ];
            
            const layout = {
                title: {
                    text: 'LIS3MDL Magnetometer',
                    font: { size: 14, color: '#1c1917' }
                },
                xaxis: { 
                    title: { text: 'Time (s)', font: { size: 12 } }, 
                    gridcolor: '#e5e5e5' 
                },
                yaxis: { 
                    title: { text: 'Magnetic Field (Gauss)', font: { size: 12 } }, 
                    gridcolor: '#e5e5e5' 
                },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                showlegend: true,
                legend: { x: 1, y: 1, xanchor: 'right' },
                margin: { l: 60, r: 40, t: 50, b: 50 }
            };
            
            Plotly.newPlot('chart-mag', traces, layout, {responsive: true});
        },
        renderGps() {
            const data = this.simulationData;
            const sensor = data.sensor_data;
            
            const traces = [
                { x: data.time, y: sensor.gps_x, name: 'GPS X', line: { color: '#009688', width: 1.5 } },
                { x: data.time, y: sensor.gps_y, name: 'GPS Y', line: { color: '#FF5722', width: 1.5 } },
                { x: data.time, y: sensor.gps_z, name: 'GPS Z', line: { color: '#795548', width: 1.5 } },
                { x: data.time, y: sensor.gps_vel_x, name: 'GPS Vel X', line: { color: '#607D8B', width: 1.5 }, yaxis: 'y2' },
                { x: data.time, y: sensor.gps_vel_y, name: 'GPS Vel Y', line: { color: '#8BC34A', width: 1.5 }, yaxis: 'y2' },
                { x: data.time, y: sensor.gps_vel_z, name: 'GPS Vel Z', line: { color: '#FFC107', width: 1.5 }, yaxis: 'y2' }
            ];
            
            const layout = {
                title: {
                    text: 'GPS Position & Velocity',
                    font: { size: 14, color: '#1c1917' }
                },
                xaxis: { 
                    title: { text: 'Time (s)', font: { size: 12 } }, 
                    gridcolor: '#e5e5e5' 
                },
                yaxis: { 
                    title: { text: 'Position (m)', font: { size: 12 } }, 
                    gridcolor: '#e5e5e5' 
                },
                yaxis2: {
                    title: { text: 'Velocity (m/s)', font: { size: 12 } },
                    overlaying: 'y',
                    side: 'right',
                    gridcolor: '#e5e5e5'
                },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                showlegend: true,
                legend: { x: 1, y: 1, xanchor: 'right' },
                margin: { l: 60, r: 40, t: 50, b: 50 }
            };
            
            Plotly.newPlot('chart-gps', traces, layout, {responsive: true});
        },
        renderAdxl() {
            const data = this.simulationData;
            const sensor = data.sensor_data;
            
            // Chart for Y and Z
            const tracesYz = [
                { x: data.time, y: sensor.adxl_y, name: 'ADXL375 Y', line: { color: '#2196F3', width: 1.5 } },
                { x: data.time, y: sensor.adxl_z, name: 'ADXL375 Z', line: { color: '#4CAF50', width: 1.5 } }
            ];
            
            const layoutYz = {
                title: {
                    text: 'ADXL375 Accelerometer (Y & Z)',
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
            
            Plotly.newPlot('chart-adxl-xy', tracesYz, layoutYz, {responsive: true});
            
            // Chart for X
            const tracesX = [
                { x: data.time, y: sensor.adxl_x, name: 'ADXL375 X', line: { color: '#FF5722', width: 1.5 } }
            ];
            
            const layoutX = {
                title: {
                    text: 'ADXL375 Accelerometer (X)',
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
            
            Plotly.newPlot('chart-adxl-z', tracesX, layoutX, {responsive: true});
        },
        renderBaro() {
            const data = this.simulationData;
            const sensor = data.sensor_data;
            
            // Calculate altitude from pressure
            const p0 = 101325.0;
            const h_scale = 8500.0;
            const baro_alt = sensor.baro_pressure.map(p => h_scale * Math.log(p0 / p));
            
            // Chart for Pressure
            const tracesPressure = [
                { x: data.time, y: sensor.baro_pressure, name: 'MS5611 Pressure', line: { color: '#9C27B0', width: 1.5 } }
            ];
            
            const layoutPressure = {
                title: {
                    text: 'MS5611 Barometer Pressure',
                    font: { size: 14, color: '#1c1917' }
                },
                xaxis: { 
                    title: { text: 'Time (s)', font: { size: 12 } }, 
                    gridcolor: '#e5e5e5' 
                },
                yaxis: { 
                    title: { text: 'Pressure (Pa)', font: { size: 12 } }, 
                    gridcolor: '#e5e5e5' 
                },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                showlegend: true,
                legend: { x: 1, y: 1, xanchor: 'right' },
                margin: { l: 60, r: 40, t: 50, b: 50 }
            };
            
            Plotly.newPlot('chart-baro-pressure', tracesPressure, layoutPressure, {responsive: true});
            
            // Chart for Calculated Altitude
            const tracesAlt = [
                { x: data.time, y: baro_alt, name: 'MS5611 Calculated Altitude', line: { color: '#FF9800', width: 1.5 } }
            ];
            
            const layoutAlt = {
                title: {
                    text: 'MS5611 Barometer Calculated Altitude',
                    font: { size: 14, color: '#1c1917' }
                },
                xaxis: { 
                    title: { text: 'Time (s)', font: { size: 12 } }, 
                    gridcolor: '#e5e5e5' 
                },
                yaxis: { 
                    title: { text: 'Altitude (m)', font: { size: 12 } }, 
                    gridcolor: '#e5e5e5' 
                },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                showlegend: true,
                legend: { x: 1, y: 1, xanchor: 'right' },
                margin: { l: 60, r: 40, t: 50, b: 50 }
            };
            
            Plotly.newPlot('chart-baro-altitude', tracesAlt, layoutAlt, {responsive: true});
        }
    }));
});

// Rocket presets
const rocketPresets = {
    '30km': {
        dry_mass: 50,
        propellant_mass: 150,
        thrust: 15000,
        burn_time: 25,
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