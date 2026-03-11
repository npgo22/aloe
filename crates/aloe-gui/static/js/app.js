// Unit conversion utilities
const UNITS = {
    metric: {
        distance: { factor: 1, suffix: 'm', name: 'meters' },
        distance_km: { factor: 0.001, suffix: 'km', name: 'kilometers' },
        velocity: { factor: 1, suffix: 'm/s', name: 'm/s' },
        acceleration: { factor: 1, suffix: 'm/s²', name: 'm/s²' },
        force: { factor: 1, suffix: 'N', name: 'Newtons' },
        mass: { factor: 1, suffix: 'kg', name: 'kg' },
        pressure: { factor: 1, suffix: 'Pa', name: 'Pascals' },
    },
    imperial: {
        distance: { factor: 3.28084, suffix: 'ft', name: 'feet' },
        distance_km: { factor: 3280.84, suffix: 'ft', name: 'feet' },
        velocity: { factor: 3.28084, suffix: 'ft/s', name: 'ft/s' },
        acceleration: { factor: 3.28084, suffix: 'ft/s²', name: 'ft/s²' },
        force: { factor: 0.224809, suffix: 'lbf', name: 'pound-force' },
        mass: { factor: 2.20462, suffix: 'lb', name: 'pounds' },
        pressure: { factor: 0.000145038, suffix: 'psi', name: 'psi' },
    }
};

function getUnits() {
    return localStorage.getItem('aloe-units') || 'metric';
}

function convertValue(val, type) {
    const unitSystem = UNITS[getUnits()];
    const unit = unitSystem[type];
    return val * unit.factor;
}

function getUnitSuffix(type) {
    const unitSystem = UNITS[getUnits()];
    return unitSystem[type].suffix;
}

function convertArray(arr, type) {
    const factor = UNITS[getUnits()][type].factor;
    return arr.map(v => v * factor);
}

// Redraw charts when units change
window.addEventListener('units-changed', () => {
    // Trigger a re-render if simulator data exists
    const simData = document.querySelector('[x-data="simulator"]');
    if (simData && simData.__x && simData.__x.$data.simulationData) {
        simData.__x.$data.renderCharts();
    }
});

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
        formatValue(val, type, decimals = 2) {
            if (val === undefined || val === null) return '-';
            const converted = convertValue(val, type);
            if (Math.abs(converted) < 0.01) return converted.toExponential(3);
            return converted.toFixed(decimals) + ' ' + getUnitSuffix(type);
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
            this.renderGpsPosition();
            this.renderGpsVelocity();
            this.renderAdxl();
            this.renderBaro();
            this.renderErrorPosition();
            this.renderErrorVelocity();
            this.renderErrorAltitude();
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

            // Add apogee marker (highest altitude point)
            // const apogeeIdx = data.position_z.indexOf(Math.max(...data.position_z));
            // const apogeeTrace = {
            //     x: [data.position_y[apogeeIdx]],
            //     y: [data.position_x[apogeeIdx]],
            //     z: [data.position_z[apogeeIdx]],
            //     mode: 'markers+text',
            //     type: 'scatter3d',
            //     name: 'Apogee',
            //     text: [`Apogee: ${data.position_z[apogeeIdx].toFixed(1)}m`],
            //     textposition: 'top center',
            //     textfont: { size: 12, color: '#FF5722', weight: 'bold' },
            //     marker: { size: 16, color: '#FF5722', symbol: 'diamond', line: { color: '#000', width: 3 } }
            // };

            const allTraces = [trueTrace, estTrace, quantTrace, ...truthStateTraces, ...eskfStateTraces];
            
            const xRange = [Math.min(...data.position_y) - 50, Math.max(...data.position_y) + 50];
            const yRange = [Math.min(...data.position_x) - 50, Math.max(...data.position_x) + 50];
            const zRange = [0, Math.max(...data.position_z) * 1.1];
            
            const layout = {
                title: {
                    text: '3D Flight Path Comparison (NED Coordinates)',
                    font: { size: 16, color: '#1c1917', weight: 'bold' }
                },
                scene: {
                    xaxis: {
                        title: { text: 'East (m)', font: { size: 14, weight: 'bold' } },
                        range: xRange,
                        gridcolor: '#e5e5e5'
                    },
                    yaxis: {
                        title: { text: 'North (m)', font: { size: 14, weight: 'bold' } },
                        range: yRange,
                        gridcolor: '#e5e5e5'
                    },
                    zaxis: {
                        title: { text: 'Altitude (m)', font: { size: 14, weight: 'bold' } },
                        range: zRange,
                        gridcolor: '#e5e5e5'
                    },
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
            if (!data || !data.altitude) return;
            const units = getUnits();
            const unitType = units === 'metric' ? 'distance' : 'distance';
            const yLabel = 'Altitude (' + getUnitSuffix(unitType) + ')';
            console.log("Altitude data:", {
                length: data.altitude?.length,
                max: Math.max(...(data.altitude || [0])),
                last: data.altitude?.[data.altitude.length - 1]
            });
            // Convert est_pos_z from NED (down positive) to altitude (up positive)
            const est_altitude = data.filter_data?.est_pos_z?.map(z => -z) || null;
            this.render2DChart('chart-altitude', data.time, convertArray(data.altitude, unitType), est_altitude ? convertArray(est_altitude, unitType) : null, 'Altitude vs Time', 'Time (s)', yLabel);
        },
        renderVelocity() {
            const data = this.simulationData;
            const yLabel = 'Velocity (' + getUnitSuffix('velocity') + ')';
            this.render2DChart('chart-velocity', data.time, convertArray(data.velocity, 'velocity'), convertArray(data.filter_data.est_vel_mag, 'velocity'), 'Velocity vs Time', 'Time (s)', yLabel);
        },
        renderAcceleration() {
            const data = this.simulationData;
            const yLabel = 'Acceleration (' + getUnitSuffix('acceleration') + ')';
            this.render2DChart('chart-acceleration', data.time, convertArray(data.acceleration, 'acceleration'), null, 'Acceleration vs Time', 'Time (s)', yLabel);
        },
        renderForce() {
            const data = this.simulationData;
            const yLabel = 'Force (' + getUnitSuffix('force') + ')';
            this.render2DChart('chart-force', data.time, convertArray(data.force, 'force'), null, 'Net Force vs Time', 'Time (s)', yLabel);
        },
        renderMass() {
            const data = this.simulationData;
            const yLabel = 'Mass (' + getUnitSuffix('mass') + ')';
            this.render2DChart('chart-mass', data.time, convertArray(data.mass, 'mass'), null, 'Mass vs Time', 'Time (s)', yLabel);
        },
        render2DChart(elementId, time, trueData, estData, title, xLabel, yLabel) {
            if (!trueData || trueData.length === 0) return;

            const traces = [{
                x: time,
                y: trueData,
                mode: 'lines',
                name: 'Simulated',
                line: { color: '#4CAF50', width: 2 }
            }];

            if (estData && estData.length > 0) {
                traces.push({
                    x: time,
                    y: estData,
                    mode: 'lines',
                    name: 'ESKF',
                    line: { color: '#FF5722', width: 2, dash: 'dash' }
                });
            }

            const simData = this.simulationData;
            const maxVal = Math.max(...trueData.filter(v => isFinite(v)));

            // State change vertical lines
            const shapes = (simData.state_changes_sim || []).map(sc => ({
                type: 'line',
                x0: sc.time,
                x1: sc.time,
                y0: 0,
                y1: maxVal * 1.1,
                line: { color: '#9C27B0', width: 2, dash: 'dash' }
            }));

            // State change annotations
            const annotations = (simData.state_changes_sim || []).map((sc, idx) => ({
                x: sc.time,
                y: maxVal * (0.9 - idx * 0.15),
                text: sc.state || sc.description,
                showarrow: true,
                arrowhead: 2,
                ax: 40,
                ay: 0,
                bgcolor: 'rgba(255,255,255,0.9)',
                bordercolor: '#9C27B0',
                borderwidth: 1
            }));

            // Add apogee annotation if applicable
            // if (apogeeTime !== null) {
            //     annotations.push({
            //         x: apogeeTime,
            //         y: apogeeValue,
            //         text: `Apogee: ${apogeeValue.toFixed(1)}m @ ${apogeeTime.toFixed(1)}s`,
            //         showarrow: true,
            //         arrowhead: 3,
            //         ax: -50,
            //         ay: -40,
            //         bgcolor: 'rgba(255,87,34,0.9)',
            //         bordercolor: '#FF5722',
            //         borderwidth: 2,
            //         font: { color: 'white', weight: 'bold' }
            //     });
            // }
            
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
                { x: data.time, y: data.true_accel_x, name: 'True North', line: { color: '#F44336', width: 2, dash: 'solid' }, opacity: 0.6 },
                { x: data.time, y: sensor.accel_x, name: 'Sensor North', line: { color: '#F44336', width: 1, dash: 'dot' } },
                { x: data.time, y: data.true_accel_y, name: 'True East', line: { color: '#2196F3', width: 2, dash: 'solid' }, opacity: 0.6 },
                { x: data.time, y: sensor.accel_y, name: 'Sensor East', line: { color: '#2196F3', width: 1, dash: 'dot' } },
                { x: data.time, y: data.true_accel_z, name: 'True Down', line: { color: '#4CAF50', width: 2, dash: 'solid' }, opacity: 0.6 },
                { x: data.time, y: sensor.accel_z, name: 'Sensor Down', line: { color: '#4CAF50', width: 1, dash: 'dot' } }
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
                { x: data.time, y: data.true_gyro_x, name: 'True North', line: { color: '#FF9800', width: 2, dash: 'solid' }, opacity: 0.6 },
                { x: data.time, y: sensor.gyro_x, name: 'Sensor North', line: { color: '#FF9800', width: 1, dash: 'dot' } },
                { x: data.time, y: data.true_gyro_y, name: 'True East', line: { color: '#9C27B0', width: 2, dash: 'solid' }, opacity: 0.6 },
                { x: data.time, y: sensor.gyro_y, name: 'Sensor East', line: { color: '#9C27B0', width: 1, dash: 'dot' } },
                { x: data.time, y: data.true_gyro_z, name: 'True Down', line: { color: '#00BCD4', width: 2, dash: 'solid' }, opacity: 0.6 },
                { x: data.time, y: sensor.gyro_z, name: 'Sensor Down', line: { color: '#00BCD4', width: 1, dash: 'dot' } }
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
                { x: data.time, y: sensor.mag_x, name: 'Mag North', line: { color: '#E91E63', width: 1.5 } },
                { x: data.time, y: sensor.mag_y, name: 'Mag East', line: { color: '#673AB7', width: 1.5 } },
                { x: data.time, y: sensor.mag_z, name: 'Mag Down', line: { color: '#3F51B5', width: 1.5 } }
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
        renderGpsPosition() {
            const data = this.simulationData;
            const sensor = data.sensor_data;
            const unitSuffix = getUnitSuffix('distance');

            const traces = [
                { x: data.time, y: convertArray(sensor.gps_x, 'distance'), name: 'GPS North', line: { color: '#009688', width: 1.5 } },
                { x: data.time, y: convertArray(sensor.gps_y, 'distance'), name: 'GPS East', line: { color: '#FF5722', width: 1.5 } },
                { x: data.time, y: convertArray(sensor.gps_z, 'distance'), name: 'GPS Down', line: { color: '#795548', width: 1.5 } }
            ];

            const layout = {
                title: {
                    text: 'GPS Position',
                    font: { size: 14, color: '#1c1917' }
                },
                xaxis: {
                    title: { text: 'Time (s)', font: { size: 12 } },
                    gridcolor: '#e5e5e5'
                },
                yaxis: {
                    title: { text: 'Position (' + unitSuffix + ')', font: { size: 12 } },
                    gridcolor: '#e5e5e5'
                },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                showlegend: true,
                legend: { x: 1, y: 1, xanchor: 'right' },
                margin: { l: 60, r: 40, t: 50, b: 50 }
            };

            Plotly.newPlot('chart-gps-position', traces, layout, {responsive: true});
        },

        renderGpsVelocity() {
            const data = this.simulationData;
            const sensor = data.sensor_data;
            const unitSuffix = getUnitSuffix('velocity');

            const traces = [
                { x: data.time, y: convertArray(sensor.gps_vel_x, 'velocity'), name: 'GPS Vel North', line: { color: '#607D8B', width: 1.5 } },
                { x: data.time, y: convertArray(sensor.gps_vel_y, 'velocity'), name: 'GPS Vel East', line: { color: '#8BC34A', width: 1.5 } },
                { x: data.time, y: convertArray(sensor.gps_vel_z, 'velocity'), name: 'GPS Vel Down', line: { color: '#FFC107', width: 1.5 } }
            ];

            const layout = {
                title: {
                    text: 'GPS Velocity',
                    font: { size: 14, color: '#1c1917' }
                },
                xaxis: {
                    title: { text: 'Time (s)', font: { size: 12 } },
                    gridcolor: '#e5e5e5'
                },
                yaxis: {
                    title: { text: 'Velocity (' + unitSuffix + ')', font: { size: 12 } },
                    gridcolor: '#e5e5e5'
                },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                showlegend: true,
                legend: { x: 1, y: 1, xanchor: 'right' },
                margin: { l: 60, r: 40, t: 50, b: 50 }
            };

            Plotly.newPlot('chart-gps-velocity', traces, layout, {responsive: true});
        },
        renderAdxl() {
            const data = this.simulationData;
            const sensor = data.sensor_data;
            
            // Chart for Y and Z
            const tracesYz = [
                { x: data.time, y: sensor.adxl_y, name: 'ADXL375 East', line: { color: '#2196F3', width: 1.5 } },
                { x: data.time, y: sensor.adxl_z, name: 'ADXL375 Down', line: { color: '#4CAF50', width: 1.5 } }
            ];

            const layoutYz = {
                title: {
                    text: 'ADXL375 Accelerometer (East & Down)',
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
            
            Plotly.newPlot('chart-adxl-yz', tracesYz, layoutYz, {responsive: true});

            // Chart for X
            const tracesX = [
                { x: data.time, y: sensor.adxl_x, name: 'ADXL375 North', line: { color: '#FF5722', width: 1.5 } }
            ];

            const layoutX = {
                title: {
                    text: 'ADXL375 Accelerometer (North)',
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

            Plotly.newPlot('chart-adxl-x', tracesX, layoutX, {responsive: true});
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
        },
        renderErrorPosition() {
            const data = this.simulationData;
            if (!data || !data.filter_data || !data.filter_data.est_pos_x?.length) return;

            // Calculate position errors (estimated - true)
            // data.position_x/y are NED north/east, data.position_z is altitude up
            // data.filter_data.est_pos_x/y/z are all NED (z is down)
            const error_n = data.filter_data.est_pos_x.map((est, i) => est - (data.position_x[i] || 0));
            const error_e = data.filter_data.est_pos_y.map((est, i) => est - (data.position_y[i] || 0));
            // Convert position_z (altitude up) to NED down for consistent comparison
            const error_d = data.filter_data.est_pos_z.map((est, i) => est - (-(data.position_z[i] || 0)));

            const unitSuffix = getUnitSuffix('distance');
            const traces = [
                { x: data.time, y: convertArray(error_n, 'distance'), name: 'North Error (' + unitSuffix + ')', line: { color: '#E91E63', width: 1.5 } },
                { x: data.time, y: convertArray(error_e, 'distance'), name: 'East Error (' + unitSuffix + ')', line: { color: '#9C27B0', width: 1.5 } },
                { x: data.time, y: convertArray(error_d, 'distance'), name: 'Down Error (' + unitSuffix + ')', line: { color: '#3F51B5', width: 1.5 } }
            ];

            const layout = {
                title: {
                    text: 'ESKF Position Error vs Time',
                    font: { size: 14, color: '#1c1917' }
                },
                xaxis: {
                    title: { text: 'Time (s)', font: { size: 12 } },
                    gridcolor: '#e5e5e5'
                },
                yaxis: {
                    title: { text: 'Error (' + unitSuffix + ')', font: { size: 12 } },
                    gridcolor: '#e5e5e5',
                    zeroline: true,
                    zerolinecolor: '#666',
                    zerolinewidth: 1
                },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                showlegend: true,
                legend: { x: 1, y: 1, xanchor: 'right' },
                margin: { l: 60, r: 40, t: 50, b: 50 }
            };

            Plotly.newPlot('chart-error-position', traces, layout, {responsive: true});
        },
        renderErrorVelocity() {
            const data = this.simulationData;
            if (!data || !data.filter_data || !data.filter_data.est_vel_x?.length) return;

            // Calculate velocity errors (estimated - true)
            const error_vn = data.filter_data.est_vel_x.map((est, i) => est - (data.velocity_x[i] || 0));
            const error_ve = data.filter_data.est_vel_y.map((est, i) => est - (data.velocity_y[i] || 0));
            const error_vd = data.filter_data.est_vel_z.map((est, i) => est - (data.velocity_z[i] || 0));

            const unitSuffix = getUnitSuffix('velocity');
            const traces = [
                { x: data.time, y: convertArray(error_vn, 'velocity'), name: 'North Velocity Error (' + unitSuffix + ')', line: { color: '#E91E63', width: 1.5 } },
                { x: data.time, y: convertArray(error_ve, 'velocity'), name: 'East Velocity Error (' + unitSuffix + ')', line: { color: '#9C27B0', width: 1.5 } },
                { x: data.time, y: convertArray(error_vd, 'velocity'), name: 'Down Velocity Error (' + unitSuffix + ')', line: { color: '#3F51B5', width: 1.5 } }
            ];

            const layout = {
                title: {
                    text: 'ESKF Velocity Error vs Time',
                    font: { size: 14, color: '#1c1917' }
                },
                xaxis: {
                    title: { text: 'Time (s)', font: { size: 12 } },
                    gridcolor: '#e5e5e5'
                },
                yaxis: {
                    title: { text: 'Error (' + unitSuffix + ')', font: { size: 12 } },
                    gridcolor: '#e5e5e5',
                    zeroline: true,
                    zerolinecolor: '#666',
                    zerolinewidth: 1
                },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                showlegend: true,
                legend: { x: 1, y: 1, xanchor: 'right' },
                margin: { l: 60, r: 40, t: 50, b: 50 }
            };

            Plotly.newPlot('chart-error-velocity', traces, layout, {responsive: true});
        },
        renderErrorAltitude() {
            const data = this.simulationData;
            if (!data || !data.filter_data || !data.filter_data.est_pos_z?.length) return;

            // Calculate altitude error (estimated - true)
            // data.position_z is already altitude AGL (positive up) from backend
            // data.filter_data.est_pos_z is NED down, need to convert to altitude up
            const true_alt = data.position_z || [];  // Already altitude up
            const est_alt = data.filter_data.est_pos_z.map(z => -z);  // Convert NED down to altitude up
            const error_alt = est_alt.map((est, i) => est - (true_alt[i] || 0));

            const unitSuffix = getUnitSuffix('distance');
            const traces = [
                { x: data.time, y: convertArray(error_alt, 'distance'), name: 'Altitude Error (' + unitSuffix + ')', line: { color: '#FF5722', width: 2 } }
            ];

            const layout = {
                title: {
                    text: 'ESKF Altitude Error vs Time',
                    font: { size: 14, color: '#1c1917' }
                },
                xaxis: {
                    title: { text: 'Time (s)', font: { size: 12 } },
                    gridcolor: '#e5e5e5'
                },
                yaxis: {
                    title: { text: 'Altitude Error (' + unitSuffix + ')', font: { size: 12 } },
                    gridcolor: '#e5e5e5',
                    zeroline: true,
                    zerolinecolor: '#666',
                    zerolinewidth: 2
                },
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                showlegend: true,
                legend: { x: 1, y: 1, xanchor: 'right' },
                margin: { l: 60, r: 40, t: 50, b: 50 }
            };

            Plotly.newPlot('chart-error-altitude', traces, layout, {responsive: true});
        }
    }));
});

// Rocket presets matching presets.rs
const rocketPresets = {
    '30k-ft': {
        dry_mass: 25.0,
        propellant_mass: 15.0,
        thrust: 4000,
        burn_time: 6.0,
        drag_coeff: 0.32,
        ref_area: 0.020,
        cg_full: 1.8,
        cg_empty: 1.6,
        cp_location: 2.5,
        inertia_x: 3.0,
        inertia_y: 3.0,
        inertia_z: 0.015,
        wind_north: 0,
        wind_east: 0
    },
    '15k-ft': {
        dry_mass: 12.0,
        propellant_mass: 8.0,
        thrust: 2000,
        burn_time: 4.5,
        drag_coeff: 0.35,
        ref_area: 0.015,
        cg_full: 1.2,
        cg_empty: 1.1,
        cp_location: 1.6,
        inertia_x: 1.2,
        inertia_y: 1.2,
        inertia_z: 0.006,
        wind_north: 0,
        wind_east: 0
    },
    '12k-ft': {
        dry_mass: 6.0,
        propellant_mass: 2.5,
        thrust: 1200,
        burn_time: 3.5,
        drag_coeff: 0.38,
        ref_area: 0.010,
        cg_full: 1.0,
        cg_empty: 0.9,
        cp_location: 1.4,
        inertia_x: 0.4,
        inertia_y: 0.4,
        inertia_z: 0.004,
        wind_north: 0,
        wind_east: 0
    },
    '10k-ft': {
        dry_mass: 4.0,
        propellant_mass: 2.0,
        thrust: 800,
        burn_time: 3.0,
        drag_coeff: 0.40,
        ref_area: 0.008,
        cg_full: 0.8,
        cg_empty: 0.7,
        cp_location: 1.2,
        inertia_x: 0.25,
        inertia_y: 0.25,
        inertia_z: 0.0025,
        wind_north: 0,
        wind_east: 0
    },
    '5k-ft': {
        dry_mass: 0.5,
        propellant_mass: 0.15,
        thrust: 80,
        burn_time: 2.5,
        drag_coeff: 0.42,
        ref_area: 0.003,
        cg_full: 0.60,
        cg_empty: 0.55,
        cp_location: 0.85,
        inertia_x: 0.01,
        inertia_y: 0.01,
        inertia_z: 0.0001,
        wind_north: 0,
        wind_east: 0
    },
    '3k-ft': {
        dry_mass: 0.3,
        propellant_mass: 0.1,
        thrust: 50,
        burn_time: 2.0,
        drag_coeff: 0.45,
        ref_area: 0.002,
        cg_full: 0.50,
        cg_empty: 0.45,
        cp_location: 0.75,
        inertia_x: 0.006,
        inertia_y: 0.006,
        inertia_z: 0.00006,
        wind_north: 0,
        wind_east: 0
    },
    'high-drag': {
        dry_mass: 2.0,
        propellant_mass: 0.5,
        thrust: 200,
        burn_time: 2.0,
        drag_coeff: 2.0,
        ref_area: 0.02,
        cg_full: 0.8,
        cg_empty: 0.75,
        cp_location: 1.2,
        inertia_x: 0.2,
        inertia_y: 0.2,
        inertia_z: 0.001,
        wind_north: 0,
        wind_east: 0
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