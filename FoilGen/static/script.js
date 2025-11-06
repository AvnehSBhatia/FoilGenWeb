// Global state
let state = {
    re: null,
    mach: null,
    reBin: null,
    maxCl: null,
    maxLd: null,
    alpha: null,
    clPoints: [],
    clCdPoints: [],
    clInterpolated: null,
    clCdInterpolated: null
};

// Optimization cancellation
let currentRequestId = null;
let currentReader = null;

// Initialize charts
let clChart = null;
let clCdChart = null;

// API base URL
const API_BASE = '';

// Utility functions
function showError(message) {
    const errorBox = document.getElementById('error');
    errorBox.textContent = message;
    errorBox.style.display = 'block';
}

function hideError() {
    document.getElementById('error').style.display = 'none';
}

function formatNumber(num) {
    if (Math.abs(num) >= 1e6) return num.toExponential(2);
    if (Math.abs(num) >= 1e3) return num.toFixed(0);
    return num.toFixed(6);
}

// Step 1: Compute Reynolds & Mach
async function computeReynolds() {
    const chord = parseFloat(document.getElementById('chord').value);
    const speed = parseFloat(document.getElementById('speed').value);
    
    if (!chord || !speed) {
        showError('Please enter both chord length and speed');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/compute_reynolds`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chord_ft: chord, speed_mph: speed })
        });
        
        const data = await response.json();
        
        state.re = data.re;
        state.mach = data.mach;
        state.reBin = data.re_bin;
        state.maxCl = data.max_cl;
        state.maxLd = data.max_ld;
        
        document.getElementById('re-value').textContent = formatNumber(data.re);
        document.getElementById('mach-value').textContent = data.mach.toFixed(6);
        document.getElementById('reynolds-result').style.display = 'block';
        hideError();
        
        // Update chart limits if they exist
        updateChartLimits();
    } catch (error) {
        showError('Error computing Reynolds number: ' + error.message);
    }
}

// Step 2: Generate Alpha Vector
async function generateAlpha() {
    const aMin = parseFloat(document.getElementById('alpha-min').value);
    const aMax = parseFloat(document.getElementById('alpha-max').value);
    
    if (aMin >= aMax) {
        showError('Minimum angle must be less than maximum angle');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/generate_alpha`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ a_min: aMin, a_max: aMax })
        });
        
        const data = await response.json();
        
        state.alpha = data.alpha;
        
        document.getElementById('alpha-count').textContent = data.count;
        document.getElementById('alpha-result').style.display = 'block';
        hideError();
        
        // Initialize charts if they don't exist
        if (!clChart) initClChart();
        if (!clCdChart) initClCdChart();
    } catch (error) {
        showError('Error generating alpha vector: ' + error.message);
    }
}

// Helper function to add/update a point, replacing duplicates
function addOrUpdatePoint(pointsArray, alpha, value, tolerance = 0.01) {
    // Check if a point with this alpha already exists (within tolerance)
    const existingIndex = pointsArray.findIndex(p => Math.abs(p[0] - alpha) < tolerance);
    
    if (existingIndex !== -1) {
        // Replace existing point
        pointsArray[existingIndex] = [alpha, value];
        return false; // Indicates replacement
    } else {
        // Add new point
        pointsArray.push([alpha, value]);
        // Sort by alpha to keep points ordered
        pointsArray.sort((a, b) => a[0] - b[0]);
        return true; // Indicates addition
    }
}

// Cl Point Management
function addClPoint() {
    const input = document.getElementById('cl-point-input');
    const value = input.value.trim();
    
    if (!value) return;
    
    const parts = value.split(',');
    if (parts.length !== 2) {
        showError('Please enter points as "alpha,value" (e.g., "5.0,1.2")');
        return;
    }
    
    const alpha = parseFloat(parts[0].trim());
    const clValue = parseFloat(parts[1].trim());
    
    if (isNaN(alpha) || isNaN(clValue)) {
        showError('Please enter valid numbers');
        return;
    }
    
    const wasNew = addOrUpdatePoint(state.clPoints, alpha, clValue);
    input.value = '';
    updateClPointsList();
    hideError();
}

function removeClPoint(index) {
    state.clPoints.splice(index, 1);
    updateClPointsList();
}

function clearClPoints() {
    state.clPoints = [];
    updateClPointsList();
    if (clChart) {
        Plotly.deleteTraces('cl-chart', 0);
        Plotly.deleteTraces('cl-chart', 0);
    }
}

function updateClPointsList() {
    const list = document.getElementById('cl-points-list');
    list.innerHTML = '';
    
    state.clPoints.forEach((point, index) => {
        const badge = document.createElement('div');
        badge.className = 'point-badge';
        badge.innerHTML = `
            <span>α=${point[0].toFixed(1)}°, Cl=${point[1].toFixed(3)}</span>
            <button class="remove-btn" onclick="removeClPoint(${index})">×</button>
        `;
        list.appendChild(badge);
    });
}

// Cl/Cd Point Management
function addClCdPoint() {
    const input = document.getElementById('clcd-point-input');
    const value = input.value.trim();
    
    if (!value) return;
    
    const parts = value.split(',');
    if (parts.length !== 2) {
        showError('Please enter points as "alpha,value" (e.g., "5.0,100.0")');
        return;
    }
    
    const alpha = parseFloat(parts[0].trim());
    const clCdValue = parseFloat(parts[1].trim());
    
    if (isNaN(alpha) || isNaN(clCdValue)) {
        showError('Please enter valid numbers');
        return;
    }
    
    addOrUpdatePoint(state.clCdPoints, alpha, clCdValue);
    input.value = '';
    updateClCdPointsList();
    hideError();
}

function removeClCdPoint(index) {
    state.clCdPoints.splice(index, 1);
    updateClCdPointsList();
}

function clearClCdPoints() {
    state.clCdPoints = [];
    updateClCdPointsList();
    if (clCdChart) {
        Plotly.deleteTraces('clcd-chart', 0);
        Plotly.deleteTraces('clcd-chart', 0);
    }
}

function updateClCdPointsList() {
    const list = document.getElementById('clcd-points-list');
    list.innerHTML = '';
    
    state.clCdPoints.forEach((point, index) => {
        const badge = document.createElement('div');
        badge.className = 'point-badge';
        badge.innerHTML = `
            <span>α=${point[0].toFixed(1)}°, Cl/Cd=${point[1].toFixed(1)}</span>
            <button class="remove-btn" onclick="removeClCdPoint(${index})">×</button>
        `;
        list.appendChild(badge);
    });
}

// Initialize Charts
function initClChart() {
    const layout = {
        title: 'Cl vs Angle of Attack (Click to add points)',
        xaxis: { title: 'Angle of Attack (deg)' },
        yaxis: { title: 'Cl', range: state.maxCl ? [0, state.maxCl * 1.15] : null },
        hovermode: 'closest',
        showlegend: true,
        height: 400
    };
    
    const gd = document.getElementById('cl-chart');
    
    Plotly.newPlot('cl-chart', [], layout, {responsive: true}).then(function() {
        // Add click handler for Cl chart - handle clicks anywhere on plot
        gd.on('plotly_click', function(data) {
            if (data.points && data.points.length > 0) {
                const point = data.points[0];
                const alpha = point.x;
                const value = point.y;
                
                addOrUpdatePoint(state.clPoints, alpha, value);
                updateClPointsList();
                
                if (state.alpha) {
                    interpolateCl();
                }
            }
        });
        
        // Add mouse event for clicking anywhere on the plot
        gd.addEventListener('click', function(event) {
            // Skip if this is a plotly_click event (already handled above)
            if (event.target.classList.contains('point')) return;
            
            const bbox = gd.getBoundingClientRect();
            const xPixel = event.clientX - bbox.left;
            const yPixel = event.clientY - bbox.top;
            
            // Get plot layout
            const xaxis = gd._fullLayout.xaxis;
            const yaxis = gd._fullLayout.yaxis;
            
            if (!xaxis || !yaxis) return;
            
            // Convert pixel to data coordinates
            const xData = (xPixel - xaxis._offset) / xaxis._length * (xaxis.range[1] - xaxis.range[0]) + xaxis.range[0];
            const yData = (1 - (yPixel - yaxis._offset) / yaxis._length) * (yaxis.range[1] - yaxis.range[0]) + yaxis.range[0];
            
            // Only add if click is within plot area (not on axes/legend)
            if (xPixel >= xaxis._offset && xPixel <= xaxis._offset + xaxis._length &&
                yPixel >= yaxis._offset && yPixel <= yaxis._offset + yaxis._length) {
                addOrUpdatePoint(state.clPoints, xData, yData);
                updateClPointsList();
                
                if (state.alpha) {
                    interpolateCl();
                }
            }
        });
    });
}

function initClCdChart() {
    const layout = {
        title: 'Cl/Cd vs Angle of Attack (Click to add points)',
        xaxis: { title: 'Angle of Attack (deg)' },
        yaxis: { title: 'Cl/Cd', range: state.maxLd ? [0, state.maxLd * 1.15] : null },
        hovermode: 'closest',
        showlegend: true,
        height: 400
    };
    
    const gd = document.getElementById('clcd-chart');
    
    Plotly.newPlot('clcd-chart', [], layout, {responsive: true}).then(function() {
        // Add click handler for Cl/Cd chart - handle clicks anywhere on plot
        gd.on('plotly_click', function(data) {
            if (data.points && data.points.length > 0) {
                const point = data.points[0];
                const alpha = point.x;
                const value = point.y;
                
                addOrUpdatePoint(state.clCdPoints, alpha, value);
                updateClCdPointsList();
                
                if (state.alpha) {
                    interpolateClCd();
                }
            }
        });
        
        // Add mouse event for clicking anywhere on the plot
        gd.addEventListener('click', function(event) {
            // Skip if this is a plotly_click event (already handled above)
            if (event.target.classList.contains('point')) return;
            
            const bbox = gd.getBoundingClientRect();
            const xPixel = event.clientX - bbox.left;
            const yPixel = event.clientY - bbox.top;
            
            // Get plot layout
            const xaxis = gd._fullLayout.xaxis;
            const yaxis = gd._fullLayout.yaxis;
            
            if (!xaxis || !yaxis) return;
            
            // Convert pixel to data coordinates
            const xData = (xPixel - xaxis._offset) / xaxis._length * (xaxis.range[1] - xaxis.range[0]) + xaxis.range[0];
            const yData = (1 - (yPixel - yaxis._offset) / yaxis._length) * (yaxis.range[1] - yaxis.range[0]) + yaxis.range[0];
            
            // Only add if click is within plot area (not on axes/legend)
            if (xPixel >= xaxis._offset && xPixel <= xaxis._offset + xaxis._length &&
                yPixel >= yaxis._offset && yPixel <= yaxis._offset + yaxis._length) {
                addOrUpdatePoint(state.clCdPoints, xData, yData);
                updateClCdPointsList();
                
                if (state.alpha) {
                    interpolateClCd();
                }
            }
        });
    });
}

function updateChartLimits() {
    if (clChart && state.maxCl) {
        Plotly.relayout('cl-chart', {'yaxis.range': [0, state.maxCl * 1.15]});
    }
    if (clCdChart && state.maxLd) {
        Plotly.relayout('clcd-chart', {'yaxis.range': [0, state.maxLd * 1.15]});
    }
}

// Interpolate Cl
async function interpolateCl() {
    if (!state.alpha) {
        showError('Please generate alpha vector first');
        return;
    }
    
    if (state.clPoints.length === 0) {
        showError('Please add at least one Cl point');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/interpolate_points`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                control_points: state.clPoints,
                alpha_vector: state.alpha
            })
        });
        
        const data = await response.json();
        state.clInterpolated = data.interpolated;
        
        // Update chart
        const controlAlphas = state.clPoints.map(p => p[0]);
        const controlValues = state.clPoints.map(p => p[1]);
        
        const trace1 = {
            x: controlAlphas,
            y: controlValues,
            mode: 'markers',
            type: 'scatter',
            name: 'Control Points',
            marker: { size: 10, color: 'blue' }
        };
        
        const trace2 = {
            x: state.alpha,
            y: state.clInterpolated,
            mode: 'lines',
            type: 'scatter',
            name: 'Spline Interpolation',
            line: { color: 'green', width: 2 }
        };
        
        const gd = document.getElementById('cl-chart');
        
        Plotly.newPlot('cl-chart', [trace1, trace2], {
            title: 'Cl vs Angle of Attack (Click to add points)',
            xaxis: { title: 'Angle of Attack (deg)' },
            yaxis: { title: 'Cl', range: state.maxCl ? [0, state.maxCl * 1.15] : null },
            hovermode: 'closest',
            showlegend: true,
            height: 400
        }, {responsive: true}).then(function() {
            // Re-add click handlers after plot update
            gd.on('plotly_click', function(data) {
                if (data.points && data.points.length > 0) {
                    const point = data.points[0];
                    const alpha = point.x;
                    const value = point.y;
                    
                    addOrUpdatePoint(state.clPoints, alpha, value);
                    updateClPointsList();
                    
                    if (state.alpha) {
                        interpolateCl();
                    }
                }
            });
            
            // Re-add click handler for clicking anywhere
            gd.addEventListener('click', function(event) {
                if (event.target.classList.contains('point')) return;
                
                const bbox = gd.getBoundingClientRect();
                const xPixel = event.clientX - bbox.left;
                const yPixel = event.clientY - bbox.top;
                
                const xaxis = gd._fullLayout.xaxis;
                const yaxis = gd._fullLayout.yaxis;
                
                if (!xaxis || !yaxis) return;
                
                const xData = (xPixel - xaxis._offset) / xaxis._length * (xaxis.range[1] - xaxis.range[0]) + xaxis.range[0];
                const yData = (1 - (yPixel - yaxis._offset) / yaxis._length) * (yaxis.range[1] - yaxis.range[0]) + yaxis.range[0];
                
                if (xPixel >= xaxis._offset && xPixel <= xaxis._offset + xaxis._length &&
                    yPixel >= yaxis._offset && yPixel <= yaxis._offset + yaxis._length) {
                    addOrUpdatePoint(state.clPoints, xData, yData);
                    updateClPointsList();
                    
                    if (state.alpha) {
                        interpolateCl();
                    }
                }
            });
        });
        
        hideError();
    } catch (error) {
        showError('Error interpolating Cl: ' + error.message);
    }
}

// Interpolate Cl/Cd
async function interpolateClCd() {
    if (!state.alpha) {
        showError('Please generate alpha vector first');
        return;
    }
    
    if (state.clCdPoints.length === 0) {
        showError('Please add at least one Cl/Cd point');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/interpolate_points`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                control_points: state.clCdPoints,
                alpha_vector: state.alpha
            })
        });
        
        const data = await response.json();
        state.clCdInterpolated = data.interpolated;
        
        // Update chart
        const controlAlphas = state.clCdPoints.map(p => p[0]);
        const controlValues = state.clCdPoints.map(p => p[1]);
        
        const trace1 = {
            x: controlAlphas,
            y: controlValues,
            mode: 'markers',
            type: 'scatter',
            name: 'Control Points',
            marker: { size: 10, color: 'blue' }
        };
        
        const trace2 = {
            x: state.alpha,
            y: state.clCdInterpolated,
            mode: 'lines',
            type: 'scatter',
            name: 'Spline Interpolation',
            line: { color: 'green', width: 2 }
        };
        
        const gd = document.getElementById('clcd-chart');
        
        Plotly.newPlot('clcd-chart', [trace1, trace2], {
            title: 'Cl/Cd vs Angle of Attack (Click to add points)',
            xaxis: { title: 'Angle of Attack (deg)' },
            yaxis: { title: 'Cl/Cd', range: state.maxLd ? [0, state.maxLd * 1.15] : null },
            hovermode: 'closest',
            showlegend: true,
            height: 400
        }, {responsive: true}).then(function() {
            // Re-add click handlers after plot update
            gd.on('plotly_click', function(data) {
                if (data.points && data.points.length > 0) {
                    const point = data.points[0];
                    const alpha = point.x;
                    const value = point.y;
                    
                    addOrUpdatePoint(state.clCdPoints, alpha, value);
                    updateClCdPointsList();
                    
                    if (state.alpha) {
                        interpolateClCd();
                    }
                }
            });
            
            // Re-add click handler for clicking anywhere
            gd.addEventListener('click', function(event) {
                if (event.target.classList.contains('point')) return;
                
                const bbox = gd.getBoundingClientRect();
                const xPixel = event.clientX - bbox.left;
                const yPixel = event.clientY - bbox.top;
                
                const xaxis = gd._fullLayout.xaxis;
                const yaxis = gd._fullLayout.yaxis;
                
                if (!xaxis || !yaxis) return;
                
                const xData = (xPixel - xaxis._offset) / xaxis._length * (xaxis.range[1] - xaxis.range[0]) + xaxis.range[0];
                const yData = (1 - (yPixel - yaxis._offset) / yaxis._length) * (yaxis.range[1] - yaxis.range[0]) + yaxis.range[0];
                
                if (xPixel >= xaxis._offset && xPixel <= xaxis._offset + xaxis._length &&
                    yPixel >= yaxis._offset && yPixel <= yaxis._offset + yaxis._length) {
                    addOrUpdatePoint(state.clCdPoints, xData, yData);
                    updateClCdPointsList();
                    
                    if (state.alpha) {
                        interpolateClCd();
                    }
                }
            });
        });
        
        hideError();
    } catch (error) {
        showError('Error interpolating Cl/Cd: ' + error.message);
    }
}

// Generate Airfoil
async function generateAirfoil() {
    // Validation
    if (!state.re || !state.mach) {
        showError('Please compute Reynolds and Mach numbers first');
        return;
    }
    
    if (!state.alpha) {
        showError('Please generate alpha vector first');
        return;
    }
    
    if (!state.clInterpolated || state.clInterpolated.length === 0) {
        showError('Please interpolate Cl curve first');
        return;
    }
    
    if (!state.clCdInterpolated || state.clCdInterpolated.length === 0) {
        showError('Please interpolate Cl/Cd curve first');
        return;
    }
    
    const minThickness = parseFloat(document.getElementById('min-thickness').value);
    const maxThickness = parseFloat(document.getElementById('max-thickness').value);
    
    if (!minThickness || !maxThickness || minThickness >= maxThickness) {
        showError('Please enter valid thickness values (minimum must be less than maximum)');
        return;
    }
    
    // Disable generate button
    const generateBtn = document.getElementById('generate-btn');
    generateBtn.disabled = true;
    generateBtn.textContent = 'Optimizing...';
    
    // Show optimization status, hide results
    document.getElementById('optimization-status').style.display = 'block';
    document.getElementById('loading').style.display = 'none';
    document.getElementById('error').style.display = 'none';
    document.getElementById('results').style.display = 'none';
    
    // Reset progress
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('progress-text').textContent = '0%';
    document.getElementById('progress-details-text').textContent = 'Initializing...';
    document.getElementById('best-airfoil-preview-img').style.display = 'none';
    document.getElementById('best-airfoil-stats').style.display = 'none';
    
    try {
        const response = await fetch(`${API_BASE}/api/optimize_airfoil`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                re: state.re,
                mach: state.mach,
                alpha: state.alpha,
                cl: state.clInterpolated,
                cl_cd: state.clCdInterpolated,
                min_thickness: minThickness,
                max_thickness: maxThickness
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const reader = response.body.getReader();
        currentReader = reader;
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.type === 'init') {
                            // Store request ID for cancellation
                            currentRequestId = data.request_id;
                        } else if (data.type === 'cancelled') {
                            // Optimization was cancelled
                            document.getElementById('progress-details-text').textContent = 'Optimization cancelled';
                            document.getElementById('generate-btn').disabled = false;
                            document.getElementById('generate-btn').textContent = 'Generate Airfoil';
                            document.getElementById('optimization-status').style.display = 'none';
                            currentRequestId = null;
                            currentReader = null;
                            return;
                        } else if (data.type === 'progress') {
                            // Update progress bar
                            const progress = Math.min(100, Math.max(0, data.progress));
                            document.getElementById('progress-fill').style.width = progress + '%';
                            document.getElementById('progress-text').textContent = progress.toFixed(1) + '%';
                            document.getElementById('progress-details-text').textContent = 
                                `Iteration ${data.iteration} of ${data.total} | Testing: Max=${data.current_max_t.toFixed(3)}, Min=${data.current_min_t.toFixed(3)} | Best Error: ${data.best_error.toFixed(6)}`;
                            
                            // Update best airfoil preview if new best found
                            if (data.best_airfoil_plot) {
                                const previewImg = document.getElementById('best-airfoil-preview-img');
                                previewImg.src = 'data:image/png;base64,' + data.best_airfoil_plot;
                                previewImg.style.display = 'block';
                                
                                const statsDiv = document.getElementById('best-airfoil-stats');
                                document.getElementById('best-max-thickness').textContent = data.best_max_t.toFixed(3);
                                document.getElementById('best-min-thickness').textContent = data.best_min_t.toFixed(3);
                                document.getElementById('best-error').textContent = data.best_error.toFixed(6);
                                statsDiv.style.display = 'block';
                            }
                        } else if (data.type === 'complete') {
                            if (data.success) {
                                // Show loading for final processing
                                document.getElementById('loading').style.display = 'flex';
                                
                                // Display final results - check if uncorrected data is available
                                const hasUncorrected = data.uncorrected && data.uncorrected.stats;
                                
                                if (hasUncorrected) {
                                    // Show comparison view
                                    document.getElementById('correction-comparison').style.display = 'block';
                                    document.getElementById('regular-stats').style.display = 'none';
                                    
                                    // Uncorrected stats
                                    const ucStats = data.uncorrected.stats;
                                    document.getElementById('stat-clmax-uncorrected').textContent = formatNumber(ucStats.clmax);
                                    document.getElementById('stat-clmax-alpha-uncorrected').textContent = `at α = ${ucStats.alpha_clmax.toFixed(2)}°`;
                                    document.getElementById('stat-cdmin-uncorrected').textContent = formatNumber(ucStats.cdmin);
                                    document.getElementById('stat-cdmin-alpha-uncorrected').textContent = `at α = ${ucStats.alpha_cdmin.toFixed(2)}°`;
                                    document.getElementById('stat-ldmax-uncorrected').textContent = formatNumber(ucStats.ldmax);
                                    document.getElementById('stat-ldmax-alpha-uncorrected').textContent = `at α = ${ucStats.alpha_ldmax.toFixed(2)}°`;
                                    
                                    // Corrected stats
                                    document.getElementById('stat-clmax-corrected').textContent = formatNumber(data.stats.clmax);
                                    document.getElementById('stat-clmax-alpha-corrected').textContent = `at α = ${data.stats.alpha_clmax.toFixed(2)}°`;
                                    document.getElementById('stat-cdmin-corrected').textContent = formatNumber(data.stats.cdmin);
                                    document.getElementById('stat-cdmin-alpha-corrected').textContent = `at α = ${data.stats.alpha_cdmin.toFixed(2)}°`;
                                    document.getElementById('stat-ldmax-corrected').textContent = formatNumber(data.stats.ldmax);
                                    document.getElementById('stat-ldmax-alpha-corrected').textContent = `at α = ${data.stats.alpha_ldmax.toFixed(2)}°`;
                                } else {
                                    // Show regular stats view
                                    document.getElementById('correction-comparison').style.display = 'none';
                                    document.getElementById('regular-stats').style.display = 'grid';
                                    
                                    if (data.stats && Object.keys(data.stats).length > 0) {
        document.getElementById('stat-clmax').textContent = formatNumber(data.stats.clmax);
        document.getElementById('stat-clmax-alpha').textContent = `at α = ${data.stats.alpha_clmax.toFixed(2)}°`;
        document.getElementById('stat-cdmin').textContent = formatNumber(data.stats.cdmin);
        document.getElementById('stat-cdmin-alpha').textContent = `at α = ${data.stats.alpha_cdmin.toFixed(2)}°`;
        document.getElementById('stat-ldmax').textContent = formatNumber(data.stats.ldmax);
        document.getElementById('stat-ldmax-alpha').textContent = `at α = ${data.stats.alpha_ldmax.toFixed(2)}°`;
                                    }
                                }
        
        document.getElementById('airfoil-plot').src = 'data:image/png;base64,' + data.plots.airfoil_shape;
        document.getElementById('curves-plot').src = 'data:image/png;base64,' + data.plots.aerodynamic_curves;
        
        document.getElementById('results').style.display = 'block';
        document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
                                
                                // Check for Hall of Fame notification
                                const resultsSection = document.getElementById('results');
                                const existingNotification = resultsSection.querySelector('.hof-notification');
                                if (existingNotification) {
                                    existingNotification.remove();
                                }
                                
                                if (data.hof_added_ld || data.hof_added_cl) {
                                    const notification = document.createElement('div');
                                    notification.className = 'hof-notification';
                                    let message = '🎉 Congratulations! This airfoil has been added to the ';
                                    if (data.hof_added_ld && data.hof_added_cl) {
                                        message += 'Hall of Fame for both Peak L/D and Peak Cl!';
                                    } else if (data.hof_added_ld) {
                                        message += 'Hall of Fame for Peak L/D!';
                                    } else {
                                        message += 'Hall of Fame for Peak Cl!';
                                    }
                                    message += ' <a href="/hof">View Hall of Fame</a>';
                                    notification.innerHTML = message;
                                    resultsSection.insertBefore(notification, resultsSection.firstChild);
                                }
                                
                                // Hide optimization status
                                document.getElementById('optimization-status').style.display = 'none';
                                document.getElementById('loading').style.display = 'none';
                            } else {
                                throw new Error(data.error || 'Optimization failed');
                            }
                        } else if (data.type === 'error') {
                            throw new Error(data.error);
                        }
                    } catch (e) {
                        console.error('Error parsing SSE data:', e, line);
                    }
                }
            }
        }
        
        hideError();
    } catch (error) {
        showError('Error optimizing airfoil: ' + error.message);
        document.getElementById('optimization-status').style.display = 'none';
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = 'Generate Airfoil';
        currentRequestId = null;
        currentReader = null;
    }
}

// Cancel optimization
async function cancelOptimization() {
    if (currentRequestId) {
        try {
            await fetch(`${API_BASE}/api/cancel_optimization`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ request_id: currentRequestId })
            });
            
            // Cancel the reader
            if (currentReader) {
                currentReader.cancel();
            }
            
            document.getElementById('progress-details-text').textContent = 'Cancelling...';
        } catch (error) {
            console.error('Error cancelling optimization:', error);
        }
    }
}

// Export configuration
function exportConfig() {
    const config = {
        chord_ft: parseFloat(document.getElementById('chord').value) || 0,
        speed_mph: parseFloat(document.getElementById('speed').value) || 0,
        re: state.re || 0,
        mach: state.mach || 0,
        alpha_min: parseFloat(document.getElementById('alpha-min').value) || 0,
        alpha_max: parseFloat(document.getElementById('alpha-max').value) || 0,
        alpha: state.alpha || [],
        cl_points: state.clPoints || [],
        clcd_points: state.clCdPoints || [],
        cl_interpolated: state.clInterpolated || [],
        clcd_interpolated: state.clCdInterpolated || [],
        min_thickness: parseFloat(document.getElementById('min-thickness').value) || 0.05,
        max_thickness: parseFloat(document.getElementById('max-thickness').value) || 0.15
    };
    
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'airfoil_config.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Import configuration
async function importConfig(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    try {
        const text = await file.text();
        const config = JSON.parse(text);
        
        // Load configuration into UI
        if (config.chord_ft) document.getElementById('chord').value = config.chord_ft;
        if (config.speed_mph) document.getElementById('speed').value = config.speed_mph;
        if (config.alpha_min) document.getElementById('alpha-min').value = config.alpha_min;
        if (config.alpha_max) document.getElementById('alpha-max').value = config.alpha_max;
        if (config.min_thickness) document.getElementById('min-thickness').value = config.min_thickness;
        if (config.max_thickness) document.getElementById('max-thickness').value = config.max_thickness;
        // Handle legacy config format (backward compatibility)
        if (config.max_thickness_min && !config.min_thickness) {
            document.getElementById('min-thickness').value = config.max_thickness_min;
        }
        if (config.max_thickness_max && !config.max_thickness) {
            document.getElementById('max-thickness').value = config.max_thickness_max;
        }
        
        // Load state
        if (config.re) state.re = config.re;
        if (config.mach) state.mach = config.mach;
        if (config.alpha) state.alpha = config.alpha;
        if (config.cl_points) state.clPoints = config.cl_points;
        if (config.clcd_points) state.clCdPoints = config.clcd_points;
        if (config.cl_interpolated) state.clInterpolated = config.cl_interpolated;
        if (config.clcd_interpolated) state.clCdInterpolated = config.clcd_interpolated;
        
        // Update UI
        if (state.re && state.mach) {
            document.getElementById('re-value').textContent = formatNumber(state.re);
            document.getElementById('mach-value').textContent = state.mach.toFixed(6);
            document.getElementById('reynolds-result').style.display = 'block';
        }
        
        if (state.alpha) {
            document.getElementById('alpha-count').textContent = state.alpha.length;
            document.getElementById('alpha-result').style.display = 'block';
        }
        
        // Update charts
        if (state.clPoints.length > 0) {
            updateClPointsList();
            if (state.alpha && state.clInterpolated) {
                interpolateCl();
            }
        }
        
        if (state.clCdPoints.length > 0) {
            updateClCdPointsList();
            if (state.alpha && state.clCdInterpolated) {
                interpolateClCd();
            }
        }
        
        // Reset file input
        event.target.value = '';
        hideError();
    } catch (error) {
        showError('Error importing configuration: ' + error.message);
    }
}

// Settings modal functions
function toggleSettings() {
    const modal = document.getElementById('settings-modal');
    modal.style.display = modal.style.display === 'none' ? 'block' : 'none';
    
    // Load saved colors
    loadColors();
}

function closeSettings() {
    document.getElementById('settings-modal').style.display = 'none';
}

// Load colors from localStorage
function loadColors() {
    const colors = {
        primary: localStorage.getItem('color-primary') || '#2563eb',
        success: localStorage.getItem('color-success') || '#10b981',
        secondary: localStorage.getItem('color-secondary') || '#6b7280',
        background: localStorage.getItem('color-background') || '#f9fafb',
        card: localStorage.getItem('color-card') || '#ffffff'
    };
    
    document.getElementById('color-primary').value = colors.primary;
    document.getElementById('color-success').value = colors.success;
    document.getElementById('color-secondary').value = colors.secondary;
    document.getElementById('color-background').value = colors.background;
    document.getElementById('color-card').value = colors.card;
    
    updateColors();
}

// Helper function to create gradient end color (slightly darker/more saturated)
function createGradientEnd(color) {
    // Convert hex to RGB
    const num = parseInt(color.replace('#', ''), 16);
    const r = (num >> 16) & 0xff;
    const g = (num >> 8) & 0xff;
    const b = num & 0xff;
    
    // Create a darker, more saturated version for gradient end
    const darkerR = Math.max(0, Math.min(255, r * 0.7));
    const darkerG = Math.max(0, Math.min(255, g * 0.7));
    const darkerB = Math.max(0, Math.min(255, b * 0.7));
    
    return '#' + [darkerR, darkerG, darkerB].map(x => {
        const hex = Math.round(x).toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    }).join('');
}

// Update colors
function updateColors() {
    const primary = document.getElementById('color-primary').value;
    const success = document.getElementById('color-success').value;
    const secondary = document.getElementById('color-secondary').value;
    const background = document.getElementById('color-background').value;
    const card = document.getElementById('color-card').value;
    
    // Save to localStorage
    localStorage.setItem('color-primary', primary);
    localStorage.setItem('color-success', success);
    localStorage.setItem('color-secondary', secondary);
    localStorage.setItem('color-background', background);
    localStorage.setItem('color-card', card);
    
    // Update CSS variables
    document.documentElement.style.setProperty('--primary-color', primary);
    document.documentElement.style.setProperty('--success-color', success);
    document.documentElement.style.setProperty('--secondary-color', secondary);
    document.documentElement.style.setProperty('--background', background);
    document.documentElement.style.setProperty('--card-background', card);
    
    // Update hover colors (slightly darker)
    const primaryHover = darkenColor(primary, 0.1);
    const successHover = darkenColor(success, 0.1);
    document.documentElement.style.setProperty('--primary-hover', primaryHover);
    document.documentElement.style.setProperty('--success-hover', successHover);
    
    // Update stat card gradients based on user colors
    // Card 1: Primary color gradient
    document.documentElement.style.setProperty('--stat-card-1-start', primary);
    document.documentElement.style.setProperty('--stat-card-1-end', createGradientEnd(primary));
    
    // Card 2: Success color gradient
    document.documentElement.style.setProperty('--stat-card-2-start', success);
    document.documentElement.style.setProperty('--stat-card-2-end', createGradientEnd(success));
    
    // Card 3: Blend of primary and success for variety
    const card3Start = blendColors(primary, success, 0.7);
    document.documentElement.style.setProperty('--stat-card-3-start', card3Start);
    document.documentElement.style.setProperty('--stat-card-3-end', createGradientEnd(card3Start));
}

// Helper function to blend two colors
function blendColors(color1, color2, ratio) {
    const num1 = parseInt(color1.replace('#', ''), 16);
    const num2 = parseInt(color2.replace('#', ''), 16);
    
    const r1 = (num1 >> 16) & 0xff;
    const g1 = (num1 >> 8) & 0xff;
    const b1 = num1 & 0xff;
    
    const r2 = (num2 >> 16) & 0xff;
    const g2 = (num2 >> 8) & 0xff;
    const b2 = num2 & 0xff;
    
    const r = Math.round(r1 * ratio + r2 * (1 - ratio));
    const g = Math.round(g1 * ratio + g2 * (1 - ratio));
    const b = Math.round(b1 * ratio + b2 * (1 - ratio));
    
    return '#' + [r, g, b].map(x => {
        const hex = x.toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    }).join('');
}

// Reset colors to default
function resetColors() {
    localStorage.removeItem('color-primary');
    localStorage.removeItem('color-success');
    localStorage.removeItem('color-secondary');
    localStorage.removeItem('color-background');
    localStorage.removeItem('color-card');
    loadColors();
}

// Helper function to darken color
function darkenColor(color, amount) {
    const num = parseInt(color.replace('#', ''), 16);
    const r = Math.max(0, Math.min(255, ((num >> 16) & 0xff) * (1 - amount)));
    const g = Math.max(0, Math.min(255, ((num >> 8) & 0xff) * (1 - amount)));
    const b = Math.max(0, Math.min(255, (num & 0xff) * (1 - amount)));
    return '#' + [r, g, b].map(x => {
        const hex = Math.round(x).toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    }).join('');
}

// Download .dat file
function downloadDat() {
    window.open(`${API_BASE}/api/download_dat`, '_blank');
}

// Enter key handlers and initialization
document.addEventListener('DOMContentLoaded', function() {
    // Load colors on page load
    loadColors();
    
    // Close settings modal when clicking outside
    window.onclick = function(event) {
        const settingsModal = document.getElementById('settings-modal');
        if (event.target === settingsModal) {
            closeSettings();
        }
    };
    
    // Enter key handlers
    document.getElementById('cl-point-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            addClPoint();
        }
    });
    
    document.getElementById('clcd-point-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            addClCdPoint();
        }
    });
});
