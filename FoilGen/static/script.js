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
    
    const maxThicknessMin = parseFloat(document.getElementById('max-thickness-min').value);
    const maxThicknessMax = parseFloat(document.getElementById('max-thickness-max').value);
    
    if (!maxThicknessMin || !maxThicknessMax || maxThicknessMin >= maxThicknessMax) {
        showError('Please enter valid max thickness range (min < max)');
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
                max_thickness_min: maxThicknessMin,
                max_thickness_max: maxThicknessMax
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const reader = response.body.getReader();
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
                        
                        if (data.type === 'progress') {
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
                                
                                // Display final results
                                if (data.stats && Object.keys(data.stats).length > 0) {
                                    document.getElementById('stat-clmax').textContent = formatNumber(data.stats.clmax);
                                    document.getElementById('stat-clmax-alpha').textContent = `at α = ${data.stats.alpha_clmax.toFixed(2)}°`;
                                    document.getElementById('stat-cdmin').textContent = formatNumber(data.stats.cdmin);
                                    document.getElementById('stat-cdmin-alpha').textContent = `at α = ${data.stats.alpha_cdmin.toFixed(2)}°`;
                                    document.getElementById('stat-ldmax').textContent = formatNumber(data.stats.ldmax);
                                    document.getElementById('stat-ldmax-alpha').textContent = `at α = ${data.stats.alpha_ldmax.toFixed(2)}°`;
                                }
                                
                                document.getElementById('airfoil-plot').src = 'data:image/png;base64,' + data.plots.airfoil_shape;
                                document.getElementById('curves-plot').src = 'data:image/png;base64,' + data.plots.aerodynamic_curves;
                                
                                document.getElementById('results').style.display = 'block';
                                document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
                                
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
    }
}

// Download .dat file
function downloadDat() {
    window.open(`${API_BASE}/api/download_dat`, '_blank');
}

// Enter key handlers
document.addEventListener('DOMContentLoaded', function() {
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
