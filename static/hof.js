// Hall of Fame JavaScript

const API_BASE = '';
let currentCategory = 'peak_ld';
let currentReBin = 50000;
let currentEntries = [];

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // Load colors on page load
    loadColors();
    
    selectCategory('peak_ld');
    selectReBin(50000);
    
    // Close settings modal when clicking outside
    window.addEventListener('click', function(event) {
        const settingsModal = document.getElementById('settings-modal');
        if (event.target === settingsModal) {
            closeSettings();
        }
    });
});

// Category selection
function selectCategory(category) {
    currentCategory = category;
    
    // Update button styles
    document.getElementById('category-peak-ld').className = 
        category === 'peak_ld' ? 'btn btn-primary' : 'btn btn-secondary';
    document.getElementById('category-peak-cl').className = 
        category === 'peak_cl' ? 'btn btn-primary' : 'btn btn-secondary';
    
    loadAirfoils();
}

// Re bin selection
function selectReBin(reBin) {
    currentReBin = reBin;
    
    // Update button styles
    const buttons = document.querySelectorAll('.re-bin-selector button');
    buttons.forEach(btn => {
        const btnReBin = parseInt(btn.textContent.replace('k', '000').replace('m+', '000000'));
        if (btnReBin === reBin || (reBin === 1000000 && btn.textContent === '1m+')) {
            btn.className = 'btn btn-sm btn-primary';
        } else {
            btn.className = 'btn btn-sm';
        }
    });
    
    loadAirfoils();
}

// Load airfoils for current category and Re bin
async function loadAirfoils() {
    const grid = document.getElementById('airfoil-grid');
    grid.innerHTML = '<div class="loading"><div class="spinner"></div><p>Loading airfoils...</p></div>';
    
    // Update title
    const categoryName = currentCategory === 'peak_ld' ? 'Peak L/D' : 'Peak Cl';
    const reBinLabel = formatReBin(currentReBin);
    document.getElementById('grid-title').textContent = `Top 10 Airfoils - ${categoryName} at Re = ${reBinLabel}`;
    
    try {
        const response = await fetch(`${API_BASE}/api/hof/${currentReBin}/${currentCategory}`);
        const entries = await response.json();
        currentEntries = entries;
        
        if (entries.length === 0) {
            grid.innerHTML = '<div style="text-align: center; padding: 40px; color: var(--text-secondary);">No airfoils in Hall of Fame yet for this category.</div>';
            return;
        }
        
        // Display grid
        grid.innerHTML = '';
        entries.forEach((entry, index) => {
            const card = createAirfoilCard(entry, index);
            grid.appendChild(card);
        });
    } catch (error) {
        grid.innerHTML = `<div class="error-box">Error loading airfoils: ${error.message}</div>`;
    }
}

// Create airfoil card
function createAirfoilCard(entry, index) {
    const card = document.createElement('div');
    card.className = 'airfoil-card';
    card.onclick = () => showDetailModal(entry, index);
    
    const rank = document.createElement('div');
    rank.className = 'airfoil-rank';
    rank.textContent = `#${index + 1}`;
    
    const img = document.createElement('img');
    img.src = 'data:image/png;base64,' + entry.airfoil_plot;
    img.alt = `Airfoil #${index + 1}`;
    
    const stats = document.createElement('div');
    stats.className = 'airfoil-card-stats';
    if (currentCategory === 'peak_ld') {
        stats.innerHTML = `<strong>L/D:</strong> ${entry.ldmax.toFixed(2)}<br><strong>Cl:</strong> ${entry.clmax.toFixed(3)}`;
    } else {
        stats.innerHTML = `<strong>Cl:</strong> ${entry.clmax.toFixed(3)}<br><strong>L/D:</strong> ${entry.ldmax.toFixed(2)}`;
    }
    
    card.appendChild(rank);
    card.appendChild(img);
    card.appendChild(stats);
    
    return card;
}

// Show detail modal
function showDetailModal(entry, index) {
    const modal = document.getElementById('detail-modal');
    const content = document.getElementById('detail-content');
    
    const categoryName = currentCategory === 'peak_ld' ? 'Peak L/D' : 'Peak Cl';
    const reBinLabel = formatReBin(currentReBin);
    
    content.innerHTML = `
        <h2>Rank #${index + 1} - ${categoryName} at Re = ${reBinLabel}</h2>
        
        <div class="detail-stats">
            <div class="stat-card">
                <div class="stat-label">Cl Max</div>
                <div class="stat-value">${entry.clmax.toFixed(3)}</div>
                <div class="stat-detail">at α = ${entry.alpha_clmax.toFixed(2)}°</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Cd Min</div>
                <div class="stat-value">${entry.cdmin.toFixed(4)}</div>
                <div class="stat-detail">at α = ${entry.alpha_cdmin.toFixed(2)}°</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">L/D Max</div>
                <div class="stat-value">${entry.ldmax.toFixed(2)}</div>
                <div class="stat-detail">at α = ${entry.alpha_ldmax.toFixed(2)}°</div>
            </div>
        </div>
        
        <div class="plot-container">
            <h3>Airfoil Shape</h3>
            <img src="data:image/png;base64,${entry.airfoil_plot}" alt="Airfoil Shape">
        </div>
        
        <div class="plot-container">
            <h3>Aerodynamic Curves</h3>
            <img src="data:image/png;base64,${entry.curves_plot}" alt="Aerodynamic Curves">
        </div>
        
        <div class="detail-params">
            <p><strong>Thickness:</strong> Min = ${entry.min_thickness.toFixed(3)}, Max = ${entry.max_thickness.toFixed(3)}</p>
        </div>
        
        <div class="download-section">
            <button class="btn btn-primary" onclick="downloadHofAirfoil(${currentReBin}, '${currentCategory}', ${index})">
                Download .dat File
            </button>
        </div>
    `;
    
    modal.style.display = 'block';
}

// Close detail modal
function closeDetailModal() {
    document.getElementById('detail-modal').style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('detail-modal');
    if (event.target === modal) {
        closeDetailModal();
    }
}

// Download HOF airfoil
function downloadHofAirfoil(reBin, category, index) {
    window.open(`${API_BASE}/api/hof/download/${reBin}/${category}/${index}`, '_blank');
}

// Format Re bin for display
function formatReBin(reBin) {
    if (reBin >= 1000000) {
        return '1m+';
    } else if (reBin >= 1000) {
        return `${reBin / 1000}k`;
    }
    return reBin.toString();
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

// Reset colors to default
function resetColors() {
    localStorage.removeItem('color-primary');
    localStorage.removeItem('color-success');
    localStorage.removeItem('color-secondary');
    localStorage.removeItem('color-background');
    localStorage.removeItem('color-card');
    loadColors();
}

