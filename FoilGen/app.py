"""
Flask Web Application for Interactive Airfoil Design

Check out the project files: https://github.com/AvnehSBhatia/FoilML
"""
from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
import torch
import torch.nn as nn
import numpy as np
import os
import json
import base64
import io
from pathlib import Path
from scipy.interpolate import interp1d, UnivariateSpline, CubicSpline
try:
    import neuralfoil as nf
    NEURALFOIL_AVAILABLE = True
except ImportError:
    NEURALFOIL_AVAILABLE = False
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import model definitions and functions from the original script
from interactive_airfoil_design import (
    SequenceEncoder,
    AeroToAF512Net,
    AF512toXYNet,
    CorrectionNet,
    predict_xy_coordinates,
    compute_reynolds_mach,
    bin_reynolds_number,
    get_max_cl_ld_for_re_bin,
    create_alpha_vector,
    compute_cd_from_cl_clcd,
    compute_summary_statistics,
    create_feature_dict,
    normalize_user_features,
    run_pipeline,
    af512_to_coordinates
)

# Correction model is available if CorrectionNet was imported successfully
CORRECTION_MODEL_AVAILABLE = True

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Hall of Fame storage path (use /data mount from Fly.io volume if available, otherwise local)
# The /data directory is mounted from the "gallery" Fly.io volume as configured in fly.toml
HOF_DATA_DIR = '/data' if os.path.exists('/data') else 'hof_data'
HOF_DB_FILE = os.path.join(HOF_DATA_DIR, 'hall_of_fame.json')
try:
    os.makedirs(HOF_DATA_DIR, exist_ok=True)
    # Test write access
    test_file = os.path.join(HOF_DATA_DIR, '.test_write')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    print(f"Hall of Fame storage initialized at: {HOF_DATA_DIR}")
except Exception as e:
    print(f"Warning: Could not initialize HOF storage at {HOF_DATA_DIR}: {e}")
    # Fallback to local storage
    HOF_DATA_DIR = 'hof_data'
    HOF_DB_FILE = os.path.join(HOF_DATA_DIR, 'hall_of_fame.json')
    os.makedirs(HOF_DATA_DIR, exist_ok=True)
    print(f"Using fallback storage at: {HOF_DATA_DIR}")

# Global model variables (loaded once on startup)
aero_model = None
af512_to_xy_model = None
correction_model = None
normalization_data = None
device = None

# Hall of Fame bins
RE_BINS = [50000, 100000, 250000, 500000, 750000, 1000000]
MAX_HOF_ENTRIES = 10  # Top 10 per category

# Optimization cancellation tracking
optimization_cancelled = {}  # Track cancelled optimizations by request ID

def load_models():
    """Load models once at startup with optimizations for small hardware."""
    global aero_model, af512_to_xy_model, correction_model, normalization_data, device
    
    # Force CPU-only for deployment
    device = torch.device('cpu')
    
    # Optimize CPU settings for inference
    # Set number of threads for optimal performance (use all available, but limit to avoid overhead)
    num_threads = min(torch.get_num_threads(), 4)  # Limit to 4 threads for small hardware
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)  # Use single inter-op thread for better latency
    print(f"Optimized CPU threads: {torch.get_num_threads()} compute threads, 1 inter-op thread")
    
    print(f"Loading models on device: {device} (CPU-only)")
    
    # Load AeroToAF512 model
    aero_model_file = 'aero_to_af512_model.pth'
    if not os.path.exists(aero_model_file):
        raise FileNotFoundError(f"Model file {aero_model_file} not found!")
    
    model_state = torch.load(aero_model_file, map_location='cpu')
    
    # Determine architecture from saved weights
    has_lstm = any('sequence_encoder.lstm' in key for key in model_state.keys())
    has_projection = 'sequence_encoder.projection.0.weight' in model_state
    
    first_layer_key = 'network.0.weight'
    if first_layer_key in model_state:
        first_layer_weight = model_state[first_layer_key]
        combined_input_size = first_layer_weight.shape[1]
        
        hidden_sizes = []
        i = 0
        while f'network.{i*3}.weight' in model_state:
            weight_key = f'network.{i*3}.weight'
            hidden_size = model_state[weight_key].shape[0]
            hidden_sizes.append(hidden_size)
            i += 1
        
        if len(hidden_sizes) > 0:
            hidden_sizes = hidden_sizes[:-1]
        
        sequence_embedding_dim = 256
        if has_projection:
            sequence_embedding_dim = model_state['sequence_encoder.projection.0.weight'].shape[0]
        elif not has_lstm:
            if combined_input_size <= 14:
                sequence_embedding_dim = 4
            else:
                sequence_embedding_dim = combined_input_size - 10
        
        scalar_input_size = combined_input_size - sequence_embedding_dim
        
        output_size = 1024
        last_layer_idx = len(hidden_sizes) * 3
        if f'network.{last_layer_idx}.weight' in model_state:
            output_size = model_state[f'network.{last_layer_idx}.weight'].shape[0]
        
        aero_model = AeroToAF512Net(
            scalar_input_size=scalar_input_size,
            sequence_embedding_dim=sequence_embedding_dim,
            output_size=output_size,
            hidden_sizes=hidden_sizes if hidden_sizes else [256, 256, 256, 256]
        )
    else:
        aero_model = AeroToAF512Net()
    
    missing_keys, unexpected_keys = aero_model.load_state_dict(model_state, strict=False)
    aero_model = aero_model.to(device)
    aero_model.eval()
    
    # Apply dynamic quantization to AeroToAF512 model
    # Quantization reduces memory usage by ~4x and speeds up inference by 2-4x on CPU
    # Note: LSTM layers won't be quantized, only Linear layers
    print("Applying dynamic quantization to AeroToAF512 model...")
    try:
        aero_model = torch.quantization.quantize_dynamic(
            aero_model, 
            {nn.Linear},  # Only quantize Linear layers (LSTM stays float32)
            dtype=torch.qint8
        )
        print("✓ AeroToAF512 model quantized successfully")
    except Exception as e:
        print(f"⚠ Warning: Quantization failed for AeroToAF512 ({e}), continuing without quantization")
    
    # Load AF512toXY model
    af512_to_xy_model_file = 'af512_to_xy_model.pth'
    if not os.path.exists(af512_to_xy_model_file):
        raise FileNotFoundError(f"Model file {af512_to_xy_model_file} not found!")
    
    af512_to_xy_model = AF512toXYNet()
    af512_to_xy_model.load_state_dict(torch.load(af512_to_xy_model_file, map_location='cpu'))
    af512_to_xy_model = af512_to_xy_model.to(device)
    af512_to_xy_model.eval()
    
    # Apply dynamic quantization to AF512toXY model FIRST
    # This gives the biggest performance boost (2-4x speedup)
    print("Applying dynamic quantization to AF512toXY model...")
    quantization_applied = False
    try:
        af512_to_xy_model = torch.quantization.quantize_dynamic(
            af512_to_xy_model,
            {nn.Linear},
            dtype=torch.qint8
        )
        quantization_applied = True
        print("✓ AF512toXY model quantized successfully")
    except Exception as e:
        print(f"⚠ Quantization failed for AF512toXY ({e}), continuing without quantization")
    
    # Try to compile quantized AF512toXY model with TorchScript for additional speedup
    # TorchScript can provide 10-30% additional speedup on top of quantization
    if quantization_applied:
        print("Attempting to compile quantized AF512toXY model with TorchScript...")
        try:
            # Create a dummy input for tracing
            dummy_af512 = torch.randn(1, 1024)
            with torch.no_grad():
                # Trace the quantized model
                af512_to_xy_model_traced = torch.jit.trace(af512_to_xy_model, dummy_af512)
                af512_to_xy_model_traced.eval()
                # Test the traced model
                _ = af512_to_xy_model_traced(dummy_af512)
            af512_to_xy_model = af512_to_xy_model_traced
            print("✓ AF512toXY model compiled with TorchScript (quantized + traced)")
        except Exception as e:
            print(f"⚠ TorchScript tracing failed for AF512toXY ({e}), using quantized model without tracing")
    
    # Note: AeroToAF512 contains LSTM which is difficult to trace with TorchScript
    # The quantization already provides significant speedup for the Linear layers
    
    # Load normalization data
    norm_file = 'feature_normalization.npy'
    if not os.path.exists(norm_file):
        raise FileNotFoundError(f"Normalization file '{norm_file}' not found!")
    
    norm_dict = np.load(norm_file, allow_pickle=True).item()
    normalization_data = norm_dict
    
    # Load correction model if available
    if CORRECTION_MODEL_AVAILABLE:
        correction_model_file = 'correction_model.pth'
        if os.path.exists(correction_model_file):
            print(f"Loading Correction model from {correction_model_file}...")
            try:
                correction_model = CorrectionNet(
                    af512_size=1024,
                    error_size=11,
                    output_size=1024,
                    hidden_sizes=[512, 512, 512]  # Match the architecture used during training
                )
                correction_model.load_state_dict(torch.load(correction_model_file, map_location='cpu'))
                correction_model = correction_model.to(device)
                correction_model.eval()
                
                # Try to apply quantization to correction model
                try:
                    correction_model = torch.quantization.quantize_dynamic(
                        correction_model,
                        {nn.Linear},
                        dtype=torch.qint8
                    )
                    print("✓ Correction model loaded and quantized successfully")
                except Exception as e:
                    print(f"✓ Correction model loaded (quantization skipped: {e})")
            except Exception as e:
                print(f"⚠ Warning: Failed to load correction model: {e}")
                correction_model = None
        else:
            print(f"⚠ Correction model file {correction_model_file} not found. Continuing without correction.")
            correction_model = None
    else:
        print("⚠ Correction model not available (train_correction_model.py not found). Continuing without correction.")
        correction_model = None
    
    print("✓ Models loaded and optimized successfully!")

# Load models on startup
try:
    load_models()
except Exception as e:
    print(f"Error loading models: {e}")
    print("Server will start but predictions won't work until models are available.")

# Hall of Fame functions
def load_hof_database():
    """Load Hall of Fame database from JSON file."""
    if os.path.exists(HOF_DB_FILE):
        try:
            with open(HOF_DB_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading HOF database: {e}")
            return {}
    return {}

def save_hof_database(hof_db):
    """Save Hall of Fame database to JSON file."""
    try:
        with open(HOF_DB_FILE, 'w') as f:
            json.dump(hof_db, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving HOF database: {e}")
        return False

def initialize_hof_structure():
    """Initialize HOF database structure if it doesn't exist."""
    hof_db = load_hof_database()
    if not hof_db:
        hof_db = {}
        for re_bin in RE_BINS:
            hof_db[str(re_bin)] = {
                'peak_ld': [],  # List of top L/D entries
                'peak_cl': []   # List of top Cl entries
            }
        save_hof_database(hof_db)
    else:
        # Ensure all bins exist
        updated = False
        for re_bin in RE_BINS:
            if str(re_bin) not in hof_db:
                hof_db[str(re_bin)] = {'peak_ld': [], 'peak_cl': []}
                updated = True
        if updated:
            save_hof_database(hof_db)
    return hof_db

def check_hof_eligibility(re_bin, stats, hof_db=None):
    """
    Check if an airfoil qualifies for Hall of Fame.
    
    Args:
        re_bin: Binned Reynolds number
        stats: Dictionary with 'ldmax' and 'clmax' keys
        hof_db: Optional HOF database (will load if not provided)
    
    Returns:
        Tuple (qualifies_ld, qualifies_cl) - boolean flags
    """
    if hof_db is None:
        hof_db = load_hof_database()
    
    re_bin_str = str(re_bin)
    if re_bin_str not in hof_db:
        return True, True  # Empty category, always qualifies
    
    ld_entries = hof_db[re_bin_str].get('peak_ld', [])
    cl_entries = hof_db[re_bin_str].get('peak_cl', [])
    
    # Check if qualifies for peak L/D
    qualifies_ld = True
    if len(ld_entries) >= MAX_HOF_ENTRIES:
        # Check if this L/D is better than the worst in top 10
        worst_ld = min(entry['ldmax'] for entry in ld_entries)
        qualifies_ld = stats['ldmax'] > worst_ld
    
    # Check if qualifies for peak Cl
    qualifies_cl = True
    if len(cl_entries) >= MAX_HOF_ENTRIES:
        # Check if this Cl is better than the worst in top 10
        worst_cl = min(entry['clmax'] for entry in cl_entries)
        qualifies_cl = stats['clmax'] > worst_cl
    
    return qualifies_ld, qualifies_cl

def add_to_hof(re_bin, stats, x_coords, y_coords, airfoil_plot_b64, curves_plot_b64, 
                alpha, nf_cl, nf_cd, nf_ld, min_thickness, max_thickness):
    """
    Add an airfoil to Hall of Fame if it qualifies.
    
    Args:
        re_bin: Binned Reynolds number
        stats: Dictionary with performance stats
        x_coords, y_coords: Airfoil coordinates
        airfoil_plot_b64, curves_plot_b64: Base64 encoded plots
        alpha, nf_cl, nf_cd, nf_ld: Aerodynamic data
        min_thickness, max_thickness: Thickness parameters
    
    Returns:
        Tuple (added_ld, added_cl) - boolean flags indicating if added
    """
    hof_db = load_hof_database()
    re_bin_str = str(re_bin)
    
    # Initialize structure if needed
    if re_bin_str not in hof_db:
        hof_db[re_bin_str] = {'peak_ld': [], 'peak_cl': []}
    
    # Check eligibility first
    qualifies_ld, qualifies_cl = check_hof_eligibility(re_bin, stats, hof_db)
    
    if not qualifies_ld and not qualifies_cl:
        return False, False  # Doesn't qualify for either
    
    # Create entry
    entry = {
        'ldmax': float(stats['ldmax']),
        'clmax': float(stats['clmax']),
        'cdmin': float(stats.get('cdmin', 0)),
        'alpha_clmax': float(stats.get('alpha_clmax', 0)),
        'alpha_cdmin': float(stats.get('alpha_cdmin', 0)),
        'alpha_ldmax': float(stats.get('alpha_ldmax', 0)),
        'x_coords': [float(x) for x in x_coords],
        'y_coords': [float(y) for y in y_coords],
        'airfoil_plot': airfoil_plot_b64,
        'curves_plot': curves_plot_b64,
        'alpha': [float(a) for a in alpha],
        'nf_cl': [float(c) if not np.isnan(c) else None for c in nf_cl] if nf_cl is not None else None,
        'nf_cd': [float(c) if not np.isnan(c) else None for c in nf_cd] if nf_cd is not None else None,
        'nf_ld': [float(c) if not np.isnan(c) else None for c in nf_ld] if nf_ld is not None else None,
        'min_thickness': float(min_thickness),
        'max_thickness': float(max_thickness),
        're_bin': int(re_bin)
    }
    
    added_ld = False
    added_cl = False
    
    # Add to peak L/D list if qualifies
    if qualifies_ld:
        ld_entries = hof_db[re_bin_str]['peak_ld'].copy()
        ld_entries.append(entry)
        ld_entries.sort(key=lambda x: x['ldmax'], reverse=True)
        if len(ld_entries) > MAX_HOF_ENTRIES:
            ld_entries = ld_entries[:MAX_HOF_ENTRIES]
        
        # Check if our entry is in the top MAX_HOF_ENTRIES
        if any(abs(e['ldmax'] - entry['ldmax']) < 1e-6 for e in ld_entries):
            hof_db[re_bin_str]['peak_ld'] = ld_entries
            added_ld = True
    
    # Add to peak Cl list if qualifies
    if qualifies_cl:
        cl_entries = hof_db[re_bin_str]['peak_cl'].copy()
        cl_entries.append(entry)
        cl_entries.sort(key=lambda x: x['clmax'], reverse=True)
        if len(cl_entries) > MAX_HOF_ENTRIES:
            cl_entries = cl_entries[:MAX_HOF_ENTRIES]
        
        # Check if our entry is in the top MAX_HOF_ENTRIES
        if any(abs(e['clmax'] - entry['clmax']) < 1e-6 for e in cl_entries):
            hof_db[re_bin_str]['peak_cl'] = cl_entries
            added_cl = True
    
    # Save database if anything was added
    if added_ld or added_cl:
        save_hof_database(hof_db)
    
    return added_ld, added_cl

# Initialize HOF database on startup
initialize_hof_structure()

def scale_dat_coordinates(x_coords, y_coords):
    """
    Scale coordinates so that x extremes are -1 and 1.
    This normalizes the airfoil to a standard chord length.
    
    Args:
        x_coords: Array of x coordinates
        y_coords: Array of y coordinates
    
    Returns:
        Tuple of (scaled_x_coords, scaled_y_coords)
    """
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    x_range = x_max - x_min
    
    if x_range < 1e-10:  # Avoid division by zero
        return x_coords, y_coords
    
    # Scale x to [-1, 1] range
    # Linear transformation: new_x = -1 + 2 * (x - x_min) / (x_max - x_min)
    scaled_x = -1 + 2 * (x_coords - x_min) / x_range
    
    # Scale y proportionally to maintain aspect ratio
    scaled_y = y_coords * (2 / x_range)
    
    return scaled_x, scaled_y

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/hof')
def hof():
    """Render the Hall of Fame page."""
    return render_template('hof.html')

@app.route('/api/compute_reynolds', methods=['POST'])
def api_compute_reynolds():
    """Compute Reynolds and Mach numbers from chord and speed."""
    data = request.json
    chord_ft = float(data['chord_ft'])
    speed_mph = float(data['speed_mph'])
    
    Re, Mach = compute_reynolds_mach(chord_ft, speed_mph)
    re_bin = bin_reynolds_number(Re)
    max_cl, max_ld = get_max_cl_ld_for_re_bin(re_bin)
    
    return jsonify({
        're': float(Re),
        'mach': float(Mach),
        're_bin': float(re_bin),
        'max_cl': float(max_cl),
        'max_ld': float(max_ld)
    })

@app.route('/api/generate_alpha', methods=['POST'])
def api_generate_alpha():
    """Generate alpha vector from min/max angles."""
    data = request.json
    a_min = float(data['a_min'])
    a_max = float(data['a_max'])
    increment = 0.5
    
    alpha = create_alpha_vector(a_min, a_max, increment)
    
    return jsonify({
        'alpha': alpha.tolist(),
        'count': len(alpha)
    })

@app.route('/api/interpolate_points', methods=['POST'])
def api_interpolate_points():
    """Interpolate control points to full alpha vector using splines."""
    data = request.json
    control_points = data['control_points']  # List of [alpha, value] pairs
    alpha_vector = np.array(data['alpha_vector'])
    
    if len(control_points) == 0:
        return jsonify({'error': 'No control points provided'}), 400
    
    # Remove duplicates by alpha, keeping the last value for each alpha
    unique_points = {}
    for alpha, value in control_points:
        unique_points[alpha] = value
    sorted_points = sorted(unique_points.items())
    alphas = np.array([p[0] for p in sorted_points])
    values = np.array([p[1] for p in sorted_points])
    
    if len(sorted_points) == 1:
        interpolated = np.full_like(alpha_vector, values[0])
    else:
        # Use cubic spline interpolation for smooth curves
        try:
            if len(sorted_points) >= 4:
                # Use CubicSpline for smooth curves (requires at least 4 points for cubic)
                spline = CubicSpline(alphas, values, bc_type='natural', extrapolate=True)
                interpolated = spline(alpha_vector)
            elif len(sorted_points) == 3:
                # For 3 points, use quadratic spline (k=2)
                spline = UnivariateSpline(alphas, values, s=0, k=2)
                interpolated = spline(alpha_vector)
            elif len(sorted_points) == 2:
                # For 2 points, use linear interpolation
                interp_func = interp1d(alphas, values, kind='linear', 
                                      bounds_error=False, fill_value='extrapolate')
                interpolated = interp_func(alpha_vector)
            else:
                # Fallback to linear
                interp_func = interp1d(alphas, values, kind='linear', 
                                      bounds_error=False, fill_value='extrapolate')
                interpolated = interp_func(alpha_vector)
        except Exception as e:
            # Fallback to linear if spline fails
            try:
                interp_func = interp1d(alphas, values, kind='linear', 
                                      bounds_error=False, fill_value='extrapolate')
                interpolated = interp_func(alpha_vector)
            except:
                # Last resort: constant value
                interpolated = np.full_like(alpha_vector, values[0])
    
    return jsonify({
        'interpolated': interpolated.tolist()
    })

@app.route('/api/generate_airfoil', methods=['POST'])
def api_generate_airfoil():
    """Generate airfoil from user inputs."""
    if aero_model is None or af512_to_xy_model is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.json
        
        # Extract all inputs
        Re = float(data['re'])
        Mach = float(data['mach'])
        alpha = np.array(data['alpha'])
        cl = np.array(data['cl'])
        cl_cd = np.array(data['cl_cd'])
        min_thickness = float(data['min_thickness'])
        max_thickness = float(data['max_thickness'])
        
        # Compute Cd and statistics (for model input, we still use user input)
        cd = compute_cd_from_cl_clcd(cl, cl_cd)
        user_stats = compute_summary_statistics(alpha, cl, cd, cl_cd)
        
        # Create feature dictionary (using user input stats for model)
        feature_dict = create_feature_dict(
            Re, Mach, alpha, cl, cd, cl_cd,
            min_thickness, max_thickness, user_stats
        )
        
        # Run pipeline
        pred_x, pred_y = run_pipeline(
            aero_model, af512_to_xy_model, feature_dict,
            normalization_data, correction_model=correction_model, device=device
        )
        
        # Run NeuralFoil analysis if available
        nf_cl_arr = None
        nf_cd_arr = None
        nf_ld_arr = None
        nf_stats = None
        if NEURALFOIL_AVAILABLE:
            try:
                coordinates = np.column_stack([pred_x, pred_y])
                nf_cl_list = []
                nf_cd_list = []
                nf_ld_list = []
                
                for alpha_val in alpha:
                    try:
                        nf_results = nf.get_aero_from_coordinates(
                            coordinates=coordinates,
                            alpha=float(alpha_val),
                            Re=Re,
                            model_size="xxxlarge"
                        )
                        
                        pred_cl = nf_results['CL']
                        pred_cd = nf_results['CD']
                        
                        if isinstance(pred_cl, np.ndarray):
                            pred_cl = float(pred_cl[0])
                        if isinstance(pred_cd, np.ndarray):
                            pred_cd = float(pred_cd[0])
                        
                        pred_ld = pred_cl / pred_cd if pred_cd > 0 else 0
                        
                        nf_cl_list.append(pred_cl)
                        nf_cd_list.append(pred_cd)
                        nf_ld_list.append(pred_ld)
                    except:
                        nf_cl_list.append(np.nan)
                        nf_cd_list.append(np.nan)
                        nf_ld_list.append(np.nan)
                
                nf_cl_arr = np.array(nf_cl_list)
                nf_cd_arr = np.array(nf_cd_list)
                nf_ld_arr = np.array(nf_ld_list)
                
                # Compute stats from NeuralFoil results (filtering out NaN values)
                valid_mask = ~(np.isnan(nf_cl_arr) | np.isnan(nf_cd_arr) | np.isnan(nf_ld_arr))
                if np.sum(valid_mask) > 0:
                    nf_alpha_valid = np.array(alpha)[valid_mask]
                    nf_cl_valid = nf_cl_arr[valid_mask]
                    nf_cd_valid = nf_cd_arr[valid_mask]
                    nf_ld_valid = nf_ld_arr[valid_mask]
                    nf_stats = compute_summary_statistics(nf_alpha_valid, nf_cl_valid, nf_cd_valid, nf_ld_valid)
            except Exception as e:
                print(f"NeuralFoil analysis failed: {e}")
        
        # Create plots
        os.makedirs('output', exist_ok=True)
        
        # Airfoil shape plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(pred_x, pred_y, 'b-', linewidth=2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (chord fraction)')
        ax.set_ylabel('Y (chord fraction)')
        ax.set_title('Generated Airfoil Shape')
        plt.tight_layout()
        
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        img_buf.seek(0)
        airfoil_plot_b64 = base64.b64encode(img_buf.read()).decode()
        plt.close()
        
        # Aerodynamic curves plot with NeuralFoil overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        alpha_arr = np.array(alpha)
        
        # Cl vs alpha
        axes[0].plot(alpha_arr, cl, 'b-o', label='User Input', linewidth=2, markersize=4)
        if nf_cl_arr is not None:
            valid_mask = ~np.isnan(nf_cl_arr)
            if np.sum(valid_mask) > 0:
                axes[0].plot(alpha_arr[valid_mask], nf_cl_arr[valid_mask], 'r--s', 
                            label='NeuralFoil', linewidth=2, markersize=4, alpha=0.7)
        axes[0].set_xlabel('Angle of Attack (deg)')
        axes[0].set_ylabel('Cl')
        axes[0].set_title('Lift Coefficient vs AoA')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cd vs alpha
        axes[1].plot(alpha_arr, cd, 'r-o', label='User Input', linewidth=2, markersize=4)
        if nf_cd_arr is not None:
            valid_mask = ~np.isnan(nf_cd_arr)
            if np.sum(valid_mask) > 0:
                axes[1].plot(alpha_arr[valid_mask], nf_cd_arr[valid_mask], 'b--s', 
                            label='NeuralFoil', linewidth=2, markersize=4, alpha=0.7)
        axes[1].set_xlabel('Angle of Attack (deg)')
        axes[1].set_ylabel('Cd')
        axes[1].set_title('Drag Coefficient vs AoA')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Cl/Cd vs alpha
        axes[2].plot(alpha_arr, cl_cd, 'g-o', label='User Input', linewidth=2, markersize=4)
        if nf_ld_arr is not None:
            valid_mask = ~np.isnan(nf_ld_arr)
            if np.sum(valid_mask) > 0:
                axes[2].plot(alpha_arr[valid_mask], nf_ld_arr[valid_mask], 'm--s', 
                            label='NeuralFoil', linewidth=2, markersize=4, alpha=0.7)
        axes[2].set_xlabel('Angle of Attack (deg)')
        axes[2].set_ylabel('Cl/Cd')
        axes[2].set_title('Lift-to-Drag Ratio vs AoA')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        img_buf.seek(0)
        curves_plot_b64 = base64.b64encode(img_buf.read()).decode()
        plt.close()
        
        # Save .dat file with scaled coordinates
        dat_path = os.path.join('output', 'generated_airfoil.dat')
        scaled_x, scaled_y = scale_dat_coordinates(pred_x, pred_y)
        with open(dat_path, 'w') as f:
            f.write("Generated Airfoil\n")
            for x, y in zip(scaled_x, scaled_y):
                f.write(f"{x:.6f} {y:.6f}\n")
        
        # Use NeuralFoil stats if available, otherwise fall back to user input stats
        display_stats = nf_stats if nf_stats is not None else user_stats
        
        return jsonify({
            'success': True,
            'x_coords': pred_x.tolist(),
            'y_coords': pred_y.tolist(),
            'stats': {
                'clmax': float(display_stats['clmax']),
                'cdmin': float(display_stats['cdmin']),
                'ldmax': float(display_stats['ldmax']),
                'alpha_clmax': float(display_stats['alpha_clmax']),
                'alpha_cdmin': float(display_stats['alpha_cdmin']),
                'alpha_ldmax': float(display_stats['alpha_ldmax'])
            },
            'plots': {
                'airfoil_shape': airfoil_plot_b64,
                'aerodynamic_curves': curves_plot_b64
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_and_evaluate_airfoil(Re, Mach, alpha, cl_target, cl_cd_target, min_thickness, max_thickness):
    """
    Generate an airfoil and evaluate its error against target curves.
    
    Returns:
        (pred_x, pred_y, nf_cl_arr, nf_cd_arr, nf_ld_arr, error, airfoil_plot_b64)
    """
    # Compute Cd and statistics
    cd_target = compute_cd_from_cl_clcd(cl_target, cl_cd_target)
    user_stats = compute_summary_statistics(alpha, cl_target, cd_target, cl_cd_target)
    
    # Create feature dictionary
    feature_dict = create_feature_dict(
        Re, Mach, alpha, cl_target, cd_target, cl_cd_target,
        min_thickness, max_thickness, user_stats
    )
    
    # Run pipeline
    pred_x, pred_y = run_pipeline(
        aero_model, af512_to_xy_model, feature_dict,
        normalization_data, correction_model=correction_model, device=device
    )
    
    # Generate airfoil shape plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pred_x, pred_y, 'b-', linewidth=2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (chord fraction)')
    ax.set_ylabel('Y (chord fraction)')
    ax.set_title('Generated Airfoil Shape')
    plt.tight_layout()
    
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
    img_buf.seek(0)
    airfoil_plot_b64 = base64.b64encode(img_buf.read()).decode()
    plt.close()
    
    # Run NeuralFoil analysis if available
    nf_cl_arr = None
    nf_cd_arr = None
    nf_ld_arr = None
    error = float('inf')
    
    if NEURALFOIL_AVAILABLE:
        try:
            coordinates = np.column_stack([pred_x, pred_y])
            nf_cl_list = []
            nf_cd_list = []
            nf_ld_list = []
            
            for alpha_val in alpha:
                try:
                    nf_results = nf.get_aero_from_coordinates(
                        coordinates=coordinates,
                        alpha=float(alpha_val),
                        Re=Re,
                        model_size="xxxlarge"
                    )
                    
                    pred_cl = nf_results['CL']
                    pred_cd = nf_results['CD']
                    
                    if isinstance(pred_cl, np.ndarray):
                        pred_cl = float(pred_cl[0])
                    if isinstance(pred_cd, np.ndarray):
                        pred_cd = float(pred_cd[0])
                    
                    pred_ld = pred_cl / pred_cd if pred_cd > 0 else 0
                    
                    nf_cl_list.append(pred_cl)
                    nf_cd_list.append(pred_cd)
                    nf_ld_list.append(pred_ld)
                except:
                    nf_cl_list.append(np.nan)
                    nf_cd_list.append(np.nan)
                    nf_ld_list.append(np.nan)
            
            nf_cl_arr = np.array(nf_cl_list)
            nf_cd_arr = np.array(nf_cd_list)
            nf_ld_arr = np.array(nf_ld_list)
            
            # Compute error: mean squared error between NeuralFoil results and target
            valid_mask = ~(np.isnan(nf_cl_arr) | np.isnan(nf_cd_arr))
            if np.sum(valid_mask) > 0:
                # Normalize errors by range of target values
                cl_range = np.max(cl_target) - np.min(cl_target)
                cl_cd_range = np.max(cl_cd_target) - np.min(cl_cd_target)
                
                # Compute squared error for Cl
                cl_target_valid = cl_target[valid_mask]
                nf_cl_valid = nf_cl_arr[valid_mask]
                if cl_range > 0:
                    cl_error = np.sum(((nf_cl_valid - cl_target_valid) / cl_range) ** 2)
                else:
                    cl_error = np.sum((nf_cl_valid - cl_target_valid) ** 2)
                
                # Compute squared error for L/D
                nf_ld_valid = nf_ld_arr[valid_mask]
                cl_cd_target_valid = cl_cd_target[valid_mask]
                if cl_cd_range > 0:
                    ld_error = np.sum(((nf_ld_valid - cl_cd_target_valid) / cl_cd_range) ** 2)
                else:
                    ld_error = np.sum((nf_ld_valid - cl_cd_target_valid) ** 2)
                
                # Weighted combination: 50% Cl, 50% L/D
                error = 0.5 * cl_error + 0.5 * ld_error
        except Exception as e:
            print(f"NeuralFoil analysis failed: {e}")
    
    return pred_x, pred_y, nf_cl_arr, nf_cd_arr, nf_ld_arr, error, airfoil_plot_b64

@app.route('/api/optimize_airfoil', methods=['POST'])
def api_optimize_airfoil():
    """Generate airfoil with specified min and max thickness values."""
    if aero_model is None or af512_to_xy_model is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    # Generate unique request ID for cancellation tracking
    import uuid
    request_id = str(uuid.uuid4())
    optimization_cancelled[request_id] = False
    
    @stream_with_context
    def generate():
        try:
            # Send request ID as first message
            yield f"data: {json.dumps({'type': 'init', 'request_id': request_id})}\n\n"
            
            data = request.json
            
            # Extract inputs
            Re = float(data['re'])
            Mach = float(data['mach'])
            alpha = np.array(data['alpha'])
            cl_target = np.array(data['cl'])
            cl_cd_target = np.array(data['cl_cd'])
            min_thickness = float(data['min_thickness'])
            max_thickness = float(data['max_thickness'])
            
            # Validate thickness values
            if min_thickness >= max_thickness:
                yield f"data: {json.dumps({'type': 'error', 'error': 'min_thickness must be less than max_thickness'})}\n\n"
                return
            
            # Check for cancellation
            if optimization_cancelled.get(request_id, False):
                yield f"data: {json.dumps({'type': 'cancelled', 'message': 'Optimization cancelled by user'})}\n\n"
                return
            
            # Generate and evaluate airfoil
            try:
                pred_x, pred_y, nf_cl_arr, nf_cd_arr, nf_ld_arr, error, airfoil_plot = \
                    generate_and_evaluate_airfoil(
                        Re, Mach, alpha, cl_target, cl_cd_target, min_thickness, max_thickness
                    )
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'error': f'Error generating airfoil: {str(e)}'})}\n\n"
                return
            
            # Generate final plots
            if pred_x is not None:
                # Aerodynamic curves plot
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                alpha_arr = np.array(alpha)
                
                # Cl vs alpha
                axes[0].plot(alpha_arr, cl_target, 'b-o', label='Target', linewidth=2, markersize=4)
                if nf_cl_arr is not None:
                    valid_mask = ~np.isnan(nf_cl_arr)
                    if np.sum(valid_mask) > 0:
                        axes[0].plot(alpha_arr[valid_mask], nf_cl_arr[valid_mask], 'r--s', 
                                    label='NeuralFoil', linewidth=2, markersize=4, alpha=0.7)
                axes[0].set_xlabel('Angle of Attack (deg)')
                axes[0].set_ylabel('Cl')
                axes[0].set_title('Lift Coefficient vs AoA')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Cd vs alpha
                cd_target = compute_cd_from_cl_clcd(cl_target, cl_cd_target)
                axes[1].plot(alpha_arr, cd_target, 'r-o', label='Target', linewidth=2, markersize=4)
                if nf_cd_arr is not None:
                    valid_mask = ~np.isnan(nf_cd_arr)
                    if np.sum(valid_mask) > 0:
                        axes[1].plot(alpha_arr[valid_mask], nf_cd_arr[valid_mask], 'b--s', 
                                    label='NeuralFoil', linewidth=2, markersize=4, alpha=0.7)
                axes[1].set_xlabel('Angle of Attack (deg)')
                axes[1].set_ylabel('Cd')
                axes[1].set_title('Drag Coefficient vs AoA')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                # Cl/Cd vs alpha
                axes[2].plot(alpha_arr, cl_cd_target, 'g-o', label='Target', linewidth=2, markersize=4)
                if nf_ld_arr is not None:
                    valid_mask = ~np.isnan(nf_ld_arr)
                    if np.sum(valid_mask) > 0:
                        axes[2].plot(alpha_arr[valid_mask], nf_ld_arr[valid_mask], 'm--s', 
                                    label='NeuralFoil', linewidth=2, markersize=4, alpha=0.7)
                axes[2].set_xlabel('Angle of Attack (deg)')
                axes[2].set_ylabel('Cl/Cd')
                axes[2].set_title('Lift-to-Drag Ratio vs AoA')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                img_buf = io.BytesIO()
                plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
                img_buf.seek(0)
                curves_plot_b64 = base64.b64encode(img_buf.read()).decode()
                plt.close()
                
                # Compute stats from NeuralFoil results
                nf_stats = None
                if nf_cl_arr is not None and nf_cd_arr is not None and nf_ld_arr is not None:
                    valid_mask = ~(np.isnan(nf_cl_arr) | np.isnan(nf_cd_arr) | np.isnan(nf_ld_arr))
                    if np.sum(valid_mask) > 0:
                        nf_alpha_valid = alpha_arr[valid_mask]
                        nf_cl_valid = nf_cl_arr[valid_mask]
                        nf_cd_valid = nf_cd_arr[valid_mask]
                        nf_ld_valid = nf_ld_arr[valid_mask]
                        nf_stats = compute_summary_statistics(nf_alpha_valid, nf_cl_valid, nf_cd_valid, nf_ld_valid)
                
                # Save .dat file with scaled coordinates
                os.makedirs('output', exist_ok=True)
                dat_path = os.path.join('output', 'generated_airfoil.dat')
                scaled_x, scaled_y = scale_dat_coordinates(pred_x, pred_y)
                with open(dat_path, 'w') as f:
                    f.write("Generated Airfoil\n")
                    for x, y in zip(scaled_x, scaled_y):
                        f.write(f"{x:.6f} {y:.6f}\n")
                
                # Check Hall of Fame eligibility
                hof_qualifies_ld = False
                hof_qualifies_cl = False
                hof_added_ld = False
                hof_added_cl = False
                if nf_stats and NEURALFOIL_AVAILABLE:
                    re_bin = bin_reynolds_number(Re)
                    hof_qualifies_ld, hof_qualifies_cl = check_hof_eligibility(re_bin, nf_stats)
                    
                    if hof_qualifies_ld or hof_qualifies_cl:
                        hof_added_ld, hof_added_cl = add_to_hof(
                            re_bin, nf_stats, pred_x, pred_y,
                            airfoil_plot, curves_plot_b64,
                            alpha, nf_cl_arr, nf_cd_arr, nf_ld_arr,
                            min_thickness, max_thickness
                        )
                
                # Send final result
                yield f"data: {json.dumps({'type': 'complete', 'success': True, 'x_coords': pred_x.tolist(), 'y_coords': pred_y.tolist(), 'stats': {k: float(v) for k, v in (nf_stats or {}).items()}, 'plots': {'airfoil_shape': airfoil_plot, 'aerodynamic_curves': curves_plot_b64}, 'max_thickness': float(max_thickness), 'min_thickness': float(min_thickness), 'error': float(error), 'hof_added_ld': hof_added_ld, 'hof_added_cl': hof_added_cl})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'complete', 'success': False, 'error': 'No valid airfoil generated'})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            # Clean up cancellation tracking
            if request_id in optimization_cancelled:
                del optimization_cancelled[request_id]
    
    return Response(generate(), mimetype='text/event-stream', headers={'X-Request-ID': request_id})

@app.route('/api/cancel_optimization', methods=['POST'])
def api_cancel_optimization():
    """Cancel an ongoing optimization."""
    request_id = request.headers.get('X-Request-ID') or request.json.get('request_id')
    if request_id:
        optimization_cancelled[request_id] = True
        return jsonify({'success': True, 'message': 'Cancellation requested'})
    return jsonify({'error': 'Request ID not provided'}), 400

@app.route('/api/download_dat', methods=['GET'])
def api_download_dat():
    """Download the generated .dat file."""
    dat_path = os.path.join('output', 'generated_airfoil.dat')
    if os.path.exists(dat_path):
        return send_file(dat_path, as_attachment=True, download_name='generated_airfoil.dat')
    else:
        return jsonify({'error': 'No airfoil file generated yet'}), 404

@app.route('/api/export_config', methods=['GET', 'POST'])
def api_export_config():
    """Export flight conditions and curves as JSON."""
    try:
        data = request.json if request.method == 'POST' else request.args.to_dict()
        
        # Extract all configuration data
        config = {
            'chord_ft': float(data.get('chord_ft', 0)),
            'speed_mph': float(data.get('speed_mph', 0)),
            're': float(data.get('re', 0)),
            'mach': float(data.get('mach', 0)),
            'alpha_min': float(data.get('alpha_min', 0)),
            'alpha_max': float(data.get('alpha_max', 0)),
            'alpha': data.get('alpha', []),
            'cl_points': data.get('cl_points', []),
            'clcd_points': data.get('clcd_points', []),
            'cl_interpolated': data.get('cl_interpolated', []),
            'clcd_interpolated': data.get('clcd_interpolated', []),
            'min_thickness': float(data.get('min_thickness', 0.01)),
            'max_thickness': float(data.get('max_thickness', 0.15))
        }
        
        # Create response with JSON download
        response = jsonify(config)
        response.headers['Content-Disposition'] = 'attachment; filename=airfoil_config.json'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/import_config', methods=['POST'])
def api_import_config():
    """Import flight conditions and curves from JSON."""
    try:
        if 'file' in request.files:
            # File upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            config = json.load(file)
        elif request.json:
            # JSON data directly
            config = request.json
        else:
            return jsonify({'error': 'No configuration data provided'}), 400
        
        # Validate and return config
        validated_config = {
            'chord_ft': float(config.get('chord_ft', 0)),
            'speed_mph': float(config.get('speed_mph', 0)),
            're': float(config.get('re', 0)),
            'mach': float(config.get('mach', 0)),
            'alpha_min': float(config.get('alpha_min', 0)),
            'alpha_max': float(config.get('alpha_max', 0)),
            'alpha': config.get('alpha', []),
            'cl_points': config.get('cl_points', []),
            'clcd_points': config.get('clcd_points', []),
            'cl_interpolated': config.get('cl_interpolated', []),
            'clcd_interpolated': config.get('clcd_interpolated', []),
            'min_thickness': float(config.get('min_thickness', 0.01)),
            'max_thickness': float(config.get('max_thickness', 0.15))
        }
        
        return jsonify({'success': True, 'config': validated_config})
    except Exception as e:
        return jsonify({'error': f'Invalid configuration file: {str(e)}'}), 400

@app.route('/api/hof', methods=['GET'])
def api_get_hof():
    """Get all Hall of Fame entries."""
    hof_db = load_hof_database()
    return jsonify(hof_db)

@app.route('/api/hof/<int:re_bin>/<category>', methods=['GET'])
def api_get_hof_category(re_bin, category):
    """Get Hall of Fame entries for a specific Re bin and category (peak_ld or peak_cl)."""
    if category not in ['peak_ld', 'peak_cl']:
        return jsonify({'error': 'Invalid category. Use peak_ld or peak_cl'}), 400
    
    hof_db = load_hof_database()
    re_bin_str = str(re_bin)
    
    if re_bin_str not in hof_db:
        return jsonify([])
    
    entries = hof_db[re_bin_str].get(category, [])
    return jsonify(entries)

@app.route('/api/hof/download/<int:re_bin>/<category>/<int:index>', methods=['GET'])
def api_download_hof_airfoil(re_bin, category, index):
    """Download a Hall of Fame airfoil as .dat file."""
    if category not in ['peak_ld', 'peak_cl']:
        return jsonify({'error': 'Invalid category'}), 400
    
    hof_db = load_hof_database()
    re_bin_str = str(re_bin)
    
    if re_bin_str not in hof_db:
        return jsonify({'error': 'Re bin not found'}), 404
    
    entries = hof_db[re_bin_str].get(category, [])
    if index < 0 or index >= len(entries):
        return jsonify({'error': 'Index out of range'}), 404
    
    entry = entries[index]
    
    # Create .dat file content in memory with scaled coordinates
    scaled_x, scaled_y = scale_dat_coordinates(entry['x_coords'], entry['y_coords'])
    
    # Create file content
    file_content = f"Hall of Fame Airfoil - Re={re_bin}, {category.replace('_', ' ').title()}, Rank #{index+1}\n"
    for x, y in zip(scaled_x, scaled_y):
        file_content += f"{x:.6f} {y:.6f}\n"
    
    # Create BytesIO object and send directly
    from io import BytesIO
    file_buffer = BytesIO(file_content.encode('utf-8'))
    file_buffer.seek(0)
    
    return send_file(
        file_buffer,
        mimetype='text/plain',
        as_attachment=True,
        download_name=f'hof_airfoil_{re_bin}_{category}_{index}.dat'
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
