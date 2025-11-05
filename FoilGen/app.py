"""
Flask Web Application for Interactive Airfoil Design
"""
from flask import Flask, render_template, request, jsonify, send_file
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
    predict_xy_coordinates,
    compute_reynolds_mach,
    bin_reynolds_number,
    get_max_cl_ld_for_re_bin,
    create_alpha_vector,
    compute_cd_from_cl_clcd,
    compute_summary_statistics,
    create_feature_dict,
    normalize_user_features,
    run_pipeline
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model variables (loaded once on startup)
aero_model = None
af512_to_xy_model = None
normalization_data = None
device = None

def load_models():
    """Load models once at startup."""
    global aero_model, af512_to_xy_model, normalization_data, device
    
    # Force CPU-only for deployment
    device = torch.device('cpu')
    
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
    
    # Load AF512toXY model
    af512_to_xy_model_file = 'af512_to_xy_model.pth'
    if not os.path.exists(af512_to_xy_model_file):
        raise FileNotFoundError(f"Model file {af512_to_xy_model_file} not found!")
    
    af512_to_xy_model = AF512toXYNet()
    af512_to_xy_model.load_state_dict(torch.load(af512_to_xy_model_file, map_location='cpu'))
    af512_to_xy_model = af512_to_xy_model.to(device)
    af512_to_xy_model.eval()
    
    # Load normalization data
    norm_file = 'feature_normalization.npy'
    if not os.path.exists(norm_file):
        raise FileNotFoundError(f"Normalization file '{norm_file}' not found!")
    
    norm_dict = np.load(norm_file, allow_pickle=True).item()
    normalization_data = norm_dict
    
    print("Models loaded successfully!")

# Load models on startup
try:
    load_models()
except Exception as e:
    print(f"Error loading models: {e}")
    print("Server will start but predictions won't work until models are available.")

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

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
            normalization_data, device=device
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
        
        # Save .dat file
        dat_path = os.path.join('output', 'generated_airfoil.dat')
        with open(dat_path, 'w') as f:
            f.write("Generated Airfoil\n")
            for x, y in zip(pred_x, pred_y):
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

@app.route('/api/download_dat', methods=['GET'])
def api_download_dat():
    """Download the generated .dat file."""
    dat_path = os.path.join('output', 'generated_airfoil.dat')
    if os.path.exists(dat_path):
        return send_file(dat_path, as_attachment=True, download_name='generated_airfoil.dat')
    else:
        return jsonify({'error': 'No airfoil file generated yet'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
