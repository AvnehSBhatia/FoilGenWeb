"""
Interactive Airfoil Design Tool
Takes user inputs and generates an airfoil using the trained models, then compares with NeuralFoil.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from scipy.interpolate import interp1d, UnivariateSpline, CubicSpline
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    import neuralfoil as nf
    NEURALFOIL_AVAILABLE = True
except ImportError:
    NEURALFOIL_AVAILABLE = False
    print("Warning: NeuralFoil not installed. Install with: pip install neuralfoil")


# ============================================================================
# Model and utility definitions (previously imported from train_aero_to_af512 and train_airfoil)
# ============================================================================

class SequenceEncoder(nn.Module):
    """LSTM-based sequence encoder for variable-length aerodynamic vectors."""
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=2, embedding_dim=256):
        super(SequenceEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM encoder
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        # Output projection to fixed embedding size
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),  # *2 for bidirectional
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch_size, seq_len, input_dim) - variable length sequences (padded)
            lengths: (batch_size,) - actual lengths of each sequence (can be on any device)
        
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        # Pack sequences if lengths provided (requires CPU, but we'll handle it)
        if lengths is not None:
            # pack_padded_sequence requires lengths on CPU, so convert temporarily
            lengths_cpu = lengths.cpu() if lengths.device.type != 'cpu' else lengths
            x = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state from both directions
        # hidden shape: (num_layers * 2, batch_size, hidden_dim) for bidirectional
        # Take the last layer's forward and backward hidden states
        forward_hidden = hidden[-2]  # Last forward hidden state
        backward_hidden = hidden[-1]  # Last backward hidden state
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Project to fixed embedding size
        embedding = self.projection(combined_hidden)
        return embedding


class AeroToAF512Net(nn.Module):
    def __init__(self, scalar_input_size=10, sequence_embedding_dim=256, output_size=1024, 
                 hidden_sizes=[256, 256, 256, 256]):
        super(AeroToAF512Net, self).__init__()
        
        # Sequence encoder: encodes variable-length [alpha, cl, cd, cl_cd] vectors
        self.sequence_encoder = SequenceEncoder(
            input_dim=4,  # alpha, cl, cd, cl_cd
            hidden_dim=128,
            num_layers=2,
            embedding_dim=sequence_embedding_dim
        )
        
        # Combine scalar features + sequence embedding
        combined_input_size = scalar_input_size + sequence_embedding_dim
        
        # Main network
        layers = []
        prev_size = combined_input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, scalars, sequence, lengths=None):
        """
        Args:
            scalars: (batch_size, scalar_input_size) - Re, Mach, etc.
            sequence: (batch_size, seq_len, 4) - variable length [alpha, cl, cd, cl_cd] sequences
            lengths: (batch_size,) - actual lengths of sequences
        """
        # Encode sequence to fixed-size embedding
        seq_embedding = self.sequence_encoder(sequence, lengths)
        
        # Concatenate scalar features with sequence embedding
        combined = torch.cat([scalars, seq_embedding], dim=1)
        
        # Pass through main network
        output = self.network(combined)
        return output


def normalize_features(features_list):
    """Normalize scalar features to [0, 1] range."""
    # Extract all scalar features
    scalars_list = [f['scalars'] for f in features_list]
    scalars_array = np.array(scalars_list)
    
    min_vals = np.min(scalars_array, axis=0, keepdims=True)
    max_vals = np.max(scalars_array, axis=0, keepdims=True)
    
    # Avoid division by zero
    ranges = max_vals - min_vals
    ranges[ranges < 1e-6] = 1.0
    
    # Normalize scalar features
    normalized_features = []
    for feature_dict in features_list:
        normalized_scalars = (feature_dict['scalars'] - min_vals.flatten()) / ranges.flatten()
        normalized_features.append({
            'scalars': normalized_scalars,
            'sequence': feature_dict['sequence']  # Keep sequences as-is (will be normalized in model)
        })
    
    return normalized_features, min_vals, max_vals


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    scalars_list = []
    sequences_list = []
    targets_list = []
    lengths_list = []
    
    for item in batch:
        feature_dict, target = item
        scalars_list.append(feature_dict['scalars'])
        sequences_list.append(feature_dict['sequence'])
        targets_list.append(target)
        lengths_list.append(len(feature_dict['sequence']))
    
    # Pad sequences to same length
    max_len = max(lengths_list)
    padded_sequences = []
    for seq in sequences_list:
        if len(seq) < max_len:
            # Pad with zeros
            pad_len = max_len - len(seq)
            pad = torch.zeros(pad_len, seq.shape[1])
            padded_seq = torch.cat([seq, pad], dim=0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    # Stack into batches
    scalars_batch = torch.stack(scalars_list)
    sequences_batch = torch.stack(padded_sequences)
    targets_batch = torch.stack(targets_list)
    lengths_batch = torch.tensor(lengths_list, dtype=torch.long)
    
    return {
        'scalars': scalars_batch,
        'sequence': sequences_batch,
        'lengths': lengths_batch
    }, targets_batch


class CorrectionNet(nn.Module):
    """Correction model that takes predicted AF512 + errors and outputs corrected AF512."""
    def __init__(self, af512_size=1024, error_size=11, output_size=1024, hidden_sizes=[512, 512, 512]):
        super(CorrectionNet, self).__init__()
        
        input_size = af512_size + error_size
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LeakyReLU(),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class AF512toXYNet(nn.Module):
    
    def __init__(self, input_size=1024, output_size=2048, hidden_sizes=[256, 256, 256, 256]):
        super(AF512toXYNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(0.1),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def predict_xy_coordinates(model, af512_data, device='cpu'):
    """Predict XY coordinates from AF512 representation (optimized for inference)."""
    model.eval()
    # Disable gradient computation for faster inference
    torch.set_grad_enabled(False)
    try:
        if af512_data.ndim == 2:
            af512_flat = af512_data.flatten()
        else:
            af512_flat = af512_data
        
        # Use contiguous tensor for better memory access patterns
        input_tensor = torch.FloatTensor(af512_flat).unsqueeze(0).contiguous().to(device)
        output = model(input_tensor)
        
        # Handle both quantized and regular models (quantized models auto-dequantize)
        if isinstance(output, torch.Tensor):
            xy_flat = output.cpu().numpy().flatten()
        else:
            # Fallback for unexpected output types
            xy_flat = np.array(output).flatten()
        
        x_coords = xy_flat[:1024]
        y_coords = xy_flat[1024:]
        
        return x_coords, y_coords
    finally:
        # Re-enable gradients (in case other code needs them)
        torch.set_grad_enabled(True)


def load_dat_file(filepath, num_points=512):
    """
    Load airfoil coordinates from a .dat file.
    Format: First line is name, subsequent lines are x y coordinates.
    Upper surface goes from trailing edge (x=1) to leading edge (x=0),
    then lower surface from leading edge back to trailing edge.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Skip the first line (airfoil name)
        coords = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split()
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    coords.append((x, y))
            except ValueError:
                continue
        
        if len(coords) < 10:
            return None, None
        
        coords = np.array(coords)
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        # Find the leading edge (minimum x coordinate)
        # There might be duplicate points at leading edge, so find first occurrence
        min_x = np.min(x_coords)
        leading_edge_indices = np.where(np.abs(x_coords - min_x) < 1e-6)[0]
        
        if len(leading_edge_indices) == 0:
            leading_edge_idx = np.argmin(x_coords)
        else:
            # Use the middle occurrence if multiple
            leading_edge_idx = leading_edge_indices[len(leading_edge_indices) // 2]
        
        # Split into upper and lower surfaces
        # Upper surface: from start to leading edge (x goes from ~1 to ~0)
        # Lower surface: from leading edge to end (x goes from ~0 to ~1)
        upper_x = x_coords[:leading_edge_idx+1].copy()
        upper_y = y_coords[:leading_edge_idx+1].copy()
        lower_x = x_coords[leading_edge_idx:].copy()
        lower_y = y_coords[leading_edge_idx:].copy()
        
        # Reverse upper surface to go from leading edge (x=0) to trailing edge (x=1)
        upper_x = upper_x[::-1]
        upper_y = upper_y[::-1]
        
        # Normalize x coordinates to [0, 1] range
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        if x_max - x_min > 1e-6:
            upper_x = (upper_x - x_min) / (x_max - x_min)
            lower_x = (lower_x - x_min) / (x_max - x_min)
        else:
            # If all x are the same, create a uniform distribution
            upper_x = np.linspace(0, 1, len(upper_x))
            lower_x = np.linspace(0, 1, len(lower_x))
        
        # Ensure x coordinates are in [0, 1] range
        upper_x = np.clip(upper_x, 0, 1)
        lower_x = np.clip(lower_x, 0, 1)
        
        # Sort by x to ensure monotonic increase
        upper_sort_idx = np.argsort(upper_x)
        upper_x = upper_x[upper_sort_idx]
        upper_y = upper_y[upper_sort_idx]
        
        lower_sort_idx = np.argsort(lower_x)
        lower_x = lower_x[lower_sort_idx]
        lower_y = lower_y[lower_sort_idx]
        
        # Remove duplicates in x coordinates
        upper_x, upper_unique_idx = np.unique(upper_x, return_index=True)
        upper_y = upper_y[upper_unique_idx]
        
        lower_x, lower_unique_idx = np.unique(lower_x, return_index=True)
        lower_y = lower_y[lower_unique_idx]
        
        # Convert to AF512 format
        x_uniform = np.linspace(0, 1, num_points)
        
        # Interpolate upper and lower surfaces
        if len(upper_x) > 1 and len(lower_x) > 1:
            upper_interp = interp1d(upper_x, upper_y, kind='linear', bounds_error=False, fill_value='extrapolate')
            upper_y_uniform = upper_interp(x_uniform)
            
            lower_interp = interp1d(lower_x, lower_y, kind='linear', bounds_error=False, fill_value='extrapolate')
            lower_y_uniform = lower_interp(x_uniform)
        else:
            # Fallback if interpolation fails
            return None, None
        
        # Combine: [upper_y, lower_y]
        af512_data = np.concatenate([upper_y_uniform, lower_y_uniform])
        
        return af512_data, (x_uniform, upper_y_uniform, lower_y_uniform)
    
    except Exception as e:
        print(f"Error loading .dat file: {e}")
        return None, None


def af512_to_coordinates(af512_data):
    """
    Convert AF512 format to x,y coordinates.
    AF512 format: (num_points, 2) where first column is upper_y and second column is lower_y.
    The x coordinates are uniformly distributed from 0 to 1.
    
    Args:
        af512_data: Array of shape (num_points, 2) where columns are [upper_y, lower_y]
    
    Returns:
        x_coords, y_coords: Arrays of x and y coordinates
    """
    num_points = af512_data.shape[0]
    x_uniform = np.linspace(0, 1, num_points)
    
    upper_y = af512_data[:, 0]
    lower_y = af512_data[:, 1]
    
    # Reconstruct full airfoil: upper surface from trailing edge (x=1) to leading edge (x=0),
    # then lower surface from leading edge (x=0) back to trailing edge (x=1)
    x_coords = np.concatenate([x_uniform[::-1], x_uniform[1:]])
    y_coords = np.concatenate([upper_y[::-1], lower_y[1:]])
    
    return x_coords, y_coords


# ============================================================================
# Main script functions
# ============================================================================

def compute_reynolds_mach(chord_ft, speed_mph):
    """
    Compute Reynolds number and Mach number from chord and speed.
    
    Args:
        chord_ft: Chord length in feet
        speed_mph: Speed in miles per hour
    
    Returns:
        Re: Reynolds number
        Mach: Mach number
    """
    # Convert units
    chord_m = chord_ft * 0.3048  # feet to meters
    speed_mps = speed_mph * 0.44704  # mph to m/s
    
    # Standard atmospheric conditions at sea level
    rho = 1.225  # kg/m^3 (density)
    mu = 1.8e-5  # Pa·s (dynamic viscosity)
    a = 343.0  # m/s (speed of sound at sea level)
    
    # Compute Reynolds number: Re = ρ * V * c / μ
    Re = rho * speed_mps * chord_m / mu
    
    # Compute Mach number: M = V / a
    Mach = speed_mps / a
    
    return Re, Mach


def bin_reynolds_number(re):
    """
    Bin Reynolds number into predefined bins: 50k, 100k, 250k, 500k, 750k, 1m+
    
    Args:
        re: Reynolds number
    
    Returns:
        binned_re: Nearest bin value (or 1000000 for 1m+)
    """
    bins = [50000, 100000, 250000, 500000, 750000]
    if re >= 1000000:
        return 1000000  # 1m+ bin
    # Find nearest bin
    binned_re = min(bins, key=lambda x: abs(x - re))
    return binned_re


def get_max_cl_ld_for_re_bin(re_bin):
    """
    Get maximum Cl and L/D values for a given Reynolds number bin.
    This is used for setting graph scaling.
    
    Args:
        re_bin: Binned Reynolds number (50k, 100k, 250k, 500k, 750k, 1m+)
    
    Returns:
        max_cl: Maximum Cl value for this Re bin
        max_ld: Maximum L/D value for this Re bin
    """
    # Typical maximum values for different Reynolds number ranges
    # These are approximate based on typical airfoil performance
    # You may want to adjust these based on your data
    re_bin_ranges = {
        50000: {'max_cl': 1.2, 'max_ld': 80},
        100000: {'max_cl': 1.4, 'max_ld': 100},
        250000: {'max_cl': 1.6, 'max_ld': 120},
        500000: {'max_cl': 1.8, 'max_ld': 140},
        750000: {'max_cl': 2.0, 'max_ld': 160},
        1000000: {'max_cl': 2.2, 'max_ld': 180}  # 1m+ bin
    }
    
    if re_bin in re_bin_ranges:
        max_cl = re_bin_ranges[re_bin]['max_cl']
        max_ld = re_bin_ranges[re_bin]['max_ld']
    else:
        # Default values if bin not found
        max_cl = 1.5
        max_ld = 100
    
    return max_cl, max_ld


def create_alpha_vector(a_min, a_max, increment=0.5):
    """Create alpha vector from a_min to a_max with given increment."""
    num_points = int((a_max - a_min) / increment) + 1
    alpha = np.linspace(a_min, a_max, num_points)
    return alpha


def compute_cd_from_cl_clcd(cl_vector, cl_cd_vector):
    """Compute Cd vector from Cl and Cl/Cd vectors: Cd = Cl / (Cl/Cd)."""
    cd_vector = cl_vector / np.where(cl_cd_vector != 0, cl_cd_vector, 1e-10)
    return cd_vector


def compute_summary_statistics(alpha, cl, cd, cl_cd):
    """
    Compute summary statistics from aerodynamic vectors.
    
    Returns:
        clmax, cdmin, ldmax (cl/cd max), and their respective alpha values
    """
    clmax_idx = np.argmax(cl)
    clmax = cl[clmax_idx]
    alpha_clmax = alpha[clmax_idx]
    
    cdmin_idx = np.argmin(cd)
    cdmin = cd[cdmin_idx]
    alpha_cdmin = alpha[cdmin_idx]
    
    ldmax_idx = np.argmax(cl_cd)
    ldmax = cl_cd[ldmax_idx]
    alpha_ldmax = alpha[ldmax_idx]
    
    return {
        'clmax': clmax,
        'cdmin': cdmin,
        'ldmax': ldmax,
        'alpha_clmax': alpha_clmax,
        'alpha_cdmin': alpha_cdmin,
        'alpha_ldmax': alpha_ldmax
    }


def create_feature_dict(re, mach, alpha, cl, cd, cl_cd, min_thickness, max_thickness, stats):
    """Create feature dictionary in the format expected by the model."""
    scalar_features = np.array([
        re, mach, stats['ldmax'], stats['clmax'], stats['cdmin'], 
        stats['alpha_clmax'], stats['alpha_cdmin'], stats['alpha_ldmax'],
        min_thickness, max_thickness
    ], dtype=np.float32)
    
    # Sequence data: [alpha, cl, cd, cl_cd] as (seq_len, 4)
    sequence_data = np.column_stack([alpha, cl, cd, cl_cd])
    
    return {
        'scalars': scalar_features,
        'sequence': sequence_data
    }


def normalize_user_features(feature_dict, normalization_data):
    """
    Normalize user features using saved normalization data.
    If normalization_data is None, try to load from file.
    Raises error if normalization data cannot be found.
    """
    if normalization_data is None:
        norm_file = 'feature_normalization.npy'
        if os.path.exists(norm_file):
            norm_dict = np.load(norm_file, allow_pickle=True).item()
            min_vals = norm_dict['min']
            max_vals = norm_dict['max']
        else:
            raise FileNotFoundError(
                f"Normalization file '{norm_file}' not found. "
                "Please ensure you have trained the model and the normalization file exists."
            )
    else:
        min_vals = normalization_data['min']
        max_vals = normalization_data['max']
    
    ranges = max_vals - min_vals
    ranges[ranges < 1e-6] = 1.0
    
    normalized_scalars = (feature_dict['scalars'] - min_vals.flatten()) / ranges.flatten()
    
    normalized_features = {
        'scalars': normalized_scalars,
        'sequence': feature_dict['sequence']
    }
    
    return normalized_features, min_vals, max_vals


def compute_errors_for_correction(pred_af512_flat, feature_dict, normalization_data, device='cpu'):
    """
    Compute error statistics needed for correction model.
    Returns error stats array (11 values) or None if NeuralFoil fails.
    """
    try:
        import neuralfoil as nf
    except ImportError:
        return None
    
    try:
        # Convert predicted AF512 to coordinates
        pred_af512 = pred_af512_flat.reshape(512, 2)
        pred_x, pred_y = af512_to_coordinates(pred_af512)
        coordinates = np.column_stack([pred_x, pred_y])
        
        # Get Re and Mach (already unnormalized in feature_dict)
        scalars = feature_dict['scalars']
        norm_data = normalization_data
        
        # Scalars are unnormalized, so use directly
        Re = float(scalars[0])
        Mach = float(scalars[1])
        
        # Get alpha values from sequence
        sequence = feature_dict['sequence']
        alpha_values = sequence[:, 0]
        
        # Compute NeuralFoil values
        from scipy.interpolate import interp1d
        
        cl_list = []
        cd_list = []
        ld_list = []
        
        for alpha in alpha_values:
            try:
                nf_results = nf.get_aero_from_coordinates(
                    coordinates=coordinates,
                    alpha=float(alpha),
                    Re=float(Re),
                    model_size="xxxlarge"
                )
                
                pred_cl = nf_results['CL']
                pred_cd = nf_results['CD']
                
                if isinstance(pred_cl, np.ndarray):
                    pred_cl = float(pred_cl[0])
                if isinstance(pred_cd, np.ndarray):
                    pred_cd = float(pred_cd[0])
                
                pred_ld = pred_cl / pred_cd if pred_cd > 0 else 0
                
                cl_list.append(pred_cl)
                cd_list.append(pred_cd)
                ld_list.append(pred_ld)
            except Exception:
                cl_list.append(np.nan)
                cd_list.append(np.nan)
                ld_list.append(np.nan)
        
        nf_results_dict = {
            'cl': np.array(cl_list),
            'cd': np.array(cd_list),
            'ld': np.array(ld_list)
        }
        
        # Compute statistics from NeuralFoil results
        valid_mask = ~(np.isnan(nf_results_dict['cl']) | np.isnan(nf_results_dict['cd']) | np.isnan(nf_results_dict['ld']))
        if np.sum(valid_mask) == 0:
            return None
        
        alpha_valid = alpha_values[valid_mask]
        cl_valid = nf_results_dict['cl'][valid_mask]
        cd_valid = nf_results_dict['cd'][valid_mask]
        ld_valid = nf_results_dict['ld'][valid_mask]
        
        clmax_idx = np.argmax(cl_valid)
        clmax = cl_valid[clmax_idx]
        alpha_clmax = alpha_valid[clmax_idx]
        
        cdmin_idx = np.argmin(cd_valid)
        cdmin = cd_valid[cdmin_idx]
        alpha_cdmin = alpha_valid[cdmin_idx]
        
        ldmax_idx = np.argmax(ld_valid)
        ldmax = ld_valid[ldmax_idx]
        alpha_ldmax = alpha_valid[ldmax_idx]
        
        nf_stats = {
            'clmax': clmax,
            'cdmin': cdmin,
            'ldmax': ldmax,
            'alpha_clmax': alpha_clmax,
            'alpha_cdmin': alpha_cdmin,
            'alpha_ldmax': alpha_ldmax
        }
        
        # Target statistics are already unnormalized in scalars
        target_ldmax = float(scalars[2])
        target_clmax = float(scalars[3])
        target_cdmin = float(scalars[4])
        target_alpha_clmax = float(scalars[5])
        target_alpha_cdmin = float(scalars[6])
        target_alpha_ldmax = float(scalars[7])
        target_min_thickness = float(scalars[8])
        target_max_thickness = float(scalars[9])
        
        # Compute errors
        error_ldmax = nf_stats['ldmax'] - target_ldmax
        error_clmax = nf_stats['clmax'] - target_clmax
        error_cdmin = nf_stats['cdmin'] - target_cdmin
        error_alpha_clmax = nf_stats['alpha_clmax'] - target_alpha_clmax
        error_alpha_cdmin = nf_stats['alpha_cdmin'] - target_alpha_cdmin
        error_alpha_ldmax = nf_stats['alpha_ldmax'] - target_alpha_ldmax
        
        # Compute thickness errors
        min_x_idx = np.argmin(pred_x)
        upper_x = pred_x[:min_x_idx+1]
        upper_y = pred_y[:min_x_idx+1]
        lower_x = pred_x[min_x_idx:]
        lower_y = pred_y[min_x_idx:]
        upper_x = upper_x[::-1]
        upper_y = upper_y[::-1]
        x_common = np.linspace(0, 1, 100)
        upper_interp = interp1d(upper_x, upper_y, kind='linear', bounds_error=False, fill_value=np.nan)
        lower_interp = interp1d(lower_x, lower_y, kind='linear', bounds_error=False, fill_value=np.nan)
        upper_y_interp = upper_interp(x_common)
        lower_y_interp = lower_interp(x_common)
        thickness = upper_y_interp - lower_y_interp
        valid_mask_thick = ~np.isnan(thickness)
        if np.sum(valid_mask_thick) > 0:
            thickness_valid = thickness[valid_mask_thick]
            pred_min_thickness = np.min(thickness_valid)
            pred_max_thickness = np.max(thickness_valid)
        else:
            pred_min_thickness = 0.0
            pred_max_thickness = 0.0
        
        error_min_thickness = pred_min_thickness - target_min_thickness
        error_max_thickness = pred_max_thickness - target_max_thickness
        
        # Compute sequence errors
        target_cl = sequence[:, 1]
        target_cd = sequence[:, 2]
        target_cl_cd = sequence[:, 3]
        
        error_cl = nf_results_dict['cl'] - target_cl
        error_cd = nf_results_dict['cd'] - target_cd
        error_ld = nf_results_dict['ld'] - target_cl_cd
        
        # Create error stats array
        error_stats = np.array([
            error_ldmax,
            error_clmax,
            error_cdmin,
            error_alpha_clmax,
            error_alpha_cdmin,
            error_alpha_ldmax,
            error_min_thickness,
            error_max_thickness,
            np.nanmean(error_cl) if len(error_cl) > 0 else 0.0,
            np.nanmean(error_cd) if len(error_cd) > 0 else 0.0,
            np.nanmean(error_ld) if len(error_ld) > 0 else 0.0,
        ], dtype=np.float32)
        
        return np.nan_to_num(error_stats, nan=0.0)
    except Exception as e:
        return None


def apply_correction_model(correction_model, pred_af512_flat, error_stats, device='cpu'):
    """Apply correction model to predicted AF512."""
    if correction_model is None or error_stats is None:
        return pred_af512_flat
    
    correction_model.eval()
    with torch.no_grad():
        # Concatenate AF512 and error stats
        input_features = np.concatenate([pred_af512_flat, error_stats])
        input_tensor = torch.FloatTensor(input_features).unsqueeze(0).to(device)
        
        # Get corrected AF512
        corrected_af512 = correction_model(input_tensor)
        corrected_af512_flat = corrected_af512.cpu().numpy().flatten()
        
        return corrected_af512_flat


def run_pipeline(aero_model, af512_to_xy_model, feature_dict, normalization_data, 
                 correction_model=None, device='cpu', return_uncorrected=False):
    """
    Run the full pipeline: User inputs → Aero → AF512 → [Correction] → XY coordinates.
    Optimized for fast inference on small hardware.
    
    Args:
        return_uncorrected: If True and correction is applied, also return uncorrected coordinates
    
    Returns:
        pred_x, pred_y: Predicted airfoil coordinates (corrected if correction model available)
        If return_uncorrected=True and correction was applied, also returns (uncorrected_x, uncorrected_y)
    """
    # Normalize features
    normalized_features, min_vals, max_vals = normalize_user_features(feature_dict, normalization_data)
    
    # Convert to tensors with contiguous memory layout for better performance
    scalars_tensor = torch.FloatTensor(normalized_features['scalars']).unsqueeze(0).contiguous().to(device)
    sequence_tensor = torch.FloatTensor(normalized_features['sequence']).unsqueeze(0).contiguous().to(device)
    lengths_tensor = torch.tensor([len(normalized_features['sequence'])], dtype=torch.long).to(device)
    
    # Step 1: Aero → AF512
    aero_model.eval()
    # Disable gradient computation globally for faster inference
    torch.set_grad_enabled(False)
    try:
        pred_af512_tensor = aero_model(scalars_tensor, sequence_tensor, lengths_tensor)
        
        # Handle both quantized and regular models
        if isinstance(pred_af512_tensor, torch.Tensor):
            pred_af512_flat = pred_af512_tensor.cpu().numpy().flatten()
        else:
            pred_af512_flat = np.array(pred_af512_tensor).flatten()
    finally:
        # Re-enable gradients for other code
        torch.set_grad_enabled(True)
    
    # Step 2: Get uncorrected XY first if correction model is available and requested
    uncorrected_x = None
    uncorrected_y = None
    if return_uncorrected and correction_model is not None:
        # Save uncorrected coordinates before any correction
        uncorrected_x, uncorrected_y = predict_xy_coordinates(af512_to_xy_model, pred_af512_flat, device=device)
    
    # Step 3: Apply correction if available
    correction_applied = False
    if correction_model is not None:
        error_stats = compute_errors_for_correction(
            pred_af512_flat, feature_dict, normalization_data, device
        )
        if error_stats is not None:
            corrected_af512 = apply_correction_model(
                correction_model, pred_af512_flat, error_stats, device
            )
            pred_af512_flat = corrected_af512
            correction_applied = True
    
    # Step 4: AF512 → XY
    pred_x, pred_y = predict_xy_coordinates(af512_to_xy_model, pred_af512_flat, device=device)
    
    # Return both corrected and uncorrected if we have uncorrected data
    # (even if correction wasn't applied, we still want to show comparison if correction model exists)
    if return_uncorrected and uncorrected_x is not None:
        return pred_x, pred_y, uncorrected_x, uncorrected_y
    return pred_x, pred_y


def save_dat_file(x_coords, y_coords, filename):
    """Save airfoil coordinates to .dat file format."""
    with open(filename, 'w') as f:
        f.write("Generated Airfoil\n")
        for x, y in zip(x_coords, y_coords):
            f.write(f"{x:.6f} {y:.6f}\n")
    print(f"Saved airfoil to {filename}")


def plot_airfoil_shape(x_coords, y_coords, save_path=None):
    """Plot the airfoil shape."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(x_coords, y_coords, 'b-', linewidth=2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (chord fraction)')
    ax.set_ylabel('Y (chord fraction)')
    ax.set_title('Generated Airfoil Shape')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved airfoil shape plot to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_aerodynamic_curves(alpha, cl, cd, cl_cd, save_path=None, nf_results=None):
    """Plot Cl, Cd, and Cl/Cd vs alpha. Optionally overlay NeuralFoil results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    alpha_arr = np.array(alpha)
    
    # Cl vs alpha
    axes[0].plot(alpha_arr, cl, 'b-o', label='User Input', linewidth=2, markersize=4)
    if nf_results is not None:
        nf_cl = nf_results.get('nf_cl', None)
        if nf_cl is not None:
            valid_mask = ~np.isnan(nf_cl)
            if np.sum(valid_mask) > 0:
                axes[0].plot(alpha_arr[valid_mask], nf_cl[valid_mask], 'r--s', 
                            label='NeuralFoil', linewidth=2, markersize=4, alpha=0.7)
    axes[0].set_xlabel('Angle of Attack (deg)')
    axes[0].set_ylabel('Cl')
    axes[0].set_title('Lift Coefficient vs AoA')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Cd vs alpha
    axes[1].plot(alpha_arr, cd, 'r-o', label='User Input', linewidth=2, markersize=4)
    if nf_results is not None:
        nf_cd = nf_results.get('nf_cd', None)
        if nf_cd is not None:
            valid_mask = ~np.isnan(nf_cd)
            if np.sum(valid_mask) > 0:
                axes[1].plot(alpha_arr[valid_mask], nf_cd[valid_mask], 'b--s', 
                            label='NeuralFoil', linewidth=2, markersize=4, alpha=0.7)
    axes[1].set_xlabel('Angle of Attack (deg)')
    axes[1].set_ylabel('Cd')
    axes[1].set_title('Drag Coefficient vs AoA')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Cl/Cd vs alpha
    axes[2].plot(alpha_arr, cl_cd, 'g-o', label='User Input', linewidth=2, markersize=4)
    if nf_results is not None:
        nf_ld = nf_results.get('nf_ld', None)
        if nf_ld is not None:
            valid_mask = ~np.isnan(nf_ld)
            if np.sum(valid_mask) > 0:
                axes[2].plot(alpha_arr[valid_mask], nf_ld[valid_mask], 'm--s', 
                            label='NeuralFoil', linewidth=2, markersize=4, alpha=0.7)
    axes[2].set_xlabel('Angle of Attack (deg)')
    axes[2].set_ylabel('Cl/Cd')
    axes[2].set_title('Lift-to-Drag Ratio vs AoA')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved aerodynamic curves to {save_path}")
    else:
        plt.show()
    plt.close()


def run_neuralfoil_comparison(x_coords, y_coords, alpha, cl_input, cd_input, cl_cd_input, 
                               re, mach, save_dir='output'):
    """Run NeuralFoil analysis and compare with user inputs."""
    if not NEURALFOIL_AVAILABLE:
        print("NeuralFoil not available. Skipping comparison.")
        return None
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare coordinates for NeuralFoil (Nx2 array)
    coordinates = np.column_stack([x_coords, y_coords])
    
    nf_cl_list = []
    nf_cd_list = []
    nf_ld_list = []
    errors = []
    
    print("\nRunning NeuralFoil analysis (xxxlarge model)...")
    
    for i, alpha_val in enumerate(tqdm(alpha, desc="NeuralFoil analysis")):
        try:
            nf_results = nf.get_aero_from_coordinates(
                coordinates=coordinates,
                alpha=float(alpha_val),
                Re=re,
                model_size="xxxlarge"
            )
            
            # Extract CL and CD
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
            
        except Exception as e:
            errors.append(f"Alpha {alpha_val}: {str(e)}")
            nf_cl_list.append(np.nan)
            nf_cd_list.append(np.nan)
            nf_ld_list.append(np.nan)
    
    if len(errors) > 0:
        print(f"\nWarnings ({len(errors)} errors):")
        for err in errors[:5]:
            print(f"  {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    
    nf_cl_arr = np.array(nf_cl_list)
    nf_cd_arr = np.array(nf_cd_list)
    nf_ld_arr = np.array(nf_ld_list)
    
    # Calculate errors
    valid_mask = ~(np.isnan(nf_cl_arr) | np.isnan(nf_cd_arr))
    if np.sum(valid_mask) == 0:
        print("No valid NeuralFoil results!")
        return None
    
    cl_error = nf_cl_arr[valid_mask] - np.array(cl_input)[valid_mask]
    cd_error = nf_cd_arr[valid_mask] - np.array(cd_input)[valid_mask]
    ld_error = nf_ld_arr[valid_mask] - np.array(cl_cd_input)[valid_mask]
    
    cl_mae = np.mean(np.abs(cl_error))
    cd_mae = np.mean(np.abs(cd_error))
    ld_mae = np.mean(np.abs(ld_error))
    
    print(f"\n{'='*60}")
    print("NeuralFoil Comparison Results:")
    print(f"{'='*60}")
    print(f"CL MAE: {cl_mae:.6f}")
    print(f"CD MAE: {cd_mae:.6f}")
    print(f"L/D MAE: {ld_mae:.6f}")
    print(f"{'='*60}\n")
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    alpha_arr = np.array(alpha)
    
    # Cl comparison
    axes[0, 0].plot(alpha_arr, cl_input, 'b-o', label='User Input', linewidth=2, markersize=4)
    axes[0, 0].plot(alpha_arr[valid_mask], nf_cl_arr[valid_mask], 'r--s', label='NeuralFoil', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Angle of Attack (deg)')
    axes[0, 0].set_ylabel('Cl')
    axes[0, 0].set_title('Cl Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cd comparison
    axes[0, 1].plot(alpha_arr, cd_input, 'b-o', label='User Input', linewidth=2, markersize=4)
    axes[0, 1].plot(alpha_arr[valid_mask], nf_cd_arr[valid_mask], 'r--s', label='NeuralFoil', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Angle of Attack (deg)')
    axes[0, 1].set_ylabel('Cd')
    axes[0, 1].set_title('Cd Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # L/D comparison
    axes[0, 2].plot(alpha_arr, cl_cd_input, 'b-o', label='User Input', linewidth=2, markersize=4)
    axes[0, 2].plot(alpha_arr[valid_mask], nf_ld_arr[valid_mask], 'r--s', label='NeuralFoil', linewidth=2, markersize=4)
    axes[0, 2].set_xlabel('Angle of Attack (deg)')
    axes[0, 2].set_ylabel('Cl/Cd')
    axes[0, 2].set_title('L/D Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Error distributions
    axes[1, 0].hist(cl_error, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Cl Error (NeuralFoil - Input)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Cl Error Distribution (MAE: {cl_mae:.6f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(cd_error, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Cd Error (NeuralFoil - Input)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Cd Error Distribution (MAE: {cd_mae:.6f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].hist(ld_error, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 2].set_xlabel('L/D Error (NeuralFoil - Input)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title(f'L/D Error Distribution (MAE: {ld_mae:.6f})')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'neuralfoil_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved NeuralFoil comparison to {save_path}")
    plt.close()
    
    return {
        'nf_cl': nf_cl_arr,
        'nf_cd': nf_cd_arr,
        'nf_ld': nf_ld_arr,
        'cl_mae': cl_mae,
        'cd_mae': cd_mae,
        'ld_mae': ld_mae
    }


class InteractivePlot:
    """Interactive matplotlib plot for entering points by typing."""
    def __init__(self, title, xlabel, ylabel, alpha_range, y_max=None, y_min=0):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.alpha_range = alpha_range
        self.y_max = y_max if y_max is not None else 2.0
        self.y_min = y_min
        
        # Store control points as list of (alpha, value) tuples
        self.points = []
        
    def update_plot(self):
        """Update the plot with current points and interpolated curve."""
        plt.figure(figsize=(12, 8))
        plt.title(f"{self.title}", fontsize=14, fontweight='bold')
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.grid(True, alpha=0.3)
        plt.xlim(self.alpha_range[0] - 1, self.alpha_range[1] + 1)
        plt.ylim(self.y_min, self.y_max * 1.15)
        
        if len(self.points) > 0:
            # Remove duplicates by alpha for display
            unique_points = {}
            for alpha, value in self.points:
                unique_points[alpha] = value
            sorted_points = sorted(unique_points.items())
            alphas = np.array([p[0] for p in sorted_points])
            values = np.array([p[1] for p in sorted_points])
            
            # Plot control points
            plt.plot(alphas, values, 'bo', markersize=10, label='Control Points', zorder=5)
            
            # Plot spline interpolation if we have enough points
            if len(sorted_points) >= 2:
                # Create smooth curve for display
                smooth_alpha = np.linspace(self.alpha_range[0], self.alpha_range[1], 200)
                try:
                    interp_values = self._interpolate_points(smooth_alpha)
                    plt.plot(smooth_alpha, interp_values, 'g-', linewidth=2, 
                               alpha=0.8, label='Spline Interpolation', zorder=2)
                except Exception as e:
                    # Fallback to linear if spline fails
                    try:
                        interp_func = interp1d(alphas, values, kind='linear', 
                                              bounds_error=False, fill_value='extrapolate')
                        interp_values = interp_func(smooth_alpha)
                        plt.plot(smooth_alpha, interp_values, 'g--', linewidth=2, 
                                   alpha=0.7, label='Linear Interpolation (fallback)', zorder=2)
                    except:
                        pass
            elif len(sorted_points) == 1:
                # Single point - show horizontal line
                plt.axhline(y=values[0], color='g', linestyle='--', 
                               alpha=0.5, label='Constant value', zorder=2)
        
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)  # Brief pause to ensure plot updates
    
    def _interpolate_points(self, alpha_vector):
        """Interpolate using control points with splines."""
        if len(self.points) == 0:
            raise ValueError("No points defined")
        
        # Remove duplicates by alpha, keeping the last value for each alpha
        unique_points = {}
        for alpha, value in self.points:
            unique_points[alpha] = value
        sorted_points = sorted(unique_points.items())
        alphas = np.array([p[0] for p in sorted_points])
        values = np.array([p[1] for p in sorted_points])
        
        alpha_vector = np.array(alpha_vector)
        
        if len(sorted_points) == 1:
            return np.full_like(alpha_vector, values[0])
        
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
        
        return interpolated
    
    def interpolate(self, alpha_vector):
        """Interpolate values for given alpha vector using splines."""
        if len(self.points) == 0:
            raise ValueError("No points have been added. Please add at least one point.")
        
        return self._interpolate_points(alpha_vector)
    
    def show(self):
        """Show the plot and prompt for point input."""
        print(f"\nInteractive plot: {self.title}")
        print(f"  Y-axis range: {self.y_min:.2f} to {self.y_max:.2f}")
        print(f"  Alpha range: {self.alpha_range[0]:.1f} to {self.alpha_range[1]:.1f}")
        print("\nEnter points as 'alpha,value' (e.g., '5.0,1.2')")
        print("  - You don't need to enter all points - spline will interpolate missing ones")
        print("  - Type 'done' or 'd' when finished")
        print("  - Type 'clear' or 'c' to clear all points")
        print("  - Type 'list' or 'l' to see current points")
        print("  - Type 'plot' or 'p' to update the plot")
        print("  - Type 'delete <index>' or 'del <index>' to remove a point by index")
        print()
        
        # Show initial empty plot
        self.update_plot()
        
        while True:
            user_input = input(f"Enter point (alpha,value) or command [{len(self.points)} points so far]: ").strip()
            
            if not user_input:
                continue
            
            user_input_lower = user_input.lower()
            
            # Check for commands
            if user_input_lower in ['done', 'd', 'q', 'quit']:
                if len(self.points) == 0:
                    print("  Warning: No points entered! Please add at least one point.")
                    continue
                break
            elif user_input_lower in ['clear', 'c']:
                self.points = []
                print("  Cleared all points.")
                self.update_plot()
            elif user_input_lower in ['list', 'l']:
                if len(self.points) == 0:
                    print("  No points yet.")
                else:
                    print("  Current points:")
                    for i, (alpha, value) in enumerate(self.points):
                        print(f"    [{i}] alpha={alpha:.3f}, value={value:.6f}")
            elif user_input_lower in ['plot', 'p']:
                self.update_plot()
            elif user_input_lower.startswith('delete ') or user_input_lower.startswith('del '):
                try:
                    idx_str = user_input_lower.split()[-1]
                    idx = int(idx_str)
                    if 0 <= idx < len(self.points):
                        removed = self.points.pop(idx)
                        print(f"  Removed point: alpha={removed[0]:.3f}, value={removed[1]:.6f}")
                        self.update_plot()
                    else:
                        print(f"  Error: Index {idx} out of range (0-{len(self.points)-1})")
                except ValueError:
                    print("  Error: Invalid index. Use 'delete <index>' or 'del <index>'")
            else:
                # Try to parse as alpha,value pair
                try:
                    parts = user_input.split(',')
                    if len(parts) != 2:
                        raise ValueError("Need exactly two values")
                    
                    alpha = float(parts[0].strip())
                    value = float(parts[1].strip())
                    
                    # Check if alpha is in range (optional check)
                    if alpha < self.alpha_range[0] - 5 or alpha > self.alpha_range[1] + 5:
                        response = input(f"  Warning: Alpha ({alpha}) is outside typical range [{self.alpha_range[0]}, {self.alpha_range[1]}]. Continue? (y/n): ")
                        if response.lower() != 'y':
                            continue
                    
                    # Check if a point with this alpha already exists (within tolerance)
                    tolerance = 1e-6
                    existing_idx = None
                    for i, (existing_alpha, _) in enumerate(self.points):
                        if abs(existing_alpha - alpha) < tolerance:
                            existing_idx = i
                            break
                    
                    if existing_idx is not None:
                        # Replace existing point
                        old_value = self.points[existing_idx][1]
                        self.points[existing_idx] = (alpha, value)
                        print(f"  Replaced point: alpha={alpha:.3f}, value={old_value:.6f} -> {value:.6f}")
                    else:
                        # Add new point
                        self.points.append((alpha, value))
                        print(f"  Added point: alpha={alpha:.3f}, value={value:.6f}")
                    
                    # Sort by alpha to keep points ordered
                    self.points.sort(key=lambda x: x[0])
                    
                    self.update_plot()
                    
                except ValueError as e:
                    print(f"  Error: Could not parse '{user_input}'. Expected format: 'alpha,value' (e.g., '5.0,1.2')")
        
        plt.close('all')  # Close all plots when done
        return self


def get_user_input():
    """Get all user inputs interactively."""
    print("="*60)
    print("Interactive Airfoil Design Tool")
    print("="*60)
    
    # Get chord and speed
    print("\n1. Enter flight conditions:")
    chord_ft = float(input("   Chord length (ft): "))
    speed_mph = float(input("   Speed (mph): "))
    
    # Compute Re and Mach
    Re, Mach = compute_reynolds_mach(chord_ft, speed_mph)
    print(f"\n   Computed Re: {Re:.2e}")
    print(f"   Computed Mach: {Mach:.4f}")
    
    # Get angle range
    print("\n2. Enter angle of attack range:")
    a_min = float(input("   Minimum angle (deg): "))
    a_max = float(input("   Maximum angle (deg): "))
    increment = 0.5
    alpha = create_alpha_vector(a_min, a_max, increment)
    print(f"   Generated {len(alpha)} angles from {a_min} to {a_max} deg (increment: {increment})")
    
    # Bin Re for graph scaling (but still use actual Re for calculations)
    re_bin = bin_reynolds_number(Re)
    max_cl, max_ld = get_max_cl_ld_for_re_bin(re_bin)
    print(f"\n   Re bin for graph scaling: {re_bin:.0f}")
    print(f"   Graph scaling - Max Cl: {max_cl:.2f}, Max L/D: {max_ld:.2f}")
    
    # Get Cl vector using interactive plot with Re-binned scaling
    print("\n3. Enter Cl values (lift coefficient) - Interactive Plot")
    cl_plot = InteractivePlot("Cl vs Angle of Attack", "Angle of Attack (deg)", "Cl", 
                              (a_min, a_max), y_max=max_cl * 1.15, y_min=0)
    cl_plot.show()
    
    cl = cl_plot.interpolate(alpha)
    print(f"   Interpolated {len(cl)} Cl values from control points")
    
    # Get Cl/Cd vector using interactive plot with Re-binned scaling
    print("\n4. Enter Cl/Cd values (lift-to-drag ratio) - Interactive Plot")
    cl_cd_plot = InteractivePlot("Cl/Cd vs Angle of Attack", "Angle of Attack (deg)", "Cl/Cd", 
                                 (a_min, a_max), y_max=max_ld * 1.15, y_min=0)
    cl_cd_plot.show()
    
    cl_cd = cl_cd_plot.interpolate(alpha)
    print(f"   Interpolated {len(cl_cd)} Cl/Cd values from control points")
    
    # Compute Cd from Cl and Cl/Cd
    cd = compute_cd_from_cl_clcd(cl, cl_cd)
    print(f"\n   Computed Cd vector (Cl / Cl_Cd)")
    
    # Compute summary statistics
    stats = compute_summary_statistics(alpha, cl, cd, cl_cd)
    print("\n5. Summary Statistics:")
    print(f"   ClMax: {stats['clmax']:.6f} at alpha = {stats['alpha_clmax']:.4f} deg")
    print(f"   CdMin: {stats['cdmin']:.6f} at alpha = {stats['alpha_cdmin']:.4f} deg")
    print(f"   L/D Max: {stats['ldmax']:.6f} at alpha = {stats['alpha_ldmax']:.4f} deg")
    
    # Get thickness values
    print("\n6. Enter thickness values:")
    min_thickness = float(input("   Minimum thickness (chord fraction): "))
    max_thickness = float(input("   Maximum thickness (chord fraction): "))
    
    return {
        're': Re,
        'mach': Mach,
        'alpha': alpha,
        'cl': cl,
        'cd': cd,
        'cl_cd': cl_cd,
        'min_thickness': min_thickness,
        'max_thickness': max_thickness,
        'stats': stats
    }


def main():
    # Force CPU-only
    device = torch.device('cpu')
    
    print(f"\nUsing device: {device} (CPU-only)\n")
    
    # Load models
    aero_model_file = 'aero_to_af512_model.pth'
    af512_to_xy_model_file = 'af512_to_xy_model.pth'
    
    if not os.path.exists(aero_model_file):
        print(f"Error: Model file {aero_model_file} not found!")
        return
    
    if not os.path.exists(af512_to_xy_model_file):
        print(f"Error: Model file {af512_to_xy_model_file} not found!")
        return
    
    # Load AeroToAF512 model
    print(f"Loading AeroToAF512 model...")
    
    # First, load the state dict to inspect architecture
    model_state = torch.load(aero_model_file, map_location='cpu')
    
    # Check if sequence encoder exists in saved model
    has_lstm = any('sequence_encoder.lstm' in key for key in model_state.keys())
    has_projection = 'sequence_encoder.projection.0.weight' in model_state
    
    # Determine architecture from saved weights
    first_layer_key = 'network.0.weight'
    if first_layer_key in model_state:
        first_layer_weight = model_state[first_layer_key]
        combined_input_size = first_layer_weight.shape[1]
        first_hidden_size = first_layer_weight.shape[0]
        
        # Determine hidden sizes by inspecting all network layers
        hidden_sizes = []
        i = 0
        while f'network.{i*3}.weight' in model_state:
            weight_key = f'network.{i*3}.weight'
            hidden_size = model_state[weight_key].shape[0]
            hidden_sizes.append(hidden_size)
            i += 1
        
        # Remove the last layer (output layer)
        if len(hidden_sizes) > 0:
            hidden_sizes = hidden_sizes[:-1]
        
        # Try to determine sequence_embedding_dim
        sequence_embedding_dim = 256  # default
        if has_projection:
            sequence_embedding_dim = model_state['sequence_encoder.projection.0.weight'].shape[0]
        elif not has_lstm:
            # If no LSTM, maybe the model doesn't use sequence encoder
            # In this case, sequence_embedding_dim might be 0 or very small
            # Let's assume it's the difference from expected
            if combined_input_size <= 14:
                # Might be an old model without sequence encoder
                sequence_embedding_dim = 4  # small embedding or no embedding
            else:
                sequence_embedding_dim = combined_input_size - 10  # assume 10 scalar features
        
        # scalar_input_size = combined_input_size - sequence_embedding_dim
        scalar_input_size = combined_input_size - sequence_embedding_dim
        
        # Determine output size from last layer
        output_size = 1024  # default
        last_layer_idx = len(hidden_sizes) * 3
        if f'network.{last_layer_idx}.weight' in model_state:
            output_size = model_state[f'network.{last_layer_idx}.weight'].shape[0]
        
        print(f"  Inspecting saved model...")
        print(f"    Has LSTM encoder: {has_lstm}")
        print(f"    Has projection: {has_projection}")
        print(f"    Combined input size: {combined_input_size}")
        
        # If LSTM keys are missing, this model might be incompatible
        if not has_lstm and combined_input_size != 14:
            print(f"  Warning: Model appears to be missing sequence encoder components!")
            print(f"    Attempting to load anyway...")
        
        print(f"  Detected architecture:")
        print(f"    scalar_input_size: {scalar_input_size}")
        print(f"    sequence_embedding_dim: {sequence_embedding_dim}")
        print(f"    output_size: {output_size}")
        print(f"    hidden_sizes: {hidden_sizes}")
        
        # Create model with detected architecture
        aero_model = AeroToAF512Net(
            scalar_input_size=scalar_input_size,
            sequence_embedding_dim=sequence_embedding_dim,
            output_size=output_size,
            hidden_sizes=hidden_sizes if hidden_sizes else [256, 256, 256, 256]
        )
    else:
        # Fallback to defaults if we can't determine architecture
        print("  Warning: Could not determine architecture from saved model. Using defaults.")
        aero_model = AeroToAF512Net(
            scalar_input_size=10,
            sequence_embedding_dim=256,
            output_size=1024,
            hidden_sizes=[256, 256, 256, 256]
        )
    
    # Try to load with strict=False to handle missing keys
    missing_keys = []
    unexpected_keys = []
    try:
        aero_model.load_state_dict(model_state, strict=True)
    except RuntimeError as e:
        print(f"  Error loading with strict=True: {e}")
        print(f"  Attempting to load with strict=False (will skip missing keys)...")
        missing_keys, unexpected_keys = aero_model.load_state_dict(model_state, strict=False)
        if missing_keys:
            print(f"  Missing keys: {len(missing_keys)} keys missing")
            if any('sequence_encoder' in key for key in missing_keys):
                print(f"  ⚠️  WARNING: Sequence encoder components are missing!")
                print(f"     The model will use randomly initialized sequence encoder weights.")
                print(f"     This may affect prediction quality. Consider retraining the model.")
        if unexpected_keys:
            print(f"  Unexpected keys: {len(unexpected_keys)} unexpected keys")
    
    aero_model = aero_model.to(device)
    if missing_keys and any('sequence_encoder' in key for key in missing_keys):
        print("  ⚠️  Model loaded with missing sequence encoder - predictions may be inaccurate!")
    else:
        print("AeroToAF512 model loaded successfully")
    
    # Load AF512toXY model
    print(f"Loading AF512toXY model...")
    af512_to_xy_model = AF512toXYNet(
        input_size=1024,
        output_size=2048,
        hidden_sizes=[256, 256, 256, 256]
    )
    af512_to_xy_model.load_state_dict(torch.load(af512_to_xy_model_file, map_location='cpu'))
    af512_to_xy_model = af512_to_xy_model.to(device)
    print("AF512toXY model loaded successfully\n")
    
    # Load normalization data
    norm_file = 'feature_normalization.npy'
    normalization_data = None
    if os.path.exists(norm_file):
        norm_dict = np.load(norm_file, allow_pickle=True).item()
        normalization_data = norm_dict
        print("Normalization data loaded successfully\n")
    else:
        print(f"Error: Normalization file '{norm_file}' not found!")
        print("This file is required for proper feature normalization.")
        print("Please ensure you have trained the model and the normalization file exists.\n")
        return
    
    # Get user inputs
    user_data = get_user_input()
    
    # Create feature dictionary
    feature_dict = create_feature_dict(
        user_data['re'], user_data['mach'], user_data['alpha'],
        user_data['cl'], user_data['cd'], user_data['cl_cd'],
        user_data['min_thickness'], user_data['max_thickness'],
        user_data['stats']
    )
    
    # Run pipeline
    print("\n" + "="*60)
    print("Running Pipeline: Aero → AF512 → XY")
    print("="*60)
    pred_x, pred_y = run_pipeline(
        aero_model, af512_to_xy_model, feature_dict, 
        normalization_data, device=device
    )
    print("Pipeline completed successfully!")
    
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save .dat file
    dat_path = os.path.join(output_dir, 'generated_airfoil.dat')
    save_dat_file(pred_x, pred_y, dat_path)
    
    # Run NeuralFoil comparison first (so we can overlay results)
    nf_results = None
    if NEURALFOIL_AVAILABLE:
        print("\n" + "="*60)
        print("Running NeuralFoil Comparison")
        print("="*60)
        nf_results = run_neuralfoil_comparison(
            pred_x, pred_y, user_data['alpha'], user_data['cl'], 
            user_data['cd'], user_data['cl_cd'], user_data['re'], 
            user_data['mach'], save_dir=output_dir
        )
    
    # Generate plots (with NeuralFoil overlay if available)
    print("\nGenerating plots...")
    plot_airfoil_shape(pred_x, pred_y, save_path=os.path.join(output_dir, 'airfoil_shape.png'))
    plot_aerodynamic_curves(
        user_data['alpha'], user_data['cl'], user_data['cd'], user_data['cl_cd'],
        save_path=os.path.join(output_dir, 'aerodynamic_curves.png'),
        nf_results=nf_results
        )
    
    print("\n" + "="*60)
    print("All outputs saved to 'output' directory!")
    print("="*60)


if __name__ == "__main__":
    main()
