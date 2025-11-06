"""
Train a correction model that takes predicted AF512 and error values,
and outputs corrected AF512.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import csv
import signal
import sys
from scipy.interpolate import interp1d
from tqdm import tqdm

try:
    import neuralfoil as nf
    NEURALFOIL_AVAILABLE = True
except ImportError:
    NEURALFOIL_AVAILABLE = False
    print("Warning: NeuralFoil not installed. Install with: pip install neuralfoil")

# Import functions from training scripts
from train_airfoil import (
    load_dat_file,
    af512_to_coordinates
)
from train_aero_to_af512 import (
    load_csv_data,
    AeroToAF512Net,
    normalize_features,
    collate_fn,
    predict_af512
)


def load_second_half_data(csv_dir='unpacked_csv', bigfoil_dir='bigfoil', num_points=512):
    """Load the second half of the dataset."""
    features, af512_data = load_csv_data(csv_dir, bigfoil_dir, num_points)
    
    if features is None or len(features) == 0:
        return None, None
    
    total_size = len(features)
    half_size = total_size // 2
    
    # Get second half
    features = features[half_size:]
    af512_data = af512_data[half_size:]
    
    print(f"Loaded second half of dataset: {len(features)} samples (from index {half_size} to {total_size-1})")
    return features, af512_data


def compute_neuralfoil_values(coordinates, alpha_values, Re, model_size="xxxlarge"):
    """
    Compute aerodynamic values using NeuralFoil for all alpha values.
    
    Returns:
        Dict with keys: 'cl', 'cd', 'ld' (all as arrays)
    """
    cl_list = []
    cd_list = []
    ld_list = []
    
    for alpha in alpha_values:
        try:
            nf_results = nf.get_aero_from_coordinates(
                coordinates=coordinates,
                alpha=float(alpha),
                Re=float(Re),
                model_size=model_size
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
        except Exception as e:
            cl_list.append(np.nan)
            cd_list.append(np.nan)
            ld_list.append(np.nan)
    
    return {
        'cl': np.array(cl_list),
        'cd': np.array(cd_list),
        'ld': np.array(ld_list)
    }


def compute_statistics(alpha, cl, cd, ld):
    """Compute summary statistics from aerodynamic vectors."""
    valid_mask = ~(np.isnan(cl) | np.isnan(cd) | np.isnan(ld))
    if np.sum(valid_mask) == 0:
        return None
    
    alpha_valid = alpha[valid_mask]
    cl_valid = cl[valid_mask]
    cd_valid = cd[valid_mask]
    ld_valid = ld[valid_mask]
    
    clmax_idx = np.argmax(cl_valid)
    clmax = cl_valid[clmax_idx]
    alpha_clmax = alpha_valid[clmax_idx]
    
    cdmin_idx = np.argmin(cd_valid)
    cdmin = cd_valid[cdmin_idx]
    alpha_cdmin = alpha_valid[cdmin_idx]
    
    ldmax_idx = np.argmax(ld_valid)
    ldmax = ld_valid[ldmax_idx]
    alpha_ldmax = alpha_valid[ldmax_idx]
    
    return {
        'clmax': clmax,
        'cdmin': cdmin,
        'ldmax': ldmax,
        'alpha_clmax': alpha_clmax,
        'alpha_cdmin': alpha_cdmin,
        'alpha_ldmax': alpha_ldmax
    }


def compute_thickness(x_coords, y_coords):
    """Compute min and max thickness from airfoil coordinates."""
    # Find leading edge (minimum x)
    min_x_idx = np.argmin(x_coords)
    
    # Split into upper and lower surfaces
    upper_x = x_coords[:min_x_idx+1]
    upper_y = y_coords[:min_x_idx+1]
    lower_x = x_coords[min_x_idx:]
    lower_y = y_coords[min_x_idx:]
    
    # Reverse upper surface to match x ordering
    upper_x = upper_x[::-1]
    upper_y = upper_y[::-1]
    
    # Interpolate to common x coordinates
    x_common = np.linspace(0, 1, 100)
    upper_interp = interp1d(upper_x, upper_y, kind='linear', bounds_error=False, fill_value=np.nan)
    lower_interp = interp1d(lower_x, lower_y, kind='linear', bounds_error=False, fill_value=np.nan)
    
    upper_y_interp = upper_interp(x_common)
    lower_y_interp = lower_interp(x_common)
    
    # Compute thickness at each x
    thickness = upper_y_interp - lower_y_interp
    
    # Remove NaN values
    valid_mask = ~np.isnan(thickness)
    if np.sum(valid_mask) == 0:
        return 0.0, 0.0
    
    thickness_valid = thickness[valid_mask]
    min_thickness = np.min(thickness_valid)
    max_thickness = np.max(thickness_valid)
    
    return min_thickness, max_thickness


def generate_correction_data(aero_model, features, target_af512, normalization_data, device='cpu'):
    """
    Generate correction data by:
    1. Running the trained model to get predicted AF512
    2. Converting to coordinates
    3. Using NeuralFoil to compute actual values
    4. Computing errors
    """
    if not NEURALFOIL_AVAILABLE:
        raise RuntimeError("NeuralFoil is required for this script")
    
    print("Generating correction data...")
    
    # Limit to first 1000 samples for faster processing
    num_samples = min(1000, len(features))
    print(f"Processing {num_samples} samples (limited from {len(features)} total)")
    
    correction_data = []
    
    for idx in tqdm(range(num_samples), desc="Processing samples"):
        try:
            feature_dict = features[idx]
            scalars = feature_dict['scalars']
            sequence = feature_dict['sequence']
            
            # Get target AF512
            target_af512_flat = target_af512[idx]
            
            # Predict AF512 using trained model
            pred_af512_flat = predict_af512(aero_model, scalars, sequence, device=device)
            
            # Convert predicted AF512 to coordinates
            pred_af512 = pred_af512_flat.reshape(512, 2)
            pred_x, pred_y = af512_to_coordinates(pred_af512)
            coordinates = np.column_stack([pred_x, pred_y])
            
            # Denormalize Re and Mach for NeuralFoil
            norm_data = normalization_data
            re_norm = scalars[0]
            mach_norm = scalars[1]
            
            re_min = norm_data['min'][0, 0]
            re_max = norm_data['max'][0, 0]
            re_range = re_max - re_min
            Re = re_norm * re_range + re_min if re_range > 1e-6 else re_min
            
            mach_min = norm_data['min'][0, 1]
            mach_max = norm_data['max'][0, 1]
            mach_range = mach_max - mach_min
            Mach = mach_norm * mach_range + mach_min if mach_range > 1e-6 else mach_min
            
            # Get alpha values from sequence
            alpha_values = sequence[:, 0]
            
            # Compute NeuralFoil values
            nf_results = compute_neuralfoil_values(coordinates, alpha_values, Re)
            
            # Get target values from feature dict (need to denormalize)
            # Target values are in the original feature dict but we need to extract them
            # For now, we'll compute statistics from target and compare
            
            # Compute statistics from NeuralFoil results
            nf_stats = compute_statistics(alpha_values, nf_results['cl'], nf_results['cd'], nf_results['ld'])
            
            if nf_stats is None:
                continue
            
            # Get target statistics from normalized features (need to denormalize)
            # The features contain: re, mach, ldmax, clmax, cdmin, alpha_clmax, alpha_cdmin, alpha_ldmax, min_thickness, max_thickness
            target_ldmax_norm = scalars[2]
            target_clmax_norm = scalars[3]
            target_cdmin_norm = scalars[4]
            target_alpha_clmax_norm = scalars[5]
            target_alpha_cdmin_norm = scalars[6]
            target_alpha_ldmax_norm = scalars[7]
            target_min_thickness_norm = scalars[8]
            target_max_thickness_norm = scalars[9]
            
            # Denormalize target statistics
            def denormalize(val, idx, norm_data):
                min_val = norm_data['min'][0, idx]
                max_val = norm_data['max'][0, idx]
                range_val = max_val - min_val
                return val * range_val + min_val if range_val > 1e-6 else min_val
            
            target_ldmax = denormalize(target_ldmax_norm, 2, norm_data)
            target_clmax = denormalize(target_clmax_norm, 3, norm_data)
            target_cdmin = denormalize(target_cdmin_norm, 4, norm_data)
            target_min_thickness = denormalize(target_min_thickness_norm, 8, norm_data)
            target_max_thickness = denormalize(target_max_thickness_norm, 9, norm_data)
            
            # Compute errors (simple subtraction)
            error_ldmax = nf_stats['ldmax'] - target_ldmax
            error_clmax = nf_stats['clmax'] - target_clmax
            error_cdmin = nf_stats['cdmin'] - target_cdmin
            error_alpha_clmax = nf_stats['alpha_clmax'] - denormalize(target_alpha_clmax_norm, 5, norm_data)
            error_alpha_cdmin = nf_stats['alpha_cdmin'] - denormalize(target_alpha_cdmin_norm, 6, norm_data)
            error_alpha_ldmax = nf_stats['alpha_ldmax'] - denormalize(target_alpha_ldmax_norm, 7, norm_data)
            
            # Compute thickness from predicted coordinates
            pred_min_thickness, pred_max_thickness = compute_thickness(pred_x, pred_y)
            error_min_thickness = pred_min_thickness - target_min_thickness
            error_max_thickness = pred_max_thickness - target_max_thickness
            
            # Compute errors for all alpha points (CL, CD, L/D)
            # Get target values from sequence
            target_cl = sequence[:, 1]  # Second column is Cl
            target_cd = sequence[:, 2]  # Third column is Cd
            target_cl_cd = sequence[:, 3]  # Fourth column is Cl/Cd
            
            # Denormalize target sequence values (they're stored as-is, might need normalization info)
            # For now, assume they're already in correct scale, or we'll need to handle this differently
            # Actually, the sequence values are raw aerodynamic values, not normalized
            
            # Compute errors for each point
            error_cl = nf_results['cl'] - target_cl
            error_cd = nf_results['cd'] - target_cd
            error_ld = nf_results['ld'] - target_cl_cd
            
            # Store correction data
            correction_data.append({
                'pred_af512': pred_af512_flat,
                'target_af512': target_af512_flat,
                'error_ldmax': error_ldmax,
                'error_clmax': error_clmax,
                'error_cdmin': error_cdmin,
                'error_alpha_clmax': error_alpha_clmax,
                'error_alpha_cdmin': error_alpha_cdmin,
                'error_alpha_ldmax': error_alpha_ldmax,
                'error_min_thickness': error_min_thickness,
                'error_max_thickness': error_max_thickness,
                'error_cl': error_cl,  # Array of errors
                'error_cd': error_cd,  # Array of errors
                'error_ld': error_ld,  # Array of errors
            })
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    print(f"Generated correction data for {len(correction_data)} samples")
    return correction_data


class CorrectionDataset(Dataset):
    """Dataset for correction model training."""
    def __init__(self, correction_data, add_noise=False, noise_level=0.005):
        self.correction_data = correction_data
        self.add_noise = add_noise
        self.noise_level = noise_level  # 0.5% = 0.005
    
    def __len__(self):
        return len(self.correction_data)
    
    def __getitem__(self, idx):
        data = self.correction_data[idx]
        
        # Get predicted AF512
        pred_af512 = data['pred_af512'].copy()
        
        # Input: predicted AF512 + error values
        # Error values: error_ldmax, error_clmax, error_cdmin, error_alpha_clmax, error_alpha_cdmin, 
        #               error_alpha_ldmax, error_min_thickness, error_max_thickness
        # Plus sequence errors: we'll use mean of error_cl, error_cd, error_ld
        error_stats = np.array([
            data['error_ldmax'],
            data['error_clmax'],
            data['error_cdmin'],
            data['error_alpha_clmax'],
            data['error_alpha_cdmin'],
            data['error_alpha_ldmax'],
            data['error_min_thickness'],
            data['error_max_thickness'],
            np.nanmean(data['error_cl']) if len(data['error_cl']) > 0 else 0.0,
            np.nanmean(data['error_cd']) if len(data['error_cd']) > 0 else 0.0,
            np.nanmean(data['error_ld']) if len(data['error_ld']) > 0 else 0.0,
        ], dtype=np.float32)
        
        # Replace NaN with 0
        error_stats = np.nan_to_num(error_stats, nan=0.0)
        
        # Add noise during training (0.5% random noise)
        if self.add_noise:
            # Add noise to AF512: noise = noise_level * |value| * random_normal
            af512_noise = np.random.normal(0, 1, pred_af512.shape).astype(np.float32)
            af512_noise = self.noise_level * np.abs(pred_af512) * af512_noise
            pred_af512 = pred_af512 + af512_noise
            
            # Add noise to error stats: noise = noise_level * |value| * random_normal
            error_noise = np.random.normal(0, 1, error_stats.shape).astype(np.float32)
            error_noise = self.noise_level * np.abs(error_stats) * error_noise
            error_stats = error_stats + error_noise
        
        # Concatenate predicted AF512 with error stats
        input_features = np.concatenate([pred_af512, error_stats])
        
        # Target: corrected AF512 (target - predicted gives us the correction to apply)
        # Actually, target should be the ground truth AF512
        target = data['target_af512']
        
        return torch.FloatTensor(input_features), torch.FloatTensor(target)


class CorrectionNet(nn.Module):
    """Correction model that takes predicted AF512 + errors and outputs corrected AF512."""
    def __init__(self, af512_size=1024, error_size=11, output_size=1024, hidden_sizes=[512, 512, 512, 512]):
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


def train_correction_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device='cpu', model_file='correction_model.pth'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    print(f"Training on {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val", leave=False):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch:3d}/{num_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    torch.save(model.state_dict(), model_file)
    print(f'\nModel saved to {model_file}')
    
    return train_losses, val_losses


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    if not NEURALFOIL_AVAILABLE:
        print("Error: NeuralFoil is required for this script")
        return
    
    # Load normalization data
    norm_file = 'feature_normalization.npy'
    if not os.path.exists(norm_file):
        print(f"Error: Normalization file '{norm_file}' not found!")
        return
    normalization_data = np.load(norm_file, allow_pickle=True).item()
    
    # Load trained aero_to_af512 model
    aero_model_file = 'aero_to_af512_model.pth'
    if not os.path.exists(aero_model_file):
        print(f"Error: Model file {aero_model_file} not found!")
        return
    
    print(f"Loading trained model from {aero_model_file}...")
    aero_model = AeroToAF512Net(
        scalar_input_size=10,
        sequence_embedding_dim=256,
        output_size=1024,
        hidden_sizes=[256, 256, 256, 256]
    )
    aero_model.load_state_dict(torch.load(aero_model_file, map_location=device))
    aero_model = aero_model.to(device)
    aero_model.eval()
    print("Model loaded successfully")
    
    # Load second half of data
    print("\nLoading second half of dataset...")
    features, target_af512 = load_second_half_data('unpacked_csv', 'bigfoil', num_points=512)
    
    if features is None or len(features) == 0:
        print("No data loaded!")
        return
    
    # Normalize features (for model input)
    features_normalized, _, _ = normalize_features(features)
    
    # Generate correction data
    print("\nGenerating correction data...")
    correction_data = generate_correction_data(
        aero_model, features_normalized, target_af512, normalization_data, device=device
    )
    
    if len(correction_data) == 0:
        print("No correction data generated!")
        return
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_correction, val_correction = train_test_split(
        correction_data, test_size=0.2, random_state=42
    )
    
    # Create datasets (add noise to training set only)
    train_dataset = CorrectionDataset(train_correction, add_noise=True, noise_level=0.005)
    val_dataset = CorrectionDataset(val_correction, add_noise=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create correction model
    model = CorrectionNet(
        af512_size=1024,
        error_size=11,  # 8 error stats + 3 mean sequence errors
        output_size=1024,
        hidden_sizes=[512, 512, 512]
    )
    
    # Train model
    print("\nTraining correction model...")
    train_losses, val_losses = train_correction_model(
        model, train_loader, val_loader,
        num_epochs=100, learning_rate=0.001, device=device, model_file='correction_model.pth'
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()

