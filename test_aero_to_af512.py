"""
Test script for the AeroToAF512 model.
Tests the full pipeline: Aero → AF512 → XY coordinates and compares with ground truth.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from train_aero_to_af512 import (
    load_csv_data,
    normalize_features,
    AeroToAF512Dataset,
    AeroToAF512Net,
    collate_fn,
    predict_af512,
    af512_to_dat_format
)
from train_airfoil import (
    af512_to_coordinates,
    AF512toXYNet,
    predict_xy_coordinates,
    load_dat_file
)
try:
    from train_correction_model import (
        CorrectionNet,
        compute_neuralfoil_values,
        compute_statistics,
        compute_thickness
    )
    CORRECTION_MODEL_AVAILABLE = True
except ImportError:
    CORRECTION_MODEL_AVAILABLE = False
    print("Warning: Correction model not available. Install train_correction_model.py")
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
try:
    import neuralfoil as nf
    NEURALFOIL_AVAILABLE = True
except ImportError:
    NEURALFOIL_AVAILABLE = False
    print("Warning: NeuralFoil not installed. Install with: pip install neuralfoil")


def test_model(model, test_loader, device='mps', num_samples=10):
    """Test the model and calculate metrics."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_losses = []
    
    criterion = nn.MSELoss()
    
    print(f"Testing on {len(test_loader.dataset)} samples...")
    
    with torch.no_grad():
        for batch_data, targets in tqdm(test_loader, desc="Testing"):
            scalars = batch_data['scalars'].to(device)
            sequence = batch_data['sequence'].to(device)
            lengths = batch_data['lengths'].to(device)
            targets = targets.to(device)
            
            outputs = model(scalars, sequence, lengths)
            loss = criterion(outputs, targets)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_losses.append(loss.item())
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate overall metrics
    mse = np.mean((all_predictions - all_targets) ** 2)
    mae = np.mean(np.abs(all_predictions - all_targets))
    rmse = np.sqrt(mse)
    
    # Calculate per-sample metrics
    per_sample_mse = np.mean((all_predictions - all_targets) ** 2, axis=1)
    per_sample_mae = np.mean(np.abs(all_predictions - all_targets), axis=1)
    
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_targets)}")
    print(f"Mean MSE: {mse:.6f}")
    print(f"Mean MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Mean per-sample MSE: {np.mean(per_sample_mse):.6f}")
    print(f"Mean per-sample MAE: {np.mean(per_sample_mae):.6f}")
    print(f"Std per-sample MSE: {np.std(per_sample_mse):.6f}")
    print(f"Best sample MSE: {np.min(per_sample_mse):.6f}")
    print(f"Worst sample MSE: {np.max(per_sample_mse):.6f}")
    print(f"{'='*60}\n")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'predictions': all_predictions,
        'targets': all_targets,
        'per_sample_mse': per_sample_mse,
        'per_sample_mae': per_sample_mae
    }


def compute_errors_for_correction(pred_af512_flat, feature_dict, normalization_data, device='cpu'):
    """
    Compute error statistics needed for correction model.
    Returns error stats array (11 values) or None if NeuralFoil fails.
    """
    if not NEURALFOIL_AVAILABLE or not CORRECTION_MODEL_AVAILABLE:
        return None
    
    try:
        # Convert predicted AF512 to coordinates
        # pred_af512_flat is (1024,) - need to reshape to (512, 2)
        pred_af512 = pred_af512_flat.reshape(512, 2)
        pred_x, pred_y = af512_to_coordinates(pred_af512)
        coordinates = np.column_stack([pred_x, pred_y])
        
        # Denormalize Re and Mach
        scalars = feature_dict['scalars']
        norm_data = normalization_data
        
        re_norm = scalars[0]
        re_min = norm_data['min'][0, 0]
        re_max = norm_data['max'][0, 0]
        re_range = re_max - re_min
        Re = re_norm * re_range + re_min if re_range > 1e-6 else re_min
        
        # Get alpha values from sequence
        sequence = feature_dict['sequence']
        alpha_values = sequence[:, 0]
        
        # Compute NeuralFoil values
        nf_results = compute_neuralfoil_values(coordinates, alpha_values, Re)
        
        # Compute statistics from NeuralFoil results
        nf_stats = compute_statistics(alpha_values, nf_results['cl'], nf_results['cd'], nf_results['ld'])
        if nf_stats is None:
            return None
        
        # Denormalize target statistics
        def denormalize(val, idx, norm_data):
            min_val = norm_data['min'][0, idx]
            max_val = norm_data['max'][0, idx]
            range_val = max_val - min_val
            return val * range_val + min_val if range_val > 1e-6 else min_val
        
        target_ldmax = denormalize(scalars[2], 2, norm_data)
        target_clmax = denormalize(scalars[3], 3, norm_data)
        target_cdmin = denormalize(scalars[4], 4, norm_data)
        target_min_thickness = denormalize(scalars[8], 8, norm_data)
        target_max_thickness = denormalize(scalars[9], 9, norm_data)
        
        # Compute errors
        error_ldmax = nf_stats['ldmax'] - target_ldmax
        error_clmax = nf_stats['clmax'] - target_clmax
        error_cdmin = nf_stats['cdmin'] - target_cdmin
        error_alpha_clmax = nf_stats['alpha_clmax'] - denormalize(scalars[5], 5, norm_data)
        error_alpha_cdmin = nf_stats['alpha_cdmin'] - denormalize(scalars[6], 6, norm_data)
        error_alpha_ldmax = nf_stats['alpha_ldmax'] - denormalize(scalars[7], 7, norm_data)
        
        # Compute thickness errors
        pred_min_thickness, pred_max_thickness = compute_thickness(pred_x, pred_y)
        error_min_thickness = pred_min_thickness - target_min_thickness
        error_max_thickness = pred_max_thickness - target_max_thickness
        
        # Compute sequence errors
        target_cl = sequence[:, 1]
        target_cd = sequence[:, 2]
        target_cl_cd = sequence[:, 3]
        
        error_cl = nf_results['cl'] - target_cl
        error_cd = nf_results['cd'] - target_cd
        error_ld = nf_results['ld'] - target_cl_cd
        
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


def visualize_predictions_full_pipeline(aero_model, af512_to_xy_model, test_features, test_xy_coords, 
                                         correction_model=None, normalization_data=None,
                                         device='mps', num_samples=10, save_dir='test_results'):
    """Visualize full pipeline: Aero → AF512 → (Correction) → XY vs ground truth XY coordinates."""
    os.makedirs(save_dir, exist_ok=True)
    
    aero_model.eval()
    af512_to_xy_model.eval()
    if correction_model is not None:
        correction_model.eval()
    
    # Select random samples
    num_samples = min(num_samples, len(test_features))
    indices = np.random.choice(len(test_features), num_samples, replace=False)
    
    use_correction = correction_model is not None and normalization_data is not None
    print(f"Visualizing {num_samples} random samples (full pipeline{' with correction' if use_correction else ''})...")
    
    # Create subplots
    cols = 5
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if num_samples == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Get ground truth XY coordinates
        gt_xy = test_xy_coords[idx]
        # Ensure it's flattened
        if gt_xy.ndim > 1:
            gt_xy = gt_xy.flatten()
        gt_x = gt_xy[:1024]
        gt_y = gt_xy[1024:]
        
        # Full pipeline: Aero → AF512 → (Correction) → XY
        feature_dict = test_features[idx]
        scalars = feature_dict['scalars']
        sequence = feature_dict['sequence']
        
        # Step 1: Aero → AF512
        scalars_tensor = torch.FloatTensor(scalars).unsqueeze(0).to(device)
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
        lengths_tensor = torch.tensor([len(sequence)], dtype=torch.long).to(device)
        
        aero_model.eval()
        with torch.no_grad():
            pred_af512_tensor = aero_model(scalars_tensor, sequence_tensor, lengths_tensor)
            pred_af512_flat = pred_af512_tensor.cpu().numpy().flatten()
        
        # Step 2: Apply correction if available
        if use_correction:
            error_stats = compute_errors_for_correction(pred_af512_flat, feature_dict, normalization_data, device)
            if error_stats is not None:
                corrected_af512_flat = apply_correction_model(correction_model, pred_af512_flat, error_stats, device)
                # Use corrected AF512 for visualization
                pred_af512_flat = corrected_af512_flat
        
        # Step 3: AF512 → XY
        pred_x, pred_y = predict_xy_coordinates(af512_to_xy_model, pred_af512_flat, device=device)
        
        # Calculate error in XY space
        xy_mse = np.mean((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)
        xy_mae = np.mean(np.abs(gt_x - pred_x) + np.abs(gt_y - pred_y))
        
        # Plot
        ax.plot(gt_x, gt_y, 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
        ax.plot(pred_x, pred_y, 'r--', linewidth=2, label='Prediction' + (' (Corrected)' if use_correction else ''), alpha=0.7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Sample {i+1}\nXY MSE: {xy_mse:.6f}\nXY MAE: {xy_mae:.6f}')
        ax.legend(fontsize=8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    # Hide empty subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'full_pipeline_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


def test_full_pipeline(aero_model, af512_to_xy_model, test_loader, test_xy_coords, test_features,
                       correction_model=None, normalization_data=None, device='mps'):
    """Test the full pipeline: Aero → AF512 → (Correction) → XY and calculate metrics."""
    aero_model.eval()
    af512_to_xy_model.eval()
    if correction_model is not None:
        correction_model.eval()
    
    all_pred_xy = []
    all_pred_xy_uncorrected = []
    all_gt_xy = []
    all_xy_mse = []
    all_xy_mse_uncorrected = []
    all_xy_mae = []
    all_xy_mae_uncorrected = []
    
    use_correction = correction_model is not None and normalization_data is not None
    print(f"Testing full pipeline on {len(test_loader.dataset)} samples{' with correction' if use_correction else ''}...")
    
    with torch.no_grad():
        for batch_idx, (batch_data, _) in enumerate(tqdm(test_loader, desc="Testing pipeline")):
            scalars = batch_data['scalars'].to(device)
            sequence = batch_data['sequence'].to(device)
            lengths = batch_data['lengths'].to(device)
            
            # Step 1: Aero → AF512
            af512_outputs = aero_model(scalars, sequence, lengths)
            
            # Step 2: AF512 → XY (process each sample in batch)
            batch_size = af512_outputs.shape[0]
            pred_xy_batch = []
            pred_xy_uncorrected_batch = []
            
            start_idx = batch_idx * test_loader.batch_size
            
            for i in range(batch_size):
                af512_flat = af512_outputs[i].cpu().numpy()
                
                # Get uncorrected XY first
                pred_x_uncorrected, pred_y_uncorrected = predict_xy_coordinates(af512_to_xy_model, af512_flat, device=device)
                pred_xy_uncorrected = np.concatenate([pred_x_uncorrected, pred_y_uncorrected])
                pred_xy_uncorrected_batch.append(pred_xy_uncorrected)
                
                # Apply correction if available
                if use_correction:
                    feature_idx = start_idx + i
                    if feature_idx < len(test_features):
                        feature_dict = test_features[feature_idx]
                        error_stats = compute_errors_for_correction(af512_flat, feature_dict, normalization_data, device)
                        if error_stats is not None:
                            corrected_af512_flat = apply_correction_model(correction_model, af512_flat, error_stats, device)
                            pred_x, pred_y = predict_xy_coordinates(af512_to_xy_model, corrected_af512_flat, device=device)
                            pred_xy = np.concatenate([pred_x, pred_y])
                        else:
                            # If correction fails, use uncorrected
                            pred_xy = pred_xy_uncorrected
                    else:
                        pred_xy = pred_xy_uncorrected
                else:
                    pred_xy = pred_xy_uncorrected
                
                pred_xy_batch.append(pred_xy)
            
            pred_xy_batch = np.array(pred_xy_batch)
            pred_xy_uncorrected_batch = np.array(pred_xy_uncorrected_batch)
            
            # Get corresponding ground truth XY
            end_idx = min(start_idx + batch_size, len(test_xy_coords))
            gt_xy_batch = test_xy_coords[start_idx:end_idx]
            
            # Calculate errors
            for i in range(len(pred_xy_batch)):
                pred_xy = pred_xy_batch[i]
                pred_xy_uncorrected = pred_xy_uncorrected_batch[i]
                gt_xy = gt_xy_batch[i]
                
                # Both should be flattened arrays of shape (2048,)
                if gt_xy.ndim > 1:
                    gt_xy = gt_xy.flatten()
                
                # Ensure both are same length
                min_len = min(len(pred_xy), len(gt_xy))
                pred_xy = pred_xy[:min_len]
                pred_xy_uncorrected = pred_xy_uncorrected[:min_len]
                gt_xy = gt_xy[:min_len]
                
                xy_mse = np.mean((pred_xy - gt_xy) ** 2)
                xy_mae = np.mean(np.abs(pred_xy - gt_xy))
                xy_mse_uncorrected = np.mean((pred_xy_uncorrected - gt_xy) ** 2)
                xy_mae_uncorrected = np.mean(np.abs(pred_xy_uncorrected - gt_xy))
                
                all_pred_xy.append(pred_xy)
                all_pred_xy_uncorrected.append(pred_xy_uncorrected)
                all_gt_xy.append(gt_xy)
                all_xy_mse.append(xy_mse)
                all_xy_mse_uncorrected.append(xy_mse_uncorrected)
                all_xy_mae.append(xy_mae)
                all_xy_mae_uncorrected.append(xy_mae_uncorrected)
    
    all_pred_xy = np.array(all_pred_xy)
    all_pred_xy_uncorrected = np.array(all_pred_xy_uncorrected)
    all_gt_xy = np.array(all_gt_xy)
    
    # Calculate overall metrics
    overall_xy_mse = np.mean(all_xy_mse)
    overall_xy_mae = np.mean(all_xy_mae)
    overall_xy_rmse = np.sqrt(overall_xy_mse)
    
    print(f"\n{'='*60}")
    print(f"Full Pipeline Test Results (Aero → AF512 → {'[Correction] → ' if use_correction else ''}XY):")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_gt_xy)}")
    print(f"Mean XY MSE: {overall_xy_mse:.6f}")
    print(f"Mean XY MAE: {overall_xy_mae:.6f}")
    print(f"XY RMSE: {overall_xy_rmse:.6f}")
    print(f"Std XY MSE: {np.std(all_xy_mse):.6f}")
    print(f"Best sample XY MSE: {np.min(all_xy_mse):.6f}")
    print(f"Worst sample XY MSE: {np.max(all_xy_mse):.6f}")
    
    if use_correction:
        overall_xy_mse_uncorrected = np.mean(all_xy_mse_uncorrected)
        overall_xy_mae_uncorrected = np.mean(all_xy_mae_uncorrected)
        improvement_mse = ((overall_xy_mse_uncorrected - overall_xy_mse) / overall_xy_mse_uncorrected) * 100
        improvement_mae = ((overall_xy_mae_uncorrected - overall_xy_mae) / overall_xy_mae_uncorrected) * 100
        print(f"\nUncorrected Results:")
        print(f"  Mean XY MSE: {overall_xy_mse_uncorrected:.6f}")
        print(f"  Mean XY MAE: {overall_xy_mae_uncorrected:.6f}")
        print(f"\nCorrection Improvement:")
        print(f"  MSE improvement: {improvement_mse:.2f}%")
        print(f"  MAE improvement: {improvement_mae:.2f}%")
    
    print(f"{'='*60}\n")
    
    result = {
        'xy_mse': overall_xy_mse,
        'xy_mae': overall_xy_mae,
        'xy_rmse': overall_xy_rmse,
        'predictions': all_pred_xy,
        'targets': all_gt_xy,
        'per_sample_mse': np.array(all_xy_mse),
        'per_sample_mae': np.array(all_xy_mae)
    }
    
    if use_correction:
        result['predictions_uncorrected'] = all_pred_xy_uncorrected
        result['xy_mse_uncorrected'] = overall_xy_mse_uncorrected
        result['xy_mae_uncorrected'] = overall_xy_mae_uncorrected
        result['per_sample_mse_uncorrected'] = np.array(all_xy_mse_uncorrected)
        result['per_sample_mae_uncorrected'] = np.array(all_xy_mae_uncorrected)
    
    return result


def visualize_error_distribution(results, save_dir='test_results'):
    """Visualize error distribution."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Use xy_mse/xy_mae if available (from pipeline results), otherwise use mse/mae
    mse_key = 'xy_mse' if 'xy_mse' in results else 'mse'
    mae_key = 'xy_mae' if 'xy_mae' in results else 'mae'
    per_sample_mse_key = 'per_sample_mse'
    per_sample_mae_key = 'per_sample_mae'
    
    has_uncorrected = 'per_sample_mse_uncorrected' in results
    
    if has_uncorrected:
        # Show both corrected and uncorrected
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # MSE distribution - corrected
        axes[0, 0].hist(results[per_sample_mse_key], bins=50, edgecolor='black', alpha=0.7, label='Corrected', color='green')
        axes[0, 0].axvline(results[mse_key], color='g', linestyle='--', linewidth=2, label=f'Mean: {results[mse_key]:.6f}')
        axes[0, 0].set_xlabel('MSE')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('MSE Distribution (Corrected)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MSE distribution - uncorrected
        axes[0, 1].hist(results['per_sample_mse_uncorrected'], bins=50, edgecolor='black', alpha=0.7, label='Uncorrected', color='red')
        axes[0, 1].axvline(results['xy_mse_uncorrected'], color='r', linestyle='--', linewidth=2, label=f'Mean: {results["xy_mse_uncorrected"]:.6f}')
        axes[0, 1].set_xlabel('MSE')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('MSE Distribution (Uncorrected)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # MAE distribution - corrected
        axes[1, 0].hist(results[per_sample_mae_key], bins=50, edgecolor='black', alpha=0.7, label='Corrected', color='green')
        axes[1, 0].axvline(results[mae_key], color='g', linestyle='--', linewidth=2, label=f'Mean: {results[mae_key]:.6f}')
        axes[1, 0].set_xlabel('MAE')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('MAE Distribution (Corrected)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # MAE distribution - uncorrected
        axes[1, 1].hist(results['per_sample_mae_uncorrected'], bins=50, edgecolor='black', alpha=0.7, label='Uncorrected', color='red')
        axes[1, 1].axvline(results['xy_mae_uncorrected'], color='r', linestyle='--', linewidth=2, label=f'Mean: {results["xy_mae_uncorrected"]:.6f}')
        axes[1, 1].set_xlabel('MAE')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('MAE Distribution (Uncorrected)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Show only corrected/regular results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # MSE distribution
        axes[0].hist(results[per_sample_mse_key], bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(results[mse_key], color='r', linestyle='--', linewidth=2, label=f'Mean MSE: {results[mse_key]:.6f}')
        axes[0].set_xlabel('MSE')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('MSE Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE distribution
        axes[1].hist(results[per_sample_mae_key], bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(results[mae_key], color='r', linestyle='--', linewidth=2, label=f'Mean MAE: {results[mae_key]:.6f}')
        axes[1].set_xlabel('MAE')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('MAE Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'error_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved error distribution to {save_path}")
    plt.close()


def save_predictions_to_dat(results, test_features, save_dir='test_results', num_samples=10):
    """Save predictions and ground truth to .dat files for comparison."""
    os.makedirs(save_dir, exist_ok=True)
    
    indices = np.random.choice(len(results['predictions']), min(num_samples, len(results['predictions'])), replace=False)
    
    print(f"Saving {len(indices)} samples to .dat files...")
    
    for i, idx in enumerate(indices):
        # Ground truth - results contain XY coordinates (2048,) format
        gt_xy = results['targets'][idx]
        if gt_xy.ndim > 1:
            gt_xy = gt_xy.flatten()
        gt_x = gt_xy[:1024]
        gt_y = gt_xy[1024:]
        
        # Prediction - results contain XY coordinates (2048,) format
        pred_xy = results['predictions'][idx]
        if pred_xy.ndim > 1:
            pred_xy = pred_xy.flatten()
        pred_x = pred_xy[:1024]
        pred_y = pred_xy[1024:]
        
        # Save ground truth
        gt_path = os.path.join(save_dir, f'sample_{i+1}_ground_truth.dat')
        with open(gt_path, 'w') as f:
            f.write(f"Ground Truth Sample {i+1}\n")
            for x, y in zip(gt_x, gt_y):
                f.write(f"{x:.6f} {y:.6f}\n")
        
        # Save prediction
        pred_path = os.path.join(save_dir, f'sample_{i+1}_prediction.dat')
        with open(pred_path, 'w') as f:
            f.write(f"Prediction Sample {i+1}\n")
            for x, y in zip(pred_x, pred_y):
                f.write(f"{x:.6f} {y:.6f}\n")
        
        # Save comparison (both in one file)
        comp_path = os.path.join(save_dir, f'sample_{i+1}_comparison.dat')
        with open(comp_path, 'w') as f:
            f.write(f"Comparison Sample {i+1}\n")
            f.write("# X_GT Y_GT X_PRED Y_PRED\n")
            for x_gt, y_gt, x_pred, y_pred in zip(gt_x, gt_y, pred_x, pred_y):
                f.write(f"{x_gt:.6f} {y_gt:.6f} {x_pred:.6f} {y_pred:.6f}\n")
    
    print(f"Saved .dat files to {save_dir}")


def load_re_mach_from_csv(csv_dir='unpacked_csv'):
    """Load Re and Mach values from CSV files using the same logic as load_csv_data."""
    csv_path = Path(csv_dir)
    csv_files = list(csv_path.glob('*.csv'))
    
    re_mach_list = []
    
    for csv_file in csv_files:
        try:
            import csv
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            for row in rows:
                def safe_float(val, default=0.0):
                    try:
                        return float(val) if val and val != 'nan' and val != 'NaN' else default
                    except:
                        return default
                
                # Parse vectors to match the same validation logic as load_csv_data
                alpha_str = str(row.get('alpha', '[]')).strip()
                cl_str = str(row.get('Cl', '[]')).strip()
                cd_str = str(row.get('Cd', '[]')).strip()
                cl_cd_str = str(row.get('Cl_Cd', '[]')).strip()
                
                def parse_vector(vec_str):
                    if not vec_str or vec_str == '':
                        return np.array([])
                    if vec_str.startswith('[') and vec_str.endswith(']'):
                        try:
                            lst = [float(x.strip()) for x in vec_str.strip("[]").split(",")]
                            return np.array(lst, dtype=np.float32)
                        except:
                            return np.array([])
                    else:
                        try:
                            return np.array([float(vec_str.strip())], dtype=np.float32)
                        except:
                            return np.array([])
                
                alpha_arr = parse_vector(alpha_str)
                cl_arr = parse_vector(cl_str)
                cd_arr = parse_vector(cd_str)
                cl_cd_arr = parse_vector(cl_cd_str)
                
                min_len = min(len(alpha_arr), len(cl_arr), len(cd_arr), len(cl_cd_arr))
                if min_len == 0:
                    continue
                
                # Extract Re and Mach directly from CSV (not normalized)
                re = safe_float(row.get('Re', 0))
                mach = safe_float(row.get('Mach', 0))
                
                # Store Re and Mach for this valid row
                re_mach_list.append({'re': re, 'mach': mach})
        except:
            continue
    
    return re_mach_list


def run_neuralfoil_analysis(test_features, test_re_mach, results, device='mps', num_samples=100):
    """Run NeuralFoil analysis on predicted airfoils and compare with input characteristics."""
    if not NEURALFOIL_AVAILABLE:
        print("NeuralFoil not available. Skipping aerodynamic analysis.")
        return None
    
    print(f"\n{'='*60}")
    print("Running NeuralFoil Analysis (xxxlarge model)")
    print(f"{'='*60}")
    
    # Limit to num_samples
    num_samples = min(num_samples, len(results['predictions']), len(test_features), len(test_re_mach))
    indices = np.arange(num_samples)
    
    input_cl_list = []
    input_cd_list = []
    input_ld_list = []
    predicted_cl_list = []
    predicted_cd_list = []
    predicted_ld_list = []
    re_list = []
    alpha_list = []
    errors = []
    
    print(f"Analyzing {num_samples} samples with NeuralFoil...")
    
    for idx in tqdm(indices, desc="NeuralFoil analysis"):
        try:
            # Get predicted XY coordinates
            pred_xy = results['predictions'][idx]
            pred_x = pred_xy[:1024]
            pred_y = pred_xy[1024:]
            
            # Prepare coordinates for NeuralFoil (needs to be Nx2 array)
            coordinates = np.column_stack([pred_x, pred_y])
            
            # Get input features to extract alpha sequence
            feature_dict = test_features[idx]
            
            # Get Re and Mach directly from CSV (not normalized)
            if idx < len(test_re_mach):
                re_denorm = test_re_mach[idx]['re']
                mach_denorm = test_re_mach[idx]['mach']
            else:
                # Fallback: denormalize from features
                scalars = feature_dict['scalars']
                try:
                    norm_data = np.load('feature_normalization.npy', allow_pickle=True).item()
                    re_min = norm_data['min'][0, 0]
                    re_max = norm_data['max'][0, 0]
                    re_range = re_max - re_min
                    re_denorm = scalars[0] * re_range + re_min if re_range > 1e-6 else re_min
                    mach_min = norm_data['min'][0, 1]
                    mach_max = norm_data['max'][0, 1]
                    mach_range = mach_max - mach_min
                    mach_denorm = scalars[1] * mach_range + mach_min if mach_range > 1e-6 else mach_min
                except:
                    re_denorm = scalars[0] * (1e7 - 1e5) + 1e5
                    mach_denorm = scalars[1] * 1.0
            
            # Get sequence to extract alpha values
            sequence = feature_dict['sequence']
            alpha_values = sequence[:, 0]  # First column is alpha
            
            # For now, use the first alpha value (or we could analyze multiple)
            # We'll use alpha=0 or the first alpha in the sequence
            alpha = float(alpha_values[0]) if len(alpha_values) > 0 else 0.0
            
            # Run NeuralFoil analysis
            # NeuralFoil expects coordinates as Nx2 array
            try:
                nf_results = nf.get_aero_from_coordinates(
                    coordinates=coordinates,
                    alpha=alpha,
                    Re=re_denorm,
                    model_size="xxxlarge"
                )
                
                # Extract CL and CD (handle both scalar and array results)
                pred_cl = nf_results['CL']
                pred_cd = nf_results['CD']
                
                # If results are arrays, take first element
                if isinstance(pred_cl, np.ndarray):
                    pred_cl = float(pred_cl[0])
                if isinstance(pred_cd, np.ndarray):
                    pred_cd = float(pred_cd[0])
                
                pred_ld = pred_cl / pred_cd if pred_cd > 0 else 0
                
                # Get input values from sequence (use alpha closest to the one we tested)
                # For simplicity, use the first alpha's corresponding CL and CD
                if len(sequence) > 0:
                    input_cl = float(sequence[0, 1])  # Second column is CL
                    input_cd = float(sequence[0, 2])  # Third column is CD
                    input_ld = input_cl / input_cd if input_cd > 0 else 0
                else:
                    input_cl = input_cd = input_ld = 0.0
                
                input_cl_list.append(input_cl)
                input_cd_list.append(input_cd)
                input_ld_list.append(input_ld)
                predicted_cl_list.append(pred_cl)
                predicted_cd_list.append(pred_cd)
                predicted_ld_list.append(pred_ld)
                re_list.append(re_denorm)
                alpha_list.append(alpha)
                
            except Exception as e:
                errors.append(f"Sample {idx}: {str(e)}")
                continue
                
        except Exception as e:
            errors.append(f"Sample {idx}: {str(e)}")
            continue
    
    if len(errors) > 0:
        print(f"\nWarnings ({len(errors)} errors):")
        for err in errors[:10]:  # Show first 10 errors
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    if len(input_cl_list) == 0:
        print("No successful NeuralFoil analyses completed!")
        return None
    
    # Calculate comparison metrics
    input_cl_arr = np.array(input_cl_list)
    input_cd_arr = np.array(input_cd_list)
    input_ld_arr = np.array(input_ld_list)
    pred_cl_arr = np.array(predicted_cl_list)
    pred_cd_arr = np.array(predicted_cd_list)
    pred_ld_arr = np.array(predicted_ld_list)
    
    cl_mse = np.mean((input_cl_arr - pred_cl_arr) ** 2)
    cl_mae = np.mean(np.abs(input_cl_arr - pred_cl_arr))
    cd_mse = np.mean((input_cd_arr - pred_cd_arr) ** 2)
    cd_mae = np.mean(np.abs(input_cd_arr - pred_cd_arr))
    ld_mse = np.mean((input_ld_arr - pred_ld_arr) ** 2)
    ld_mae = np.mean(np.abs(input_ld_arr - pred_ld_arr))
    
    print(f"\n{'='*60}")
    print(f"NeuralFoil Analysis Results ({len(input_cl_list)} successful analyses):")
    print(f"{'='*60}")
    print(f"CL - MSE: {cl_mse:.6f}, MAE: {cl_mae:.6f}")
    print(f"CD - MSE: {cd_mse:.6f}, MAE: {cd_mae:.6f}")
    print(f"L/D - MSE: {ld_mse:.6f}, MAE: {ld_mae:.6f}")
    print(f"{'='*60}\n")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # CL comparison
    axes[0, 0].scatter(input_cl_arr, pred_cl_arr, alpha=0.5)
    axes[0, 0].plot([input_cl_arr.min(), input_cl_arr.max()], 
                    [input_cl_arr.min(), input_cl_arr.max()], 'r--', label='Perfect match')
    axes[0, 0].set_xlabel('Input CL')
    axes[0, 0].set_ylabel('Predicted CL (NeuralFoil)')
    axes[0, 0].set_title('CL Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # CD comparison
    axes[0, 1].scatter(input_cd_arr, pred_cd_arr, alpha=0.5)
    axes[0, 1].plot([input_cd_arr.min(), input_cd_arr.max()], 
                    [input_cd_arr.min(), input_cd_arr.max()], 'r--', label='Perfect match')
    axes[0, 1].set_xlabel('Input CD')
    axes[0, 1].set_ylabel('Predicted CD (NeuralFoil)')
    axes[0, 1].set_title('CD Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # L/D comparison
    axes[0, 2].scatter(input_ld_arr, pred_ld_arr, alpha=0.5)
    axes[0, 2].plot([input_ld_arr.min(), input_ld_arr.max()], 
                    [input_ld_arr.min(), input_ld_arr.max()], 'r--', label='Perfect match')
    axes[0, 2].set_xlabel('Input L/D')
    axes[0, 2].set_ylabel('Predicted L/D (NeuralFoil)')
    axes[0, 2].set_title('L/D Comparison')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Error distributions
    axes[1, 0].hist(input_cl_arr - pred_cl_arr, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('CL Error (Input - Predicted)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('CL Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(input_cd_arr - pred_cd_arr, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('CD Error (Input - Predicted)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('CD Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].hist(input_ld_arr - pred_ld_arr, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 2].set_xlabel('L/D Error (Input - Predicted)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('L/D Error Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = 'test_results/neuralfoil_comparison.png'
    os.makedirs('test_results', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved NeuralFoil comparison to {save_path}")
    plt.close()
    
    return {
        'input_cl': input_cl_arr,
        'input_cd': input_cd_arr,
        'input_ld': input_ld_arr,
        'predicted_cl': pred_cl_arr,
        'predicted_cd': pred_cd_arr,
        'predicted_ld': pred_ld_arr,
        'cl_mse': cl_mse,
        'cl_mae': cl_mae,
        'cd_mse': cd_mse,
        'cd_mae': cd_mae,
        'ld_mse': ld_mse,
        'ld_mae': ld_mae
    }


def load_xy_coordinates_from_data(features, bigfoil_dir='bigfoil', num_points=512):
    """Load XY coordinates from .dat files using the same logic as load_csv_data."""
    xy_coords_list = []
    csv_path = Path('unpacked_csv')
    bigfoil_path = Path(bigfoil_dir)
    
    csv_files = list(csv_path.glob('*.csv'))
    print(f"Loading XY coordinates from .dat files (matching CSV loading logic)...")
    
    # Cache for airfoil AF512 data (same airfoil name -> same XY coordinates)
    airfoil_cache = {}
    
    for csv_file in tqdm(csv_files, desc="Loading XY coords"):
        try:
            airfoil_name = csv_file.stem
            
            # Try to find matching .dat file
            dat_file = bigfoil_path / f"{airfoil_name}.dat"
            if not dat_file.exists():
                # Try without extension or with different naming
                dat_file = bigfoil_path / airfoil_name
                if not dat_file.exists():
                    continue
            
            # Load XY coordinates for this airfoil (cache it)
            if airfoil_name not in airfoil_cache:
                _, xy_tuple = load_dat_file(dat_file, num_points=num_points)
                if xy_tuple is not None and xy_tuple[0] is not None:
                    # xy_tuple is (x_full, y_full), convert to flattened array [x_coords, y_coords]
                    x_full, y_full = xy_tuple
                    xy_coords = np.concatenate([x_full, y_full])  # Shape: (2048,)
                    airfoil_cache[airfoil_name] = xy_coords
                else:
                    continue
            else:
                xy_coords = airfoil_cache[airfoil_name]
            
            # Read CSV and process rows (matching load_csv_data logic)
            import csv
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Process each row (matching the same logic as load_csv_data)
            for idx, row in enumerate(rows):
                try:
                    # Check if we would have loaded this row (same validation as load_csv_data)
                    # Parse vectors to check if they're valid
                    alpha_str = str(row.get('alpha', '[]')).strip()
                    cl_str = str(row.get('Cl', '[]')).strip()
                    cd_str = str(row.get('Cd', '[]')).strip()
                    cl_cd_str = str(row.get('Cl_Cd', '[]')).strip()
                    
                    # Parse arrays
                    def parse_vector(vec_str):
                        if not vec_str or vec_str == '':
                            return np.array([])
                        if vec_str.startswith('[') and vec_str.endswith(']'):
                            try:
                                lst = [float(x.strip()) for x in vec_str.strip("[]").split(",")]
                                return np.array(lst, dtype=np.float32)
                            except:
                                return np.array([])
                        else:
                            try:
                                return np.array([float(vec_str.strip())], dtype=np.float32)
                            except:
                                return np.array([])
                    
                    alpha_arr = parse_vector(alpha_str)
                    cl_arr = parse_vector(cl_str)
                    cd_arr = parse_vector(cd_str)
                    cl_cd_arr = parse_vector(cl_cd_str)
                    
                    # Same validation as load_csv_data
                    min_len = min(len(alpha_arr), len(cl_arr), len(cd_arr), len(cl_cd_arr))
                    if min_len == 0:
                        continue
                    
                    # If we get here, this row would have been loaded, so add XY coords
                    xy_coords_list.append(xy_coords)
                    
                except Exception:
                    # Skip rows that can't be parsed (matching load_csv_data behavior)
                    continue
        
        except Exception as e:
            continue
    
    print(f"Loaded {len(xy_coords_list)} XY coordinate sets")
    return np.array(xy_coords_list)


def main():
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load both models
    aero_model_file = 'aero_to_af512_model.pth'
    af512_to_xy_model_file = 'af512_to_xy_model.pth'
    
    if not os.path.exists(aero_model_file):
        print(f"Model file {aero_model_file} not found!")
        return
    
    if not os.path.exists(af512_to_xy_model_file):
        print(f"Model file {af512_to_xy_model_file} not found!")
        return
    
    # Create and load AeroToAF512 model
    print(f"Loading AeroToAF512 model from {aero_model_file}...")
    aero_model = AeroToAF512Net(
        scalar_input_size=10,
        sequence_embedding_dim=256,
        output_size=1024,
        hidden_sizes=[256, 256, 256, 256]
    )
    aero_model.load_state_dict(torch.load(aero_model_file, map_location=device))
    aero_model = aero_model.to(device)
    print("AeroToAF512 model loaded successfully")
    
    # Create and load AF512toXY model
    print(f"Loading AF512toXY model from {af512_to_xy_model_file}...")
    af512_to_xy_model = AF512toXYNet(
        input_size=1024,
        output_size=2048,
        hidden_sizes=[256, 256, 256, 256]
    )
    af512_to_xy_model.load_state_dict(torch.load(af512_to_xy_model_file, map_location=device))
    af512_to_xy_model = af512_to_xy_model.to(device)
    print("AF512toXY model loaded successfully")
    
    # Load correction model if available
    correction_model = None
    normalization_data = None
    correction_model_file = 'correction_model.pth'
    if CORRECTION_MODEL_AVAILABLE and os.path.exists(correction_model_file):
        print(f"Loading Correction model from {correction_model_file}...")
        correction_model = CorrectionNet(
            af512_size=1024,
            error_size=11,
            output_size=1024,
            hidden_sizes=[512, 512, 512]  # Match the architecture used during training
        )
        correction_model.load_state_dict(torch.load(correction_model_file, map_location=device))
        correction_model = correction_model.to(device)
        correction_model.eval()
        print("Correction model loaded successfully")
        
        # Load normalization data for error computation
        norm_file = 'feature_normalization.npy'
        if os.path.exists(norm_file):
            normalization_data = np.load(norm_file, allow_pickle=True).item()
            print("Normalization data loaded successfully")
        else:
            print(f"Warning: Normalization file {norm_file} not found. Correction model will not be used.")
            correction_model = None
    else:
        if not CORRECTION_MODEL_AVAILABLE:
            print("Correction model not available (train_correction_model.py not found)")
        elif not os.path.exists(correction_model_file):
            print(f"Correction model file {correction_model_file} not found. Continuing without correction.")
    
    # Load data
    print("\nLoading test data...")
    features, af512_data = load_csv_data('unpacked_csv', 'bigfoil', num_points=512)
    
    if features is None or len(features) == 0:
        print("No data loaded!")
        return
    
    # Load XY coordinates
    xy_coords_data = load_xy_coordinates_from_data(features, 'bigfoil', num_points=512)
    
    # Load Re and Mach values from CSV (before normalization/splitting)
    print("Loading Re and Mach values from CSV files...")
    re_mach_list_full = load_re_mach_from_csv('unpacked_csv')
    
    # Normalize features
    print("Normalizing features...")
    features_normalized, min_vals, max_vals = normalize_features(features)
    
    # Split data - limit to 100 random samples for testing
    from sklearn.model_selection import train_test_split
    _, test_features_full, _, test_xy_coords_full = train_test_split(
        features_normalized, xy_coords_data, test_size=0.1, random_state=42
    )
    
    # Also split re_mach_list to match
    _, test_re_mach_full = train_test_split(
        re_mach_list_full, test_size=0.1, random_state=42
    )
    
    # Randomly select 100 samples
    num_test_samples = min(100, len(test_features_full))
    if num_test_samples < len(test_features_full):
        indices = np.random.choice(len(test_features_full), num_test_samples, replace=False)
        test_features = [test_features_full[i] for i in indices]
        test_xy_coords = test_xy_coords_full[indices]
        test_re_mach = [test_re_mach_full[i] for i in indices]
    else:
        test_features = test_features_full
        test_xy_coords = test_xy_coords_full
        test_re_mach = test_re_mach_full
    
    print(f"Test samples: {len(test_features)} (randomly selected {num_test_samples} from {len(test_features_full)})")
    
    # Create dataset and loader (using dummy af512 for dataset structure)
    dummy_af512 = np.zeros((len(test_features), 1024))
    test_dataset = AeroToAF512Dataset(test_features, dummy_af512, add_noise=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Run full pipeline tests
    print("\n" + "="*60)
    pipeline_name = "Aero → AF512"
    if correction_model is not None:
        pipeline_name += " → Correction"
    pipeline_name += " → XY"
    print(f"Testing Full Pipeline: {pipeline_name}")
    print("="*60)
    
    pipeline_results = test_full_pipeline(
        aero_model, af512_to_xy_model, test_loader, test_xy_coords, test_features,
        correction_model=correction_model, normalization_data=normalization_data, device=device
    )
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_predictions_full_pipeline(
        aero_model, af512_to_xy_model, test_features, test_xy_coords,
        correction_model=correction_model, normalization_data=normalization_data,
        device=device, num_samples=20
    )
    visualize_error_distribution(pipeline_results)
    
    # Save predictions to .dat files
    print("\nSaving predictions to .dat files...")
    save_predictions_to_dat(pipeline_results, test_features, num_samples=10)
    
    # Run NeuralFoil analysis
    print("\n" + "="*60)
    print("Running NeuralFoil Analysis")
    print("="*60)
    neuralfoil_results = run_neuralfoil_analysis(
        test_features, test_re_mach, pipeline_results, device=device, num_samples=100
    )
    
    print("\n" + "="*60)
    print("Testing complete! Check the 'test_results' directory for outputs.")
    print("="*60)


if __name__ == "__main__":
    main()

