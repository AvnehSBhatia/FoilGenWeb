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

# Import functions from the original training script
from train_airfoil import (
    load_dat_file,
    naca_to_af512,
    af512_to_coordinates
)

def load_csv_data(csv_dir='unpacked_csv', bigfoil_dir='bigfoil', num_points=512):
    """
    Load CSV files and match them with corresponding .dat files to get AF512 representations.
    Returns feature arrays and AF512 target arrays.
    """
    csv_path = Path(csv_dir)
    bigfoil_path = Path(bigfoil_dir)
    
    if not csv_path.exists():
        print(f"Directory {csv_dir} not found!")
        return None, None
    
    csv_files = list(csv_path.glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files")
    
    features_list = []
    af512_list = []
    
    # Cache for airfoil AF512 data (same airfoil name -> same AF512)
    airfoil_cache = {}
    
    for csv_file in tqdm(csv_files, desc="Loading CSV files"):
        try:
            # Get airfoil name from CSV filename (e.g., "vr8.csv" -> "vr8")
            airfoil_name = csv_file.stem
            
            # Try to find corresponding .dat file
            dat_file = bigfoil_path / f"{airfoil_name}.dat"
            if not dat_file.exists():
                # Try uppercase or other variations
                dat_file = bigfoil_path / f"{airfoil_name.upper()}.dat"
                if not dat_file.exists():
                    # Try with variations
                    possible_names = [
                        airfoil_name,
                        airfoil_name.upper(),
                        airfoil_name.lower(),
                    ]
                    found = False
                    for name in possible_names:
                        test_file = bigfoil_path / f"{name}.dat"
                        if test_file.exists():
                            dat_file = test_file
                            found = True
                            break
                    if not found:
                        print(f"No .dat file found for {airfoil_name}, skipping...")
                        continue
            
            # Load or get cached AF512 data
            if airfoil_name not in airfoil_cache:
                af512, xy = load_dat_file(dat_file, num_points)
                if af512 is None or xy is None:
                    print(f"Failed to load {dat_file}, skipping...")
                    continue
                airfoil_cache[airfoil_name] = af512.flatten()
            
            af512_data = airfoil_cache[airfoil_name]
            
            # Load CSV data
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # Extract features for each row
            for idx, row in enumerate(tqdm(rows, desc=f"Processing {csv_file.name}", leave=False)):
                try:
                    # Extract features: Re, Mach, and summary statistics
                    def safe_float(val, default=0.0):
                        try:
                            return float(val) if val and val != 'nan' and val != 'NaN' else default
                        except:
                            return default
                    
                    re = safe_float(row.get('Re', 0))
                    mach = safe_float(row.get('Mach', 0))
                    
                    # Get summary statistics
                    ldmax = safe_float(row.get('LDMax', 0))
                    clmax = safe_float(row.get('ClMax', 0))
                    cdmin = safe_float(row.get('CdMin', 0))
                    alpha_clmax = safe_float(row.get('alpha_ClMax', 0))
                    alpha_cdmin = safe_float(row.get('alpha_CdMin', 0))
                    alpha_ldmax = safe_float(row.get('alpha_LDMax', 0))
                    min_thickness = safe_float(row.get('min_thickness', 0))
                    max_thickness = safe_float(row.get('max_thickness', 0))
                    
                    # Parse alpha, Cl, Cd arrays (vectors) - stored as string representations of lists
                    try:
                        alpha_str = str(row.get('alpha', '[]')).strip()
                        cl_str = str(row.get('Cl', '[]')).strip()
                        cd_str = str(row.get('Cd', '[]')).strip()
                        cl_cd_str = str(row.get('Cl_Cd', '[]')).strip()
                        
                        # Parse arrays - they're stored as string representations like "[1.0, 2.0, 3.0]"
                        def parse_vector(vec_str):
                            """Parse a vector string to numpy array."""
                            if not vec_str or vec_str == '':
                                return np.array([])
                            # Check if it's a list format
                            if vec_str.startswith('[') and vec_str.endswith(']'):
                                try:
                                    # Strip brackets and split by comma, then convert to float
                                    lst = [float(x.strip()) for x in vec_str.strip("[]").split(",")]
                                    return np.array(lst, dtype=np.float32)
                                except:
                                    return np.array([])
                            else:
                                # Try to parse as single float
                                try:
                                    return np.array([float(vec_str.strip())], dtype=np.float32)
                                except:
                                    return np.array([])
                        
                        alpha_arr = parse_vector(alpha_str)
                        cl_arr = parse_vector(cl_str)
                        cd_arr = parse_vector(cd_str)
                        cl_cd_arr = parse_vector(cl_cd_str)
                        
                        # Ensure all arrays have the same length (use minimum length)
                        min_len = min(len(alpha_arr), len(cl_arr), len(cd_arr), len(cl_cd_arr))
                        if min_len == 0:
                            # Skip if any vector is empty
                            continue
                        
                        # Truncate to minimum length if needed
                        if len(alpha_arr) > min_len:
                            alpha_arr = alpha_arr[:min_len]
                        if len(cl_arr) > min_len:
                            cl_arr = cl_arr[:min_len]
                        if len(cd_arr) > min_len:
                            cd_arr = cd_arr[:min_len]
                        if len(cl_cd_arr) > min_len:
                            cl_cd_arr = cl_cd_arr[:min_len]
                        
                    except Exception as e:
                        # Skip this row if parsing fails
                        continue
                    
                    # Store vectors as-is (variable length) - will be encoded by sequence encoder
                    # Store scalar features separately
                    scalar_features = np.array([
                        re, mach, ldmax, clmax, cdmin, alpha_clmax, alpha_cdmin, alpha_ldmax, 
                        min_thickness, max_thickness
                    ], dtype=np.float32)
                    
                    # Store sequence data: [alpha, cl, cd, cl_cd] as a 4-channel sequence
                    # Shape: (seq_len, 4) where seq_len is variable
                    sequence_data = np.column_stack([alpha_arr, cl_arr, cd_arr, cl_cd_arr])
                    
                    # Check for invalid values
                    if (np.any(np.isnan(scalar_features)) or np.any(np.isinf(scalar_features)) or
                        np.any(np.isnan(sequence_data)) or np.any(np.isinf(sequence_data))):
                        continue
                    
                    # Store both scalar features and sequence data
                    features_list.append({
                        'scalars': scalar_features,
                        'sequence': sequence_data
                    })
                    af512_list.append(af512_data)
                    
                except Exception as e:
                    print(f"Error processing row {idx} in {csv_file.name}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")
            continue
    
    print(f"\nSuccessfully loaded {len(features_list)} samples")
    return features_list, np.array(af512_list)


class AeroToAF512Dataset(Dataset):
    def __init__(self, features, af512_data, add_noise=False, noise_level=0.01):
        self.features = features  # List of dicts with 'scalars' and 'sequence'
        self.af512_data = af512_data
        self.add_noise = add_noise
        self.noise_level = noise_level
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature_dict = self.features[idx]
        scalars = feature_dict['scalars']
        sequence = feature_dict['sequence']
        af512 = self.af512_data[idx]
        
        if self.add_noise:
            scalars = scalars + np.random.normal(0, self.noise_level * np.abs(scalars), scalars.shape)
            sequence = sequence + np.random.normal(0, self.noise_level * np.abs(sequence), sequence.shape)
        
        return {
            'scalars': torch.FloatTensor(scalars),
            'sequence': torch.FloatTensor(sequence)
        }, torch.FloatTensor(af512)


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


def denormalize_features(normalized, min_vals, max_vals):
    """Denormalize features."""
    ranges = max_vals - min_vals
    ranges[ranges < 1e-6] = 1.0
    return normalized * ranges + min_vals


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


def train_model(model, train_loader, val_loader, num_epochs=200, learning_rate=0.001, device='mps', model_file='aero_to_af512_model.pth'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    # Flag to track if interrupted
    interrupted = False
    
    def signal_handler(sig, frame):
        nonlocal interrupted
        print('\n\nInterrupted! Saving model and exiting...')
        interrupted = True
        torch.save(model.state_dict(), model_file)
        print(f'Model saved to {model_file}')
        sys.exit(0)
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"Training on {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print("Press Ctrl+C to save model and exit early")
    
    epoch_pbar = tqdm(range(num_epochs), desc="Training")
    try:
        for epoch in epoch_pbar:
            model.train()
            train_loss = 0.0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train", leave=False)
            for batch_data, targets in train_pbar:
                scalars = batch_data['scalars'].to(device)
                sequence = batch_data['sequence'].to(device)
                lengths = batch_data['lengths'].to(device)  # Can be on device, will convert to CPU in encoder if needed
                targets = targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(scalars, sequence, lengths)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            model.eval()
            val_loss = 0.0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val", leave=False)
            with torch.no_grad():
                for batch_data, targets in val_pbar:
                    scalars = batch_data['scalars'].to(device)
                    sequence = batch_data['sequence'].to(device)
                    lengths = batch_data['lengths'].to(device)  # Can be on device, will convert to CPU in encoder if needed
                    targets = targets.to(device)
                    
                    outputs = model(scalars, sequence, lengths)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.6f}',
                'val_loss': f'{val_loss:.6f}'
            })
            
            if epoch % 10 == 0:
                print(f"\nEpoch {epoch:3d}/{num_epochs}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Check if interrupted
            if interrupted:
                break
        
        # Save model at the end (if not already saved by interrupt)
        if not interrupted:
            torch.save(model.state_dict(), model_file)
            print(f'\nModel saved to {model_file}')
    
    except KeyboardInterrupt:
        print('\n\nKeyboardInterrupt caught! Saving model and exiting...')
        torch.save(model.state_dict(), model_file)
        print(f'Model saved to {model_file}')
        sys.exit(0)
    
    return train_losses, val_losses


def predict_af512(model, scalars, sequence, device='mps'):
    """Predict AF512 from aerodynamic features."""
    model.eval()
    with torch.no_grad():
        # Convert numpy arrays to tensors first
        if not isinstance(scalars, torch.Tensor):
            scalars = torch.FloatTensor(scalars)
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.FloatTensor(sequence)
        
        # Add batch dimension if needed
        if scalars.ndim == 1:
            scalars = scalars.unsqueeze(0)
        if sequence.ndim == 2:
            sequence = sequence.unsqueeze(0)
        
        scalars_tensor = scalars.to(device)
        sequence_tensor = sequence.to(device)
        lengths = torch.tensor([sequence.shape[1]], dtype=torch.long).to(device)  # Can be on device, will convert to CPU in encoder if needed
        
        output = model(scalars_tensor, sequence_tensor, lengths)
        af512_flat = output.cpu().numpy().flatten()
        
        return af512_flat


def af512_to_dat_format(af512_flat, num_points=512):
    """Convert AF512 to .dat format coordinates."""
    af512 = af512_flat.reshape(num_points, 2)
    x_coords, y_coords = af512_to_coordinates(af512)
    return x_coords, y_coords


def test_and_visualize(model, test_features, test_af512, device='mps', num_samples=5):
    """Test the model and visualize predictions."""
    model.eval()
    
    # Select random samples
    num_samples = min(num_samples, len(test_features))
    indices = np.random.choice(len(test_features), num_samples, replace=False)
    
    print(f"Testing on {num_samples} samples...")
    
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 6))
    if num_samples == 1:
        axes = [axes]
    
    for i, idx in enumerate(indices):
        # Get ground truth
        gt_af512 = test_af512[idx]
        gt_x, gt_y = af512_to_dat_format(gt_af512)
        
        # Get prediction
        feature_dict = test_features[idx]
        scalars = feature_dict['scalars']
        sequence = feature_dict['sequence']
        pred_af512 = predict_af512(model, scalars, sequence, device=device)
        pred_x, pred_y = af512_to_dat_format(pred_af512)
        
        # Calculate error
        mse = np.mean((gt_af512 - pred_af512)**2)
        
        # Plot
        ax = axes[i]
        ax.plot(gt_x, gt_y, 'b-', linewidth=2, label='Ground Truth', alpha=0.7)
        ax.plot(pred_x, pred_y, 'r--', linewidth=2, label='Prediction', alpha=0.7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Sample {i+1}\nMSE: {mse:.6f}')
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.tight_layout()
    plt.savefig('aero_to_af512_test.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to aero_to_af512_test.png")
    plt.show()


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data from CSV files...")
    features, af512_data = load_csv_data('unpacked_csv', 'bigfoil', num_points=512)
    
    if features is None or len(features) == 0:
        print("No data loaded!")
        return
    
    # Use only the first half of the dataset
    total_size = len(features)
    half_size = total_size // 2
    features = features[:half_size]
    af512_data = af512_data[:half_size]
    print(f"Using first half of dataset: {half_size} samples (out of {total_size} total)")
    
    # Normalize features
    features_normalized, min_vals, max_vals = normalize_features(features)
    
    # Save normalization parameters
    np.save('feature_normalization.npy', {'min': min_vals, 'max': max_vals})
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_features, val_features, train_af512, val_af512 = train_test_split(
        features_normalized, af512_data, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = AeroToAF512Dataset(train_features, train_af512, add_noise=True, noise_level=0.01)
    val_dataset = AeroToAF512Dataset(val_features, val_af512, add_noise=False)
    
    # Use custom collate function to handle variable-length sequences
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Create model
    model = AeroToAF512Net(
        scalar_input_size=10,  # Re, Mach, LDMax, ClMax, CdMin, alpha_ClMax, alpha_CdMin, alpha_LDMax, min_thickness, max_thickness
        sequence_embedding_dim=256,
        output_size=1024,
        hidden_sizes=[256, 256, 256, 256]
    )
    
    # Train or load model
    model_file = 'aero_to_af512_model.pth'
    if os.path.exists(model_file):
        print(f"Loading existing model from {model_file}")
        try:
            model.load_state_dict(torch.load(model_file, map_location=device))
            model = model.to(device)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Model architecture may have changed. Training new model...")
            train_losses, val_losses = train_model(
                model, train_loader, val_loader,
                num_epochs=200, learning_rate=0.001, device=device, model_file=model_file
            )
    else:
        print("Training new model...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            num_epochs=200, learning_rate=0.001, device=device, model_file=model_file
        )
        # Model is already saved in train_model function
    
    # Test and visualize
    print("\nTesting model...")
    test_and_visualize(model, val_features, val_af512, device=device, num_samples=5)


if __name__ == "__main__":
    main()

