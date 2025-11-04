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
from scipy.interpolate import interp1d, UnivariateSpline
from train_aero_to_af512 import (
    normalize_features,
    AeroToAF512Net,
    collate_fn
)
from train_airfoil import (
    AF512toXYNet,
    predict_xy_coordinates,
    load_dat_file
)
from torch.utils.data import DataLoader
from tqdm import tqdm
try:
    import neuralfoil as nf
    NEURALFOIL_AVAILABLE = True
except ImportError:
    NEURALFOIL_AVAILABLE = False
    print("Warning: NeuralFoil not installed. Install with: pip install neuralfoil")


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


def run_pipeline(aero_model, af512_to_xy_model, feature_dict, normalization_data, device='mps'):
    """
    Run the full pipeline: User inputs → Aero → AF512 → XY coordinates.
    
    Returns:
        pred_x, pred_y: Predicted airfoil coordinates
    """
    # Normalize features
    normalized_features, min_vals, max_vals = normalize_user_features(feature_dict, normalization_data)
    
    # Convert to tensors
    scalars_tensor = torch.FloatTensor(normalized_features['scalars']).unsqueeze(0).to(device)
    sequence_tensor = torch.FloatTensor(normalized_features['sequence']).unsqueeze(0).to(device)
    lengths_tensor = torch.tensor([len(normalized_features['sequence'])], dtype=torch.long).to(device)
    
    # Step 1: Aero → AF512
    aero_model.eval()
    with torch.no_grad():
        pred_af512_tensor = aero_model(scalars_tensor, sequence_tensor, lengths_tensor)
        pred_af512_flat = pred_af512_tensor.cpu().numpy().flatten()
    
    # Step 2: AF512 → XY
    pred_x, pred_y = predict_xy_coordinates(af512_to_xy_model, pred_af512_flat, device=device)
    
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


def plot_aerodynamic_curves(alpha, cl, cd, cl_cd, save_path=None):
    """Plot Cl, Cd, and Cl/Cd vs alpha."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Cl vs alpha
    axes[0].plot(alpha, cl, 'b-o', linewidth=2, markersize=4)
    axes[0].set_xlabel('Angle of Attack (deg)')
    axes[0].set_ylabel('Cl')
    axes[0].set_title('Lift Coefficient vs AoA')
    axes[0].grid(True, alpha=0.3)
    
    # Cd vs alpha
    axes[1].plot(alpha, cd, 'r-o', linewidth=2, markersize=4)
    axes[1].set_xlabel('Angle of Attack (deg)')
    axes[1].set_ylabel('Cd')
    axes[1].set_title('Drag Coefficient vs AoA')
    axes[1].grid(True, alpha=0.3)
    
    # Cl/Cd vs alpha
    axes[2].plot(alpha, cl_cd, 'g-o', linewidth=2, markersize=4)
    axes[2].set_xlabel('Angle of Attack (deg)')
    axes[2].set_ylabel('Cl/Cd')
    axes[2].set_title('Lift-to-Drag Ratio vs AoA')
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
    """Interactive matplotlib plot for entering points by clicking."""
    def __init__(self, title, xlabel, ylabel, alpha_range):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.alpha_range = alpha_range
        self.points = []  # List of (alpha, value) tuples
        self.spline_smoothing = 0.0  # Spline smoothing parameter (0 = interpolation, >0 = smoothing)
        self.zoom_state = None  # Store zoom state (xlim, ylim) when zoomed
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        # Enable matplotlib toolbar for zoom/pan (should be enabled by default)
        
        self.ax.set_title(f"{title}\nLeft-click to add points | Right-click to zoom/pan | 'd' to delete | 'c' to clear | 'q' to finish", 
                         fontsize=11)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlim(alpha_range[0] - 1, alpha_range[1] + 1)
        self.ax.grid(True, alpha=0.3)
        
        self.line = None
        self.point_plot = None
        self.update_plot()
        
        # Connect events - only connect left-click, let matplotlib handle right-click for zoom/pan
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onkey)
        
        # Store original toolbar
        self.toolbar = self.fig.canvas.toolbar
        
    def onclick(self, event):
        """Handle mouse clicks."""
        if event.inaxes != self.ax:
            return
        # Only left-click adds points, right-click and middle-click are for zoom/pan
        if event.button == 1:  # Left click only
            alpha = event.xdata
            value = event.ydata
            self.points.append((alpha, value))
            # Sort by alpha
            self.points.sort(key=lambda x: x[0])
            self.update_plot()
            self.fig.canvas.draw()
    
    def onkey(self, event):
        """Handle keyboard events."""
        if event.key == 'd':  # Delete last point
            if self.points:
                self.points.pop()
                self.update_plot()
                self.fig.canvas.draw()
        elif event.key == 'c':  # Clear all
            self.points = []
            self.update_plot()
            self.fig.canvas.draw()
        elif event.key == 'q':  # Quit
            plt.close(self.fig)
        elif event.key == 'r':  # Reset view
            self.zoom_state = None  # Clear zoom state
            self.ax.set_xlim(self.alpha_range[0] - 1, self.alpha_range[1] + 1)
            if len(self.points) > 0:
                values = np.array([p[1] for p in self.points])
                if len(values) > 0:
                    y_min, y_max = values.min(), values.max()
                    y_range = y_max - y_min
                    if y_range > 0:
                        padding = y_range * 0.1
                        self.ax.set_ylim(y_min - padding, y_max + padding)
            self.fig.canvas.draw()
    
    def update_plot(self):
        """Update the plot with current points."""
        # Store current view limits if zoomed (before clearing)
        try:
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
            default_xlim = (self.alpha_range[0] - 1, self.alpha_range[1] + 1)
            # Check if zoomed (not at default view)
            is_zoomed = (abs(current_xlim[0] - default_xlim[0]) > 0.1 or 
                        abs(current_xlim[1] - default_xlim[1]) > 0.1)
            if is_zoomed:
                self.zoom_state = (current_xlim, current_ylim)
            else:
                self.zoom_state = None
        except:
            is_zoomed = False
            # Use stored zoom state if available
            if self.zoom_state is not None:
                current_xlim, current_ylim = self.zoom_state
                is_zoomed = True
            else:
                current_xlim = None
                current_ylim = None
        
        self.ax.clear()
        self.ax.set_title(f"{self.title}\nLeft-click to add points | Right-click to zoom/pan | 'd' to delete | 'c' to clear | 'q' to finish", 
                         fontsize=11)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.grid(True, alpha=0.3)
        
        if len(self.points) > 0:
            alphas = np.array([p[0] for p in self.points])
            values = np.array([p[1] for p in self.points])
            
            # Plot points
            self.ax.plot(alphas, values, 'bo', markersize=10, label='Control Points', zorder=5)
            
            # Plot spline interpolation if more than one point
            if len(self.points) > 1:
                # Show spline interpolated preview on alpha vector
                alpha_vec = np.linspace(self.alpha_range[0], self.alpha_range[1], 200)
                try:
                    interp_values = self.interpolate(alpha_vec)
                    self.ax.plot(alpha_vec, interp_values, 'g-', linewidth=2, alpha=0.8, label='Spline Interpolation', zorder=2)
                except Exception as e:
                    # Fallback to linear if spline fails
                    try:
                        from scipy.interpolate import interp1d
                        interp_func = interp1d(alphas, values, kind='linear', bounds_error=False, fill_value='extrapolate')
                        interp_values = interp_func(alpha_vec)
                        self.ax.plot(alpha_vec, interp_values, 'g--', linewidth=2, alpha=0.7, label='Linear Interpolation (fallback)', zorder=2)
                    except:
                        pass
            
            # Auto-scale y-axis only if not zoomed
            if not is_zoomed:
                if len(values) > 0:
                    y_min, y_max = values.min(), values.max()
                    y_range = y_max - y_min
                    if y_range > 0:
                        padding = y_range * 0.1
                        self.ax.set_ylim(y_min - padding, y_max + padding)
                    else:
                        self.ax.set_ylim(y_min - 0.1, y_max + 0.1)
                # Set x-axis limits
                self.ax.set_xlim(self.alpha_range[0] - 1, self.alpha_range[1] + 1)
            else:
                # Restore zoom
                self.ax.set_xlim(current_xlim)
                self.ax.set_ylim(current_ylim)
        
        else:
            # Initial view
            self.ax.set_xlim(self.alpha_range[0] - 1, self.alpha_range[1] + 1)
        
        self.ax.legend(loc='best')
    
    def interpolate(self, alpha_vector):
        """Interpolate values for given alpha vector using splines."""
        if len(self.points) == 0:
            raise ValueError("No points have been added. Please add at least one point.")
        
        alphas = np.array([p[0] for p in self.points])
        values = np.array([p[1] for p in self.points])
        
        # Sort by alpha
        sort_idx = np.argsort(alphas)
        alphas = alphas[sort_idx]
        values = values[sort_idx]
        
        alpha_vector = np.array(alpha_vector)
        
        # If only one point, return constant value
        if len(self.points) == 1:
            return np.full_like(alpha_vector, values[0])
        
        # Use cubic spline interpolation
        # s=0 means interpolation (passes through all points)
        # For smoothing, increase s (e.g., s=len(alphas) for moderate smoothing)
        try:
            # Try cubic spline first (k=3)
            spline = UnivariateSpline(alphas, values, s=self.spline_smoothing, k=min(3, len(alphas)-1))
            interpolated = spline(alpha_vector)
        except:
            # If cubic fails (e.g., too few points), fall back to linear
            interp_func = interp1d(alphas, values, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            interpolated = interp_func(alpha_vector)
        
        return interpolated
    
    def show(self):
        """Show the plot and wait for user to finish."""
        print(f"\nInteractive plot: {self.title}")
        print("Instructions:")
        print("  - Left-click to add points")
        print("  - Right-click or middle-click to zoom/pan (use toolbar)")
        print("  - Press 'd' to delete last point")
        print("  - Press 'c' to clear all points")
        print("  - Press 'r' to reset zoom")
        print("  - Press 'q' or close window when done")
        print(f"  Current points: {len(self.points)}")
        print("  Using cubic spline interpolation")
        
        # Ensure toolbar is visible for zoom/pan
        try:
            # Try to show toolbar (works for most backends)
            if hasattr(self.fig.canvas, 'toolbar') and self.fig.canvas.toolbar:
                self.fig.canvas.toolbar.update()
        except:
            pass
        
        # Use block=True to wait for window to close
        plt.show(block=True)
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
    
    # Get Cl vector using interactive plot
    print("\n3. Enter Cl values (lift coefficient) - Interactive Plot")
    cl_plot = InteractivePlot("Cl vs Angle of Attack", "Angle of Attack (deg)", "Cl", (a_min, a_max))
    cl_plot.show()
    
    if len(cl_plot.points) == 0:
        raise ValueError("No Cl points were added. Please add at least one point.")
    
    cl = cl_plot.interpolate(alpha)
    print(f"   Interpolated {len(cl)} Cl values from {len(cl_plot.points)} points")
    
    # Get Cl/Cd vector using interactive plot
    print("\n4. Enter Cl/Cd values (lift-to-drag ratio) - Interactive Plot")
    cl_cd_plot = InteractivePlot("Cl/Cd vs Angle of Attack", "Angle of Attack (deg)", "Cl/Cd", (a_min, a_max))
    cl_cd_plot.show()
    
    if len(cl_cd_plot.points) == 0:
        raise ValueError("No Cl/Cd points were added. Please add at least one point.")
    
    cl_cd = cl_cd_plot.interpolate(alpha)
    print(f"   Interpolated {len(cl_cd)} Cl/Cd values from {len(cl_cd_plot.points)} points")
    
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
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"\nUsing device: {device}\n")
    
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
    af512_to_xy_model.load_state_dict(torch.load(af512_to_xy_model_file, map_location=device))
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
    
    # Generate plots
    print("\nGenerating plots...")
    plot_airfoil_shape(pred_x, pred_y, save_path=os.path.join(output_dir, 'airfoil_shape.png'))
    plot_aerodynamic_curves(
        user_data['alpha'], user_data['cl'], user_data['cd'], user_data['cl_cd'],
        save_path=os.path.join(output_dir, 'aerodynamic_curves.png')
    )
    
    # Run NeuralFoil comparison
    if NEURALFOIL_AVAILABLE:
        print("\n" + "="*60)
        print("Running NeuralFoil Comparison")
        print("="*60)
        nf_results = run_neuralfoil_comparison(
            pred_x, pred_y, user_data['alpha'], user_data['cl'], 
            user_data['cd'], user_data['cl_cd'], user_data['re'], 
            user_data['mach'], save_dir=output_dir
        )
    
    print("\n" + "="*60)
    print("All outputs saved to 'output' directory!")
    print("="*60)


if __name__ == "__main__":
    main()
