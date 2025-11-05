# FoilGen - Interactive Airfoil Design Web Application

**FoilGen** is a web-based airfoil design tool that provides an intuitive interface for creating custom airfoils using neural networks. It is a continuation and enhancement of [FoilNet](https://github.com/AvnehSBhatia/FoilNet), offering:

- 🚀 **Faster creation times** - Real-time airfoil generation through optimized neural networks
- 🎨 **Enhanced customizability** - Interactive web interface with spline-based curve design
- 🌐 **Web interface** - No command-line knowledge required; accessible from any browser
- 📊 **Visual feedback** - Interactive charts with NeuralFoil comparison overlay

## What is FoilGen?

FoilGen uses a two-stage neural network pipeline to convert aerodynamic performance specifications into actual airfoil shapes:

1. **Aero → AF512**: Converts aerodynamic parameters (Reynolds number, Mach number, Cl, Cd, Cl/Cd curves, thickness) into an intermediate `AF512` format
2. **AF512 → XY**: Converts the `AF512` representation into actual X-Y coordinates for the airfoil

The `AF512` intermediate format (similar to FoilNet) provides:
- A reduction layer that makes optimization easier for the genetic algorithm
- Noise filtering capabilities through the neural network
- More efficient learning compared to direct XY coordinate manipulation

## Features

- **Interactive Web Interface**: Modern, responsive design with real-time updates
- **Spline Interpolation**: Smooth curve generation from sparse control points using cubic splines
- **Click-to-Add Points**: Add control points directly on charts or type them manually
- **NeuralFoil Integration**: Automatic comparison with NeuralFoil predictions overlaid on your curves
- **Duplicate Prevention**: Automatic filtering ensures only one point per alpha value
- **Reynolds Number Binning**: Intelligent graph scaling based on flight conditions
- **Export Capabilities**: Download generated airfoils as `.dat` files

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- SciPy
- Matplotlib
- Flask
- NeuralFoil
- Plotly.js (included via CDN)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd FoilML
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Additionally, install Flask and other web dependencies:

```bash
pip install Flask torch numpy scipy matplotlib neuralfoil tqdm
```

### 3. Train the Models (If Needed)

The model files should already be present in the `FoilGen/` directory. If you need to retrain:

```bash
# Train the AF512 to XY coordinate model
python train_airfoil.py

# Train the aerodynamic parameters to AF512 model
python train_aero_to_af512.py
```

After training, copy the generated model files to `FoilGen/`:

```bash
cp af512_to_xy_model.pth FoilGen/
cp aero_to_af512_model.pth FoilGen/
cp feature_normalization.npy FoilGen/
```

### 4. Navigate to FoilGen Directory

```bash
cd FoilGen
```

### 5. Run the Web Application

```bash
python app.py
```

The server will start on `http://localhost:8090`

## Usage

### Basic Workflow

1. **Enter Flight Conditions**
   - Input chord length (feet) and speed (mph)
   - The system automatically computes Reynolds number and Mach number

2. **Set Angle of Attack Range**
   - Define the minimum and maximum angles (increments of 0.5°)

3. **Define Lift Coefficient (Cl) Curve**
   - Add control points by:
     - Typing `alpha,value` (e.g., `5.0,1.2`) in the input field
     - Clicking directly on the chart
   - Points are automatically interpolated using cubic splines
   - Duplicate alpha values are automatically replaced

4. **Define Lift-to-Drag Ratio (Cl/Cd) Curve**
   - Follow the same process as Cl curve

5. **Set Thickness Constraints**
   - Enter minimum and maximum thickness (as chord fraction)

6. **Generate Airfoil**
   - Click "Generate Airfoil" to create your custom airfoil
   - NeuralFoil analysis runs automatically and results are overlaid on the curves
   - Download the generated `.dat` file

### Advanced Features

- **Reynolds Number Binning**: Graph scaling automatically adjusts based on Reynolds number bins (50k, 100k, 250k, 500k, 750k)
- **Spline Interpolation**: Uses cubic splines for 4+ points, quadratic for 3 points, linear for 2 points
- **NeuralFoil Comparison**: Automatically runs NeuralFoil analysis on generated airfoils and overlays predictions

## File Structure

```
FoilGen/
├── app.py                          # Flask web application server
├── interactive_airfoil_design.py   # Core model definitions and functions
├── aero_to_af512_model.pth         # Neural network (Aero → AF512)
├── af512_to_xy_model.pth           # Neural network (AF512 → XY)
├── feature_normalization.npy       # Feature normalization data
├── templates/
│   └── index.html                  # Frontend HTML
├── static/
│   ├── style.css                   # Styling
│   └── script.js                   # Frontend JavaScript and interactivity
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── README_WEB.md                   # Detailed web app documentation
├── QUICKSTART.md                   # Quick reference guide
└── output/                         # Generated files (created automatically)
```

## API Endpoints

- `GET /` - Main web page
- `POST /api/compute_reynolds` - Compute Reynolds and Mach numbers from chord and speed
- `POST /api/generate_alpha` - Generate alpha (angle of attack) vector
- `POST /api/interpolate_points` - Interpolate control points using splines
- `POST /api/generate_airfoil` - Generate airfoil from user inputs
- `GET /api/download_dat` - Download generated `.dat` file

## Comparison with FoilNet

| Feature | FoilNet | FoilGen |
|---------|---------|---------|
| Interface | Command-line | Web-based |
| Design Method | Genetic algorithm optimization | Interactive specification |
| Creation Time | Minutes (genetic algorithm) | Seconds (direct generation) |
| Customization | Limited to optimization objectives | Full control over curves |
| Visualization | Static plots | Interactive charts |
| NeuralFoil Integration | Yes | Yes (with overlay) |
| Learning Curve | Requires Python knowledge | Browser-based, no coding |

## Technical Details

### Model Architecture

**AeroToAF512Net**:
- Takes scalar features (Re, Mach, stats) and variable-length aerodynamic sequences
- Uses LSTM-based sequence encoder for processing aerodynamic curves
- Outputs 1024-dimensional AF512 representation

**AF512toXYNet**:
- Takes 1024-dimensional AF512 input
- Outputs 2048 values (1024 X-coordinates + 1024 Y-coordinates)

### Interpolation

- **Cubic Splines**: Used for 4+ control points with natural boundary conditions
- **Quadratic Splines**: Used for 3 control points
- **Linear Interpolation**: Used for 2 control points
- **Duplicate Handling**: Points with the same alpha (within 0.01° tolerance) are automatically replaced

## Troubleshooting

**Models not found**: Ensure `aero_to_af512_model.pth`, `af512_to_xy_model.pth`, and `feature_normalization.npy` are in the `FoilGen/` directory. If missing, run the training scripts from the parent directory.

**Port already in use**: Change the port in `app.py` (default: 8090)

**NeuralFoil errors**: Ensure NeuralFoil is installed: `pip install neuralfoil`. Some edge cases may cause analysis failures, but the airfoil generation will still work.

**Memory issues**: The models are loaded into memory on startup. Ensure you have sufficient RAM (models are ~8MB total).

## Acknowledgments

- **Mike Quayle** and **bigfoil.com** - Provided the airfoil data used for training the neural networks
- **Peter D. Sharpe** - Creator of [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) and [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox), which are core components of this project
- **FoilNet** - The original project that inspired this web-based implementation

## License

This project is a research tool. Use at your own risk for any applications.

## Contributing

Contributions are welcome! This is a continuation of the FoilNet project, and any improvements are appreciated.

## Related Projects

- [FoilNet](https://github.com/AvnehSBhatia/FoilNet) - Original genetic algorithm-based airfoil optimizer
- [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) - Fast aerodynamic analysis using neural networks
- [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox) - Aircraft design optimization framework
- [bigfoil.com](http://bigfoil.com) - Airfoil database and resources
