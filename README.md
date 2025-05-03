# Gestalt Vector Field Theory (GVFT) Simulation Framework

A computational framework for simulating Gestalt Vector Field Theory - a field-theoretic approach to modular cognition that uses continuous fields to model and generate neural network architectures. 

## Overview

GVFT represents neural architectures using continuous fields defined over a spatial domain rather than fixed graph structures. The simulation framework models:

1. **Flow Field (F)**: Vector field representing direction of information flow
2. **Strength Field (W)**: Scalar field representing connection strength
3. **Neuromodulatory Field (η)**: Scalar field representing modulatory signals

These fields interact through coupled reaction-diffusion dynamics, resulting in emergent patterns and structures.

## Features

- GPU-accelerated field evolution using PyTorch
- NeuroML support for using biological connectome data as priors
- Parameter sweeps with multiple evaluation metrics
- Modular structure formation with dynamic sampling
- Comprehensive visualization suite
- Multi-metric pattern analysis

## Requirements

### Dependencies

```
torch>=1.12.0
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
seaborn>=0.11.0
pandas>=1.3.0
lxml>=4.6.0 (for NeuroML processing)
```

A GPU is recommended but not required. The code will automatically use CUDA if available.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gvft.git
cd gvft
```

2. Create and activate a virtual environment:
```bash
python3 -m venv gvft-env
source gvft-env/bin/activate  # On Windows, use: gvft-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Simulation

Run a basic GVFT simulation:

```bash
cd sim
python main.py
```

### Using Biological Data as Priors

Generate GVFT fields from NeuroML2 connectome data:

```bash
python neuroml_to_gvft.py /path/to/neuroml_data output_directory
```

Run simulation with biological priors:

```bash
cd sim
python main.py --neuroml-fields ../source-fields --neuroml-basename PharyngealNetwork
```

### Command-Line Options

#### Main Simulation Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--grid-size` | 200 | Size of the grid for simulation |
| `--no-scaling` | False | Disable parameter scaling (maintains consistent dynamics across grid sizes) |
| `--output-dir` | ../figures | Directory to save output visualizations |
| `--full-sims-only` | False | Skip parameter sweep, run full simulations only |
| `--sweep-only` | False | Run parameter sweep only, skip full simulations |
| `--save-checkpoints` | False | Save checkpoints during parameter sweep |
| `--save-intermediates` | False | Save intermediate visualizations during parameter sweep |

#### Biological Prior Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--neuroml-fields` | None | Directory containing preprocessed NeuroML GVFT fields |
| `--neuroml-basename` | PharyngealNetwork | Base name of the processed NeuroML files |

#### Analysis Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--primary-metric` | pattern_quality | Metric to use for parameter selection (`hotspot_fraction`, `pattern_persistence`, `structural_complexity`, `flow_coherence`, `pattern_quality`) |

### NeuroML Conversion Options:

The `neuroml_to_gvft.py` script has the following options:

| Option | Default | Description |
|--------|---------|-------------|
| `--grid-size` | 200 | Size of the grid for field generation |
| `--sigma-flow-potential` | 15.0 | Sigma for flow field potential smoothing |
| `--sigma-strength-density` | 10.0 | Sigma for synaptic strength field density |
| `--sigma-eta-density` | 15.0 | Sigma for neuromodulatory field density |
| `--activity-threshold` | 0.05 | Minimum threshold for activity sources |
| `--no-vis` | False | Disable visualization output |

## Configuration Parameters

The simulation behavior can be further customized by editing `config.py`:

### Field Dynamics Parameters:

| Parameter | Description |
|-----------|-------------|
| `D_W` | Diffusion coefficient for strength field |
| `D_eta` | Diffusion coefficient for neuromodulatory field |
| `lam_F` | Decay rate for flow field |
| `lam_eta` | Decay rate for neuromodulatory field |
| `alpha` | Coupling strength from flow to strength field |
| `beta` | Weight for flow magnitude in module placement |
| `gamma` | Weight for strength field in module placement |
| `beta_coupling` | Strength of gradient coupling in flow field updates |
| `eta_coeff` | Neuromodulatory feedback strength |
| `noise_F` | Noise level in flow field |
| `noise_W` | Noise level in strength field |

### Module Sampling Parameters:

| Parameter | Description |
|-----------|-------------|
| `num_modules` | Number of modules to sample |
| `top_k` | Maximum connections per module |
| `cos_threshold` | Minimum cosine similarity for connections |
| `lambda_val` | Regularization for connection weights |

## Understanding Output Metrics

The framework provides several metrics to evaluate pattern formation:

1. **Hotspot Fraction**: Fraction of grid points where W exceeds threshold
2. **Pattern Persistence**: Correlation between initial and current field states
3. **Structural Complexity**: Spatial frequency distribution analysis
4. **Flow Coherence**: Directional organization of the flow field
5. **Pattern Quality**: Weighted combination of the above metrics

## Examples

### Basic Parameter Sweep

```bash
python main.py --primary-metric pattern_persistence
```

### Full Simulation with Biological Priors

```bash
python main.py --neuroml-fields ../source-fields --full-sims-only
```

### High-Resolution Experiment

```bash
python main.py --grid-size 300 --save-checkpoints --save-intermediates
```

## Visualization

The framework produces several types of visualizations:

1. **Phase Diagrams**: Heatmaps showing how parameters affect metrics
2. **Field Visualizations**: Snapshots of fields at different timesteps
3. **Connectivity Graphs**: Emergent module patterns and connections
4. **Metrics Evolution**: Charts tracking metrics over simulation time

## License

[MIT License] - See LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{gvft2025,
  title={Gestalt Vector Field Theory: Toward a Field-Theoretic Framework for Modular Cognition},
  author={Knots},
  year={2025}
}
```

## Troubleshooting

- **CUDA Out of Memory**: Reduce grid size or batch size in `config.py`
- **Slow Simulations**: Enable CUDA, reduce grid size, or disable `--save-intermediates`
- **NeuroML Parsing Errors**: Ensure files follow NeuroML2 format specification