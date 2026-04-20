# Brain Morphology Registration

Spatial and temporal 3D image registration for mesoscale studies of brain development.

**Original concept and MATLAB implementation:** Sergey Shuvaev  
**Python implementation:** IADoronin

## Overview

This project provides tools for registering 3D brain images to a common coordinate system, enabling analysis of brain development dynamics. The Python implementation refactors the original MATLAB code with a focus on modularity and efficient tensor-based operations using PyTorch.

### Key Features

- **3D Mesh-based Warping**: Uses control-point displacement to warp 3D volumes
- **Modular Architecture**: Built around a base `Volume` class for better code organization
- **GPU Support**: PyTorch backend enables GPU acceleration
- **Precision Modes**: Multiple interpolation modes (exact/trilinear, nearest)

## Installation

### Requirements

- Python 3.8+
- PyTorch
- NumPy

### Setup

```bash
pip install torch numpy
```

## Core Components

### 1. **Volume Class** (Base Class)

The `Volume` class extends PyTorch tensors with domain-specific functionality for 3D image processing. It maintains spatial metadata (affine transformations) while supporting all standard tensor operations.

#### Features

- **Tensor Subclassing**: Inherits from `torch.Tensor` for seamless integration with PyTorch ecosystem
- **Affine Metadata**: Stores and preserves spatial transformation matrices (4×4 affine)
- **Type Preservation**: Arithmetic operations return `Volume` objects, not plain tensors
- **GPU Support**: Full CUDA compatibility

#### Methods

##### `visualize(channel=None)`
Display 3D volume using maximum intensity projections along three axes.

```python
brain = Volume(data)
brain.visualize()  # All channels as RGB
brain.visualize(channel=0)  # Single channel
```

Output shows 2×2 grid:
- Top-left: XY projection (max over depth)
- Top-right: XZ projection (max over height)  
- Bottom-left: YZ projection (max over width)

##### `normalize()`
Normalize volume intensity to [0, 1] range using min-max scaling.

```python
brain_normalized = brain.normalize()
```

##### `resample(new_shape, mode='trilinear')`
Resize volume to new dimensions using interpolation.

**Parameters:**
- `new_shape` (tuple): Target shape (D, H, W)
- `mode` (str): Interpolation method - 'nearest', 'linear', 'bilinear', 'trilinear'

```python
brain_downsampled = brain.resample((32, 32, 32), mode='trilinear')
```

##### `rotate(theta, phi, center, interpolation='linear')`
Apply 3D rotation around specified center point.

**Parameters:**
- `theta` (float): Rotation angle around X-axis (degrees)
- `phi` (float): Rotation angle around Y-axis (degrees)
- `center` (tuple): Rotation center in voxel coordinates (x, y, z)
- `interpolation` (str): 'nearest', 'linear', 'bilinear', 'bicubic'

```python
# Rotate 30° around X, 45° around Y, centered at (50, 50, 50)
rotated = brain.rotate(30, 45, (50, 50, 50), 'linear')
```

#### Usage Example

```python
from src.utils import Volume
import torch

# Load or create data
data = torch.rand(3, 256, 256, 256)  # (C, D, H, W)

# Create Volume
brain = Volume(data)

# Preprocessing pipeline
brain = brain.normalize()
brain_downsampled = brain.resample((64, 64, 64))

# Visualization
brain_downsampled.visualize()

# Rotation for augmentation
rotated = brain.rotate(theta=15, phi=20, center=(32, 32, 32))
```

### 2. **Mesh Transform**

Transform 3D volumes using control point meshes:
```python
from src.utils.mesh_transform import mesh_transform, create_regular_mesh

# Create regular control mesh
mesh = create_regular_mesh(num_cells=(4, 4, 4), image_shape=(64, 64, 64), normalized=True)

# Apply transformation
warped = mesh_transform(volume, mesh_initial, mesh_transformed, precision="exact")
```

### 3. **Simulated Annealing**

Optimize mesh control points using simulated annealing:
```python
from src.utils.simulated_annealing import optimize_mesh

optimized_mesh = optimize_mesh(volume_reference, volume_moving)
```

## Usage Workflow

### Step 1: Load Brain Images

Load 3D brain images from TIFF or NIfTI formats:
```python
from src.utils.volume import Volume

# Load image
brain = Volume.from_file('brain_sample.nii')
```

### Step 2: Preprocess

Preprocess images (thresholding, normalization):
```python
brain.preprocess(threshold=0.2)
brain.normalize()
```

### Step 3: Create Registration Mesh

Define control points for registration:
```python
mesh_initial = create_regular_mesh((4, 4, 4), brain.shape)
```

### Step 4: Register to Template

Register individual brain to template using mesh deformation:
```python
mesh_optimized = optimize_mesh(brain_template, brain_sample)
brain_registered = mesh_transform(brain_sample, mesh_initial, mesh_optimized)
```

### Step 5: Analyze Development Dynamics

Visualize and analyze changes over developmental time:
```python
# Compute intensity changes
diff = registered_t1 - registered_t0
```

## API Reference

### `create_regular_mesh(num_cells, image_shape, normalized=True)`

Creates a regular control mesh for warping.

**Parameters:**
- `num_cells` (tuple): Number of control points in each dimension
- `image_shape` (tuple): Shape of the volume (D, H, W)
- `normalized` (bool): If True, coordinates in [-1, 1]; else in voxel coordinates

**Returns:**
- `torch.Tensor`: Mesh of shape (3, nx, ny, nz)

### `mesh_transform(volume, mesh_initial, mesh_transformed, precision="exact", align_corners=True, padding_mode="border")`

Warps a 3D volume using mesh-based displacement.

**Parameters:**
- `volume` (torch.Tensor): Input volume (D,H,W) or (C,D,H,W) or (N,C,D,H,W)
- `mesh_initial` (torch.Tensor): Initial control mesh
- `mesh_transformed` (torch.Tensor): Target control mesh
- `precision` (str): "exact" (trilinear) or "nearest"
- `align_corners` (bool): Grid alignment mode
- `padding_mode` (str): "border", "zeros", or "reflection"

**Returns:**
- `torch.Tensor`: Transformed volume with same shape as input

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

Test coverage includes:
- Identity transformations
- Random displacements
- Shape and dtype preservation

## Architecture Differences from MATLAB

| MATLAB | Python |
|--------|--------|
| Separate scripts | Unified `Volume` class |
| Global image variables | Object-oriented design |
| Interactive GUI dialogs | Programmatic API |
| CPU-only | GPU-compatible |

## Performance Notes

- Typical registration time: 5-10 minutes per brain (GPU)
- Memory usage: ~2GB for (512, 512, 256) volumes
- Supports batch processing for multiple brains

## Contributing

Improvements and extensions welcome. Please ensure all tests pass before submitting changes.

## References

- Original MATLAB implementation: Sergey Shuvaev
- PyTorch mesh warping: Uses `F.grid_sample` for efficient interpolation
- Coordinate normalization: Automatic detection of voxel vs. normalized coordinates

## License

[Add appropriate license]
