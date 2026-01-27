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

The `Volume` class provides the foundation for all 3D image operations:
```python
from src.utils.volume import Volume

# Load and manage 3D brain images
volume = Volume(image_tensor)
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
