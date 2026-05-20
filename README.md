# brain-morph

Mesh-based 3D brain image registration for mesoscale developmental studies.
PyTorch backend with GPU support.

**Author:** Ivan Doronin  
Python reimplementation of an algorithm originally developed by
**Sergey Shuvaev** (Cold Spring Harbor Laboratory, 2014–2021).

## Installation

```bash
pip install brain-morph          # core (torch, numpy, opencv, matplotlib)
pip install "brain-morph[io]"    # + NIfTI support (nibabel)
```

> **GPU:** Install a CUDA-enabled PyTorch first — see [pytorch.org](https://pytorch.org/get-started/locally/).
> Then `pip install brain-morph` will reuse your existing torch.

## Quick start

```python
import torch
from brain_morph.utils import Volume, MeshTransformer3D
from brain_morph.registration import (
    Stage, RegistrationPipeline, SAOptimizer,
)

# Load two brain volumes (D, H, W)
im_moving = Volume.load_nii("brain_moving.nii.gz", scale=4)
im_fixed  = Volume.load_nii("brain_fixed.nii.gz",  scale=4)

# Define registration pipeline (coarse → fine)
pipeline = RegistrationPipeline([
    Stage(grid_shape=(4, 4, 4), optimizer=SAOptimizer(), n_steps=2000, lam=1e-3),
    Stage(grid_shape=(8, 8, 8), optimizer=SAOptimizer(), n_steps=2000, lam=1e-3),
])

grid = pipeline.run(im_moving, im_fixed)

# Apply deformation to get warped volume
transformer = MeshTransformer3D(grid, im_moving.shape)
warped = transformer.transform(im_moving, grid)
```

See [notebooks/registration_demo.ipynb](notebooks/registration_demo.ipynb) for a full walkthrough.

## Key features

- **Coarse-to-fine pipeline** — multi-stage registration with interpolated grids between stages
- **Simulated annealing optimizer** — attention-gated SA with adaptive step size and ROI updates
- **Gradient optimizer** — Adam / SGD / LBFGS via `torch.optim`
- **Hybrid optimizer** — global SA search followed by gradient refinement
- **Deformation regularization** — volumetric (5-tetrahedra) and bending energy metrics
- **Tissue masking** — per-cell weights suppress regularization in empty regions
- **NIfTI and TIFF** input support

## Testing

```bash
pip install "brain-morph[io,dev]"
pytest tests/ -v
```

## Acknowledgements

Original algorithm and MATLAB implementation by
**Sergey Shuvaev** (CSHL, 2014–2021), licensed under GPL v3.0.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
