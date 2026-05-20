# brain-morph

Mesh-based 3D brain image registration for mesoscale developmental studies.
PyTorch backend with GPU support.

**Author:** Ivan Doronin  
Developed independently in Python, drawing on the core ideas of
**Sergey Shuvaev**'s (CSHL, 2014–2021) MATLAB work —
mesh-based warping and attention-gated simulated annealing.

## Installation

```bash
pip install brain-morph
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

## Differences from the original MATLAB implementation

- **Deformed reference grid for fixed image.** The original always uses a regular identity grid as the registration reference. This implementation allows passing an arbitrary pre-deformed grid as the reference, enabling sequential or incremental registration (e.g. aligning to a population atlas built from previous registrations).
- **GPU acceleration.** The PyTorch backend runs on any CUDA-capable GPU, giving 10–50× speedup over CPU for large volumes.
- **Gradient-based and hybrid optimizers.** The original supports only simulated annealing. This implementation adds `GradientOptimizer` (Adam / SGD / LBFGS) and `HybridOptimizer` (global SA exploration followed by gradient refinement).
- **Bending energy regularization.** In addition to the original volumetric (5-tetrahedra) deformation energy, a bending energy metric penalises second-order curvature of the displacement field.
- **Multiple similarity metrics.** Supports Pearson correlation (original), normalised cross-correlation (NCC), and MSE.
- **Modular Python API.** Optimizers, metrics, and pipeline stages are independent, composable objects — easy to extend or replace individual components.

## Testing

```bash
pip install "brain-morph[dev]"
pytest tests/ -v
```

## Acknowledgements

Core ideas drawn from:
**Sergey Shuvaev** (CSHL, 2014–2021) — [KoulakovLab/Registration](https://github.com/KoulakovLab/Registration), licensed under GPL v3.0.

Reference paper:
> Shuvaev S.A. et al. *Spatiotemporal 3D image registration for mesoscale studies of brain development.* Scientific Reports 12, 3678 (2022). https://www.nature.com/articles/s41598-022-06871-8

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
