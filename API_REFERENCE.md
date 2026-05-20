# Brain Morph — API Reference

Context for building a napari UI frontend. All public classes and functions are listed with their inputs, outputs, and callbacks.

---

## Volume  (`src/utils/volume.py`)

`Volume` is a `torch.Tensor` subclass that carries an optional affine matrix.
Shape convention: **(C, D, H, W)** — channels first.

### `Volume(data, affine=None)`
| | Type | Description |
|---|---|---|
| `data` | array-like \| Tensor | Any numeric data |
| `affine` | ndarray \| Tensor \| None | 4×4 affine matrix; defaults to identity |
| **returns** | `Volume (C, D, H, W)` | |

### `Volume.visualize(channel=None)`
| | Type | Description |
|---|---|---|
| `channel` | int \| None | None → RGB max-projections; int → single-channel grayscale |
| **returns** | None | Displays matplotlib figure with 3 orthogonal max projections |

### `Volume.normalize()`
| | Type |
|---|---|
| **returns** | `Volume` normalised to `[0, 1]` |

### `Volume.resample(new_shape, mode='trilinear')`
| | Type | Description |
|---|---|---|
| `new_shape` | `(D, H, W)` | Target spatial size |
| `mode` | str | `'nearest'`, `'bilinear'`, `'trilinear'` |
| **returns** | `Volume (C, D_new, H_new, W_new)` | |

### `Volume.rotate(theta, phi, center, interpolation='linear')`
| | Type | Description |
|---|---|---|
| `theta` | float | Rotation around X-axis, degrees |
| `phi` | float | Rotation around Y-axis, degrees |
| `center` | `(x, y, z)` | Pivot point in voxel coordinates |
| `interpolation` | str | `'nearest'`, `'linear'`, `'bilinear'`, `'bicubic'` |
| **returns** | Tensor | Rotated volume (loses Volume wrapper) |

### `Volume.save_nii(file_path)`
| | Type | Description |
|---|---|---|
| `file_path` | str | Output path `.nii` or `.nii.gz` |
| **returns** | None | Saves 3-D `(D,H,W)` for C=1, 4-D `(D,H,W,C)` for multi-channel |

### `Volume.load_nii(file_path)` *(classmethod)*
| | Type | Description |
|---|---|---|
| `file_path` | str | NIfTI file path |
| **returns** | `Volume (C, D, H, W)` | `.affine` attribute set from NIfTI header |

---

## MeshTransformer3D  (`src/utils/mesh_transformer_3d.py`)

Pre-computes which voxel belongs to which grid cell (expensive, done once at construction).
Reuse one instance for many `transform()` calls with different grids.

### `MeshTransformer3D(grid_init, image_shape, device=None, dtype=float32)`
| | Type | Description |
|---|---|---|
| `grid_init` | `Tensor (ny, nx, nz, 3)` | Regular reference control-point grid |
| `image_shape` | `(D, H, W)` | Spatial size of images to be transformed |
| `device` | torch.device \| None | CPU by default |
| `dtype` | torch.dtype | `float32` default |
| **returns** | `MeshTransformer3D` | Cell index map precomputed |

### `.transform(image, grid_target, method='bilinear')`
| | Type | Description |
|---|---|---|
| `image` | `Tensor (D,H,W)` or `(C,D,H,W)` | Moving image |
| `grid_target` | `Tensor (ny,nx,nz,3)` | Deformed control-point grid |
| `method` | str | `'bilinear'` or `'nearest'` |
| **returns** | `Tensor (C, D, H, W)` | Warped image (channel dim always present) |

### `.tension(grid_target, grid_ref=None, mode='abs', voxel_size=None, mask=None, metric=None)`
| | Type | Description |
|---|---|---|
| `grid_target` | `Tensor (ny,nx,nz,3)` | Deformed grid to evaluate |
| `grid_ref` | `Tensor \| None` | Reference; `None` → `self.grid_init` |
| `mode` | str | `'abs'` (MATLAB-compatible) or `'squared'` (smooth gradient) |
| `voxel_size` | `(sz_d, sz_h, sz_w)` \| None | Physical voxel size in µm |
| `mask` | `Tensor (D,H,W)` \| None | Binary tissue mask; cells over empty space get weight 0 |
| `metric` | `TensionMetric \| None` | Custom metric; `None` → `VolumeTension(mode)` |
| **returns** | scalar `Tensor` | Deformation energy |

### `.cell_index_map` *(property)*
| | Type | Description |
|---|---|---|
| **returns** | `Tensor (D*H*W,)` long | Cell index per voxel |

### `mesh_transform_3d(image, grid_init, grid_target, method='bilinear')` *(standalone convenience)*
Same as constructing `MeshTransformer3D` and calling `.transform()` once.

---

## compute_tension_3d  (`src/utils/compute_tension_3d.py`)

### `compute_tension_3d(grid_part, grid_full=None, mode='abs', voxel_size=None, cell_weights=None, metric=None)`
| | Type | Description |
|---|---|---|
| `grid_part` | `Tensor (ny,nx,nz,3)` | Deformed (or sub-region) grid |
| `grid_full` | `Tensor \| None` | Reference grid for per-axis normalisation; `None` → uses `grid_part` |
| `mode` | str | `'abs'` or `'squared'` |
| `voxel_size` | `(sz_d, sz_h, sz_w)` \| None | µm per voxel for isotropic weighting |
| `cell_weights` | `Tensor (ny-1,nx-1,nz-1)` \| None | Per-cell weights ∈ `[0,1]` |
| `metric` | `TensionMetric \| None` | `None` → `VolumeTension(mode)` |
| **returns** | scalar `Tensor` | |

---

## Tension metrics  (`src/utils/tension_metrics.py`)

### `TensionMetric` *(Protocol / interface)*
Any callable with signature:
```python
def __call__(
    grid_target: Tensor,          # (ny, nx, nz, 3)  normalised
    grid_ref:    Tensor,          # (ny, nx, nz, 3)  normalised
    cell_weights: Tensor | None,  # (ny-1, nx-1, nz-1)
) -> Tensor: ...                  # scalar
```

### `VolumeTension(mode='abs')`
Det-based, MATLAB `mregularize`-compatible. 5-tetrahedra hexahedral decomposition.
- `mode='abs'`: matches MATLAB exactly
- `mode='squared'`: smooth gradient for autograd

### `BendingTension(mode='abs')`
Second-derivative bending energy of the displacement field. ~27 ops/node (vs ~135 for VolumeTension).
Affine transforms (translation, rotation, uniform scaling) contribute exactly 0.
`cell_weights` accepted for API compatibility but not applied (operates on nodes, not cells).

---

## registration_cost  (`src/registration/cost.py`)

### `registration_cost(image_moving, image_fixed, transformer, grid_target, lam, mask=None, metric=None, similarity='corr', tension_mode='abs', channel_weights=None)`
| | Type | Description |
|---|---|---|
| `image_moving` | `Tensor (D,H,W)` or `(C,D,H,W)` | |
| `image_fixed` | `Tensor` | Same shape as `image_moving` |
| `transformer` | `MeshTransformer3D` | Pre-built transformer |
| `grid_target` | `Tensor (ny,nx,nz,3)` | Candidate grid |
| `lam` | float | Regularisation weight λ |
| `mask` | `Tensor (D,H,W)` \| None | Binary tissue mask |
| `metric` | `TensionMetric` \| None | `None` → `VolumeTension(tension_mode)` |
| `similarity` | str | `'corr'` (Pearson), `'ncc'`, `'mse'` (negative MSE) |
| `tension_mode` | str | `'abs'` or `'squared'` |
| `channel_weights` | `Tensor (C,)` \| None | Per-channel weights; `None` → uniform |
| **returns** | scalar `Tensor` | **Higher = better** (`similarity − λ·tension`) |

### `normalise_channel_weights(weights, n_channels, device)`
| | Type | Description |
|---|---|---|
| `weights` | `Tensor \| None` | Raw weights or `None` |
| `n_channels` | int | |
| `device` | torch.device | |
| **returns** | `Tensor (C,)` | Normalised to sum=1; uniform 1/C when `None` |

---

## Optimizers  (`src/registration/optimizers.py`)

All optimizers implement the same interface:

```python
optimizer.optimize(
    image_moving:    Tensor,          # (D,H,W) or (C,D,H,W)
    image_fixed:     Tensor,          # same shape
    transformer:     MeshTransformer3D,
    grid_start:      Tensor,          # (ny,nx,nz,3) initial grid
    n_steps:         int,
    lam:             float = 1e-3,
    mask:            Tensor | None,   # (D,H,W)
    metric:          TensionMetric | None,
    channel_weights: Tensor | None,   # (C,)
) -> Tensor (ny, nx, nz, 3)          # optimised grid
```

---

### `SAOptimizer` — Simulated Annealing

Works with **any** metric, including non-differentiable ones.

```python
SAOptimizer(
    temp_start    = 1e-3,
    temp_end      = 1e-3 / 30,   # geometric cooling to this temperature
    coeff_start   = 0.2,          # initial step size as fraction of mean cell size
    coeff_drop    = 0.9966,       # on reject: coeff *= coeff_drop; on accept: /= coeff_drop
    attention_freq = 100,         # rebuild attention map every N steps; 0 = uniform sampling
    similarity    = 'corr',
    callback      = None,         # called every callback_freq steps
    callback_freq = 50,
)
```

**`callback` signature:**
```python
def callback(step: int, cost: float, warped: Tensor) -> None:
    # step:   current iteration (0-indexed)
    # cost:   current registration cost (scalar float, higher = better)
    # warped: warped moving image — shape (D,H,W) for single-channel,
    #         (C,D,H,W) for multi-channel
    ...
```

Callback is called every `callback_freq` steps **after** accept/reject decision.

**`.optimize(...)` returns:** best grid found during entire SA run (not the final step).

---

### `GradientOptimizer` — Adam / SGD / LBFGS

Requires differentiable metric (`tension_mode='squared'`).

```python
GradientOptimizer(
    optimizer_cls    = torch.optim.Adam,
    lr               = 1e-3,
    optimizer_kwargs = None,       # extra kwargs for the torch optimizer
    similarity       = 'corr',
    tension_mode     = 'squared',  # must be 'squared' for autograd
)
```

No callback. **`.optimize(...)` returns:** last grid (no best-tracking).

---

### `HybridOptimizer` — SA exploration → Gradient refinement

```python
HybridOptimizer(
    sa_optimizer = SAOptimizer(),         # configured SA instance
    gd_optimizer = GradientOptimizer(),   # configured GD instance
    sa_fraction  = 0.7,   # fraction of n_steps given to SA; remainder to GD
)
```

Callbacks work via the `sa_optimizer` instance.

---

## Pipeline  (`src/registration/pipeline.py`)

### `Stage` *(dataclass)*
```python
@dataclass
class Stage:
    grid_shape:   tuple[int, int, int]  # (ny, nx, nz) control-point grid
    optimizer:    MeshOptimizer          # SA / Gradient / Hybrid instance
    n_steps:      int                    # optimisation steps for this stage
    lam:          float = 1e-3          # regularisation weight λ
    metric:       TensionMetric | None = None
    image_scale:  int = 1               # integer downsample factor
                                        #   1 = full resolution
                                        #   8 = 1/8 size (500× faster per SA step)
```

### `RegistrationPipeline(stages: list[Stage])`

### `.run(image_moving, image_fixed, mask=None, channel_weights=None)`
| | Type | Description |
|---|---|---|
| `image_moving` | `Tensor (D,H,W)` or `(C,D,H,W)` | Moving image |
| `image_fixed` | `Tensor` | Fixed image, same shape |
| `mask` | `Tensor (D,H,W)` \| None | Binary tissue mask forwarded to every stage |
| `channel_weights` | `Tensor (C,)` \| None | Per-channel importance; `None` = uniform |
| **returns** | `Tensor (ny, nx, nz, 3)` | Optimised grid from the **last** stage |

No pipeline-level callback — attach callbacks to individual `SAOptimizer` instances inside stages.

---

## Standalone helpers  (`src/registration/pipeline.py`)

### `interpolate_grid(grid, target_shape)`
| | Type | Description |
|---|---|---|
| `grid` | `Tensor (ny, nx, nz, 3)` | Source control-point grid |
| `target_shape` | `(ny_new, nx_new, nz_new)` | Target node count |
| **returns** | `Tensor (ny_new, nx_new, nz_new, 3)` | Trilinearly interpolated grid |

---

## Typical usage example

```python
import torch
from src.utils.volume import Volume
from src.registration.pipeline import RegistrationPipeline, Stage
from src.registration.optimizers import SAOptimizer, GradientOptimizer

# Load images — shape (C, D, H, W)
im_moving = Volume.load_nii("moving.nii")   # e.g. (3, 80, 96, 112)
im_fixed  = Volume.load_nii("fixed.nii")

# Optional tissue mask — shape (D, H, W)
mask = torch.ones(im_moving.shape[-3:], dtype=torch.bool)

# Channel importance weights
channel_weights = torch.tensor([0.5, 0.3, 0.2])

# Progress callback for the fine stage
cost_log = []
def on_step(step, cost, warped):
    cost_log.append(cost)

pipeline = RegistrationPipeline([
    Stage((3, 3, 3), SAOptimizer(n_steps=2000), n_steps=2000, lam=5e-2, image_scale=8),
    Stage((5, 5, 5), SAOptimizer(n_steps=2000), n_steps=2000, lam=2e-2, image_scale=8),
    Stage((7, 7, 7), SAOptimizer(n_steps=2000), n_steps=2000, lam=1e-2, image_scale=8),
    Stage((9, 9, 9),
          SAOptimizer(callback=on_step, callback_freq=50),
          n_steps=2000, lam=2e-3, image_scale=4),
])

grid_result = pipeline.run(im_moving, im_fixed,
                            mask=mask,
                            channel_weights=channel_weights)
# grid_result: Tensor (9, 9, 9, 3)

# Apply the result to the moving image
from src.utils.mesh_transformer_3d import MeshTransformer3D
t = MeshTransformer3D(grid_result, im_moving.shape[-3:])
# Need the coarse->fine grid interpolated to match MeshTransformer:
# In practice, pipeline returns the last stage grid directly.
```