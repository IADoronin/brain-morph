# Copilot instructions for this repository

This file gives focused, actionable guidance to AI coding assistants working in this codebase.

1. Purpose
- Help contributors implement and refactor Python re-implementation of the original MATLAB brain-registration code, focusing on 3D mesh-based warping for brain morphology studies.

2. Big picture (what to edit)
- Python core lives in [src/utils](src/utils): `Volume`-based 3D image utilities, mesh warping and optimization.
- MATLAB legacy code is in [matlab/Scripts](matlab/Scripts); use for reference only, do not convert automatically without checking tests.
- Separate implementations for 2D ([src/utils/mesh_transform_2d.py](src/utils/mesh_transform_2d.py)) and 3D ([src/utils/mesh_transform_3d.py](src/utils/mesh_transform_3d.py)) mesh transforms.

3. Key files to read first
- Project overview and usage: [README.md](README.md)
- Mesh warping implementation: [src/utils/mesh_transform_3d.py](src/utils/mesh_transform_3d.py) (primary), [src/utils/mesh_transform_2d.py](src/utils/mesh_transform_2d.py)
- Volume abstraction and utilities: [src/utils/volume.py](src/utils/volume.py)
- Optimization placeholder: [src/utils/simulated_annealing.py](src/utils/simulated_annealing.py)
- Unit tests: [tests/test_mesh_transform.py](tests/test_mesh_transform.py) (note: currently imports non-existent mesh_transform.py; update to use mesh_transform_3d)

4. Project-specific conventions (concrete examples)
- `Volume` is a torch-backed object that preserves affine metadata and should be used for I/O, visualization, and high-level ops (see `Volume.from_file`, `visualize`, `resample` in [src/utils/volume.py](src/utils/volume.py)).
- Mesh formats accepted by mesh_transform functions: either `(3, nx, ny, nz)` or `(nx, ny, nz, 3)`. Coordinates may be normalized ([-1,1]) or voxel (`[0..size-1]`) — code auto-detects and normalizes (see `_normalize_mesh` in mesh_transform_3d.py).
- Interpolation `precision` values used across code: `'exact'` (trilinear), `'coarse'` (nearest upsampling speedup), and `'nearest'`. Keep these strings consistent when adding callers.
- Tests import modules by path to avoid package issues — run tests from repo root to match that behavior (see [tests/test_mesh_transform.py](tests/test_mesh_transform.py)); update imports to reflect current file structure.

5. Common developer workflows / commands
- Install minimal deps (only torch/numpy required by core):
```bash
pip install torch numpy pytest
```
- Run tests (from repo root):
```bash
pytest tests/ -v
```
- Run quick single-file test without installing package:
```bash
python tests/test_mesh_transform.py
```

6. Integration and runtime notes
- Code is PyTorch-first and supports GPU tensors; ensure tests skip if `torch` is unavailable (tests already do this).
- Mesh transforms use `torch.nn.functional.grid_sample` and expect sampling grids in normalized coords ([-1,1]). When adding features, preserve device/dtype propagation.
- MATLAB scripts are kept for reference (`matlab/Scripts`). Do not assume 1:1 semantics; prefer Python `Volume` APIs.
- Both 2D and 3D implementations exist; prefer 3D for brain registration tasks.

7. How to contribute changes safely
- Add unit tests under `tests/` that reproduce the intended behavior (see existing mesh tests for examples).
- Preserve shapes and dtypes: functions often accept (D,H,W), (C,D,H,W) or (N,C,D,H,W) — return the same logical shape.
- When changing mesh behavior, update `create_regular_mesh` and related tests accordingly.
- Update test imports to match current file structure (e.g., import mesh_transform_3d instead of mesh_transform).

8. Checklist for reviewers / AI agents
- Does the change preserve CPU/GPU device semantics and dtype?
- Are mesh coordinate conventions (normalized vs voxel) documented and tested?
- Are new functions reachable from `Volume` convenience helpers or clearly documented if low-level?
- Do tests import the correct modules given the current file structure?

If anything here is unclear or you need extra examples (I can add short code snippets referencing specific lines), tell me which area to expand.
