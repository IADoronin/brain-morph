def compute_tension_3d(grid_part: torch.Tensor, grid_full: torch.Tensor = None) -> torch.Tensor:
    #Не проверена
    """Compute deformation energy (tension) of a 3D mesh deformation.

    This mirrors the MATLAB function `mregularize`.

    The function measures how much local tetrahedral volumes change between
    two grids:
      - `grid_part` is the deformed/transformed mesh.
      - `grid_full` is the reference mesh used for normalization.
        If None, `grid_part` is used for scaling (no scaling effect).

    The grid is expected to be of shape (ny, nx, nz, 3) with (x,y,z)
    coordinates in the last dimension.

    Returns:
        A scalar tensor representing the total tension.
    """

    if grid_full is None:
        grid_full = grid_part

    # Ensure float compute and consistent device
    grid_part = grid_part.to(dtype=torch.float32)
    grid_full = grid_full.to(dtype=torch.float32)

    mins = grid_full.amin(dim=(0, 1, 2), keepdim=True)
    maxs = grid_full.amax(dim=(0, 1, 2), keepdim=True)
    spans = (maxs - mins).clamp_min(1e-12)

    part = (grid_part - mins) / spans
    base = (grid_full - mins) / spans

    ny, nx, nz, _ = part.shape

    def _tet_det(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
        mat = torch.stack([b - a, c - a, d - a], dim=1)  # (3,3)
        return torch.det(mat)

    tension = torch.tensor(0.0, device=part.device, dtype=part.dtype)

    for i in range(ny - 1):
        for j in range(nx - 1):
            for k in range(nz - 1):
                N111 = part[i, j, k]
                N112 = part[i, j, k + 1]
                N121 = part[i, j + 1, k]
                N122 = part[i, j + 1, k + 1]
                N211 = part[i + 1, j, k]
                N212 = part[i + 1, j, k + 1]
                N221 = part[i + 1, j + 1, k]
                N222 = part[i + 1, j + 1, k + 1]

                O111 = base[i, j, k]
                O112 = base[i, j, k + 1]
                O121 = base[i, j + 1, k]
                O122 = base[i, j + 1, k + 1]
                O211 = base[i + 1, j, k]
                O212 = base[i + 1, j, k + 1]
                O221 = base[i + 1, j + 1, k]
                O222 = base[i + 1, j + 1, k + 1]

                for (n_a, n_b, n_c, n_d, o_a, o_b, o_c, o_d) in [
                    (N112, N111, N121, N211, O112, O111, O121, O211),
                    (N112, N122, N121, N222, O112, O122, O121, O222),
                    (N121, N221, N222, N211, O121, O221, O222, O211),
                    (N112, N212, N222, N211, O112, O212, O222, O211),
                    (N112, N211, N222, N121, O112, O211, O222, O121),
                ]:
                    vol_n = torch.abs(_tet_det(n_a, n_b, n_c, n_d))
                    vol_o = torch.abs(_tet_det(o_a, o_b, o_c, o_d))
                    tension = tension + torch.abs(vol_n - vol_o)

    return tension / 6.0




