#%%
import torch
import torch.nn.functional as F
import typing
import matplotlib.pyplot as plt

def get_basis_2d(points: torch.Tensor) -> torch.Tensor:
    """
    Compute 2D affine + bilinear polynomial basis for point coordinates.
    
    Used to build the basis matrix for solving affine transformations with
    bilinear warping terms. The basis consists of: constant term, linear terms
    (x, y), and bilinear cross-product term (x*y).
    
    Args:
        points: Tensor of shape (N, 2) containing N 2D points [x, y].
        
    Returns:
        Tensor of shape (N, 4) where each row contains [1, x, y, x*y] basis
        expansion for the corresponding input point.
        
    Example:
        >>> points = torch.tensor([[0., 0.], [1., 1.]])
        >>> basis = get_basis_2d(points)
        >>> basis.shape
        torch.Size([2, 4])
    """
    return torch.cat([torch.ones([points.shape[0],1]),
                      points,
                      (points[:,0]*points[:,1]).unsqueeze(-1)], 
                      dim=1)

def get_bilinear_transform_2d(
                        grid_init_cell: torch.Tensor,
                        grid_target_cell: torch.Tensor) -> torch.Tensor:
    """
    Compute bilinear transformation matrix for a 2D grid cell.
    
    Solves for the transformation coefficients that map initial grid corners
    to target grid corners using a polynomial basis system. The transformation
    is applied by: target = basis @ transform_mat.
    
    Args:
        grid_init_cell: Initial grid cell corners, shape (1, 2, 2, 2) or 
            (4, 2). Represents 4 control points of a 2×2 grid cell.
        grid_target_cell: Target grid cell corners, same shape as grid_init_cell.
            
    Returns:
        Transform matrix of shape (4, 2) containing coefficients for the
        affine + bilinear transformation.
        
    Raises:
        LinAlgError: If the basis matrix is singular and cannot be solved.
    """
    #gets transform matrix
    grid_init_f = grid_init_cell.flatten(0, -2)
    grid_target_f = grid_target_cell.flatten(0, -2)
    s_base = get_basis_2d(grid_init_f)
    transform_mat = torch.linalg.solve(s_base,grid_target_f)
    return transform_mat

def iter_grid_cells_2d(grid: torch.Tensor):
    """
    Iterate over 2D grid cells as adjacent 2×2 corner blocks.
    
    Yields corner point coordinates for each cell in a regular 2D control point
    grid. Useful for processing mesh deformation cell-by-cell.
    
    Args:
        grid: Control point grid of shape (ny, nx, 2) where (ny, nx) defines
            the grid shape and the last dimension contains 2D coordinates.
            
    Yields:
        cell corners
    """
    ny, nx = grid.shape[0] - 1, grid.shape[1] - 1
    for i in range(ny):
        for j in range(nx):
            # Extract 4 corners of cell [i:i+2, j:j+2]
            corners = grid[i:i+2, j:j+2]
            yield corners

def mesh_transform_2d(image, grid_init, grid_target, method="bilinear"):
    """
    Warp a 2D image using bilinear mesh deformation.
    
    Applies a smooth, cell-wise bilinear transformation to an image
    using moving control point grids. Each cell in the grid defines a local
    transformation that is applied to the image region it covers.
    
    Args:
        image: Input image tensor of shape (H, W) or (C, H, W) where C is the
            number of channels. Will be converted to (C, H, W) internally.
        grid_init: Initial control point grid of shape (ny, nx, 2) with
            normalized ([-1, 1]) or voxel [0..H-1] / [0..W-1] coordinates.
            Auto-detected and normalized internally.
        grid_target: Target control point grid of the same shape as grid_init.
        method: Interpolation method. Options: 'bilinear' (default), 'nearest'.
            
    Returns:
        Warped image tensor of the same shape as input image (C, H, W).
        
    Raises:
        ValueError: If grid shapes do not match or if grid dimension is not 2.
        
    Note:
        - Gets non-normalized coordinates; 
        - Device and dtype of output match the input image.
    """
    #image: (H,W) or (C,H,W), where C is number of channels
    #grid_init and grid_target: (ny, nx, 2) control point grids
    #input dara check:
    if grid_init.shape != grid_target.shape:
        raise ValueError("grid_init and grid_target must have the same shape")
    if grid_init.shape[2] != 2:
        raise ValueError("grid_init and grid_target must have 2D control points at last dimension")
    if image.dim() ==2:
        image = image.unsqueeze(0)
    
    #transform 2d image using grids
    mesh = torch.stack(
        torch.meshgrid(
            *[torch.linspace(-1,1,steps=s,device=image.device,dtype=image.dtype) for s in image.shape[1:]],
            indexing="ij"
        ),
        dim=2
    )
    #shape (H,W,2)

    mesh_f = mesh.flatten(0,-2)
    #shape (H*W,2)
    mesh_transformed_f = torch.ones_like(mesh_f)*(-1)
    assigned = torch.zeros(mesh_f.shape[0], dtype=torch.bool, device=mesh_f.device)

    cell_centers: list[torch.Tensor] = []
    cell_transforms: list[torch.Tensor] = []

    # Iterate over cells in grid
    for cell_init,cell_target in zip(
            iter_grid_cells_2d(grid_init),
            iter_grid_cells_2d(grid_target)
        ):
        # Get transform matrix for current cell
        transform_mat = get_bilinear_transform_2d(cell_init, cell_target)
        order = (0,0),(0,1),(1,1),(1,0),(0,0)
        directed_edges = torch.stack([torch.tensor(cell_init[order[i+1]]) -
                      torch.tensor(cell_init[order[i]])
                      for i in range(len(order)-1)],dim=-1)
        p_vecs = torch.stack([mesh - cell_init[order[i]] for i in range(len(order)-1)],dim=-1)
        cross_prods = directed_edges[0,:]*p_vecs[...,1,:] - \
            directed_edges[1,:]*p_vecs[...,0,:]
        mask = torch.all(cross_prods<=0,dim=-1)
        mask_f = mask.flatten()
        mesh_transformed_f[mask_f] = (get_basis_2d(mesh_f[mask_f])@transform_mat).squeeze()
        assigned[mask_f] = True
        cell_centers.append(cell_init.flatten(0, -2).mean(0))
        cell_transforms.append(transform_mat)

    # Fallback: assign pixels on cell boundaries missed by the cross-product test
    # due to floating-point precision — map each to the nearest cell centroid.
    unassigned = ~assigned
    if unassigned.any():
        centers = torch.stack(cell_centers)  # (n_cells, 2)
        nearest = torch.cdist(mesh_f[unassigned], centers).argmin(dim=1)
        for cell_idx, transform_mat in enumerate(cell_transforms):
            sel = (nearest == cell_idx)
            if sel.any():
                pts = unassigned.nonzero(as_tuple=True)[0][sel]
                mesh_transformed_f[pts] = (get_basis_2d(mesh_f[pts]) @ transform_mat).squeeze()
    mesh_transformed_2 = mesh_transformed_f.reshape(mesh.shape).flip(dims=[2])
    # As grid_sample expects normalized coordinates in [-1, 1], we should normalize mesh_transformed 
    # accordingly if it's in pixel coordinates.
    print(mesh_transformed_2[0,0,:])
    return F.grid_sample(image.unsqueeze(0), 
                         mesh_transformed_2.unsqueeze(0), 
                         mode="bilinear", align_corners=True,padding_mode="zeros").squeeze(0)

        
