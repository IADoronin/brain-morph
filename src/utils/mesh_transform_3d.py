#%%
import torch
import torch.nn.functional as F
import typing
import matplotlib.pyplot as plt
import mask_3d

def get_basis_3d(points: torch.Tensor):
    """
    Compute 3D affine + bilinear polynomial basis for point coordinates.
    
    Used to build the basis matrix for solving affine transformations with
    bilinear warping terms. The basis consists of: 1,x,y,z,xy,yz,zx,xyz
    
    Args:
        points: Tensor of shape (N, 3) containing N 3D points [x, y, z].
        
    Returns:
        Tensor of shape (N, 8) where each row contains [1,x,y,z,xy,yz,zx,xyz] basis
        expansion for the corresponding input point.
        
    Example:
        >>> points = torch.tensor([[0., 0., 0.], [1., 1., 1.]])
        >>> basis = get_basis_3d(points)
        >>> basis.shape
        torch.Size([3, 8])
    """
    return torch.cat([torch.ones([points.shape[0],1]),
                      points,
                      (points[:,0]*points[:,1]).unsqueeze(-1),
                      (points[:,1]*points[:,2]).unsqueeze(-1),
                      (points[:,2]*points[:,0]).unsqueeze(-1),
                      (points[:,0]*points[:,1]*points[:,2]).unsqueeze(-1)], 
                      dim=1)

#%%


def get_bilinear_transform_3d(
                        grid_init_cell: torch.Tensor,
                        grid_target_cell: torch.Tensor) -> torch.Tensor:
    """
    Compute bilinear transformation matrix for a 3D grid cell.
    
    Solves for the transformation coefficients that map initial grid corners
    to target grid corners using a polynomial basis system. The transformation
    is applied by: target = basis @ transform_mat.
    
    Args:
        grid_init_cell: Initial grid cell corners, shape (1, 2, 2, 2, 3) or 
            (8, 3). Represents 8 control points of a 2×2 grid cell.
        grid_target_cell: Target grid cell corners, same shape as grid_init_cell.
            
    Returns:
        Transform matrix of shape (8, 3) containing coefficients for the
        affine + bilinear transformation.
        
    Raises:
        LinAlgError: If the basis matrix is singular and cannot be solved.
    """
    #gets transform matrix
    grid_init_f = grid_init_cell.flatten(0, -2)
    grid_target_f = grid_target_cell.flatten(0, -2)
    s_base = get_basis_3d(grid_init_f)
    transform_mat = torch.linalg.solve(s_base,grid_target_f)
    return transform_mat


#%%

def iter_grid_cells_3d(grid: torch.Tensor):
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
    ny, nx, nz = grid.shape[0] - 1, grid.shape[1] - 1,grid.shape[2] - 1
    for i in range(ny):
        for j in range(nx):
            for k in range(nz):
            # Extract 8 corners of cell [i:i+2, j:j+2, k:k+2]
                corners = grid[i:i+2, j:j+2, k:k+2]
                yield corners
                
# #%%
# def mask_by_grid_cell_3d(mesh: torch.Tensor, cell: torch.Tensor):
#     """
#     Create a mask for points that lie within a 3D grid cell defined by its corners.
    
#     Uses the bilinear interapolated surfaces defined by the cell corners to determine if points are inside the cell."""
#     faces_vertexes = [
#         cell[0,:,:,:].flatten(0,-2),
#         cell[1,:,:,:].flatten(0,-2),
#         cell[:,0,:,:].flatten(0,-2),
#         cell[:,1,:,:].flatten(0,-2),
#         cell[:,:,0,:].flatten(0,-2),
#         cell[:,:,1,:].flatten(0,-2)
#     ]
#     mask = torch.zeros_like(mesh[...,0],dtype=torch.bool)
#     for face in faces_vertexes:
#         #create surface
#%%
def mesh_transform_3d(image, grid_init, grid_target, method="bilinear", mask_init = None):
    """
    Warp a 3D image using bilinear mesh deformation.
    
    Applies a smooth, cell-wise bilinear transformation to an image
    using moving control point grids. Each cell in the grid defines a local
    transformation that is applied to the image region it covers.
    
    Args:
        image: Input image tensor of shape (D, H, W) or (C, D, H, W) where C is the
            number of channels. Will be converted to (C, D, H, W) internally.
        grid_init: Initial control point grid of shape (ny, nx, nz, 3) with
            normalized ([-1, 1]) or voxel [0..D-1] / [0..H-1] / [0..W-1] coordinates.
            Auto-detected and normalized internally.
        grid_target: Target control point grid of the same shape as grid_init.
        method: Interpolation method. Options: 'bilinear' (default), 'nearest'.
        mask_init: Optional mask with shape (D, H, W) with index representing which cell each voxel belongs to. 
        If provided, only voxels with valid cell indices will be transformed.
    Returns:
        Warped image tensor of the same shape as input image (C, D, H, W).
        
    Raises:
        ValueError: If grid shapes do not match or if grid dimension is not 3  .
        
    Note:
        - Gets non-normalized coordinates; 
        - Device and dtype of output match the input image.
    """


    #input dara check:
    if grid_init.shape != grid_target.shape:
        raise ValueError("grid_init and grid_target must have the same shape")
    if grid_init.shape[-1] != 3:
        raise ValueError("grid_init and grid_target must have 3D control points at last dimension")
    if image.dim() ==3:
        image = image.unsqueeze(0)
    
    #transform 3d image using grids
    mesh = torch.stack(
        torch.meshgrid(
            *[torch.linspace(-1,1,steps=s,device=image.device,dtype=image.dtype) for s in image.shape[1:]],
            indexing="ij"
        ),
        dim=-1
    )
    #shape (D,H,W,3)

    mesh_f = mesh.flatten(0,-2)
    #shape (H*W*D,3)
    mesh_transformed_f = torch.ones_like(mesh_f)*(-1)


    # Iterate over cells in grid
    for cell_init,cell_target in zip(
            iter_grid_cells_3d(grid_init),
            iter_grid_cells_3d(grid_target)
        ):
        transform_mat = get_bilinear_transform_3d(cell_init, cell_target)
        mask_f = mask_3d.innerpoints(cell_init,mesh_f)
        # Update mesh_transformed with transformed points for current cell
        mesh_transformed_f[mask_f] = (get_basis_3d(mesh_f[mask_f])@transform_mat).squeeze()
        
    mesh_transformed_2 = mesh_transformed_f.reshape(mesh.shape).flip([-1])
    # As grid_sample expects normalized coordinates in [-1, 1], we should normalize mesh_transformed 
    # accordingly if it's in pixel coordinates.
    print(mesh_transformed_2[0,0,0,:])
    return F.grid_sample(image.unsqueeze(0), 
                         mesh_transformed_2.unsqueeze(0), 
                         mode="bilinear", align_corners=True,padding_mode="zeros").squeeze(0)

        
