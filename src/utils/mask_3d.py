#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def get_faces(cell):
    """
    Create deformed parallelogram faces by it's vertexes

    Args:
        cell: Tensor of shape (2,2,2,3)  of 3D points 

    Returns:
        Tesnor of shape(6,4,3) (Number of faces, points in faces, cooridinates of points)
    
    Example:
        >>> a = a=torch.stack(torch.meshgrid(*[torch.linspace(-1,1,2) for _ in range(3)]),dim=-1)
        >>> faces = get_faces(a)
    """
    A,B,C,D,A1,B1,C1,D1 = [0, 0, 0],[0, 1, 0],[1, 1, 0],[1, 0, 0],[0, 0, 1],[0, 1, 1],[1, 1, 1],[1, 0, 1]
    order = [[A,B,C,D],[A,D,D1,A1],[A,A1,B1,B],[D1,C1,B1,A1],[B,B1,C1,C],[D,C,C1,D1]]
    faces = []
    for i in order:
        face=[]
        for j in i:
            face.append(cell[*j,:])
        faces.append(torch.stack(face,dim=0))
    return torch.stack(faces,0)

def triangulate_faces(faces):
    central_vertex = faces.mean(dim=1, keepdim=True)
    triangles = [torch.stack([faces[:, i], faces[:, (i + 1) % 4], central_vertex.squeeze(1)], dim=1) for i in range(4)]
    # Reorder so triangles are grouped by face (face0 triangles, face1 triangles, ...)
    triangles = torch.stack(triangles, dim=0)  # (4, num_faces, 3, 3)
    triangles = triangles.permute(1, 0, 2, 3).flatten(0, 1)  # (num_faces*4, 3, 3)
    return triangles

def get_normals(triangles):
    v1 = triangles[:,1] - triangles[:,0]
    v2 = triangles[:,2] - triangles[:,0]
    normals = torch.cross(v1, v2, dim=1)
    return normals

def innerpoint(cell,point):
    faces = get_faces(cell)
    triangles = triangulate_faces(faces)
    d = triangles.shape[0]
    print(d//2)
    scalars = (get_normals(triangles)*(point-triangles[:,2,:].squeeze(1))).sum(dim=1)
    return (scalars[:d//2]<=0).all() and (scalars[d//2:]<=0).all()

def innerpoints(cell, points):
    faces = get_faces(cell)
    triangles = triangulate_faces(faces)
    normals = get_normals(triangles)
    # points: (N, 3), triangles[:,2,:]: (num_triangles, 3)
    diff = points.unsqueeze(1) - triangles[:, 2, :].unsqueeze(0)  # (N, num_triangles, 3)
    scalars = (normals.unsqueeze(0) * diff).sum(dim=2)  # (N, num_triangles)

    d = triangles.shape[0]
    half = d // 2
    inside_first = (scalars[:, :half] <= 0).all(dim=1)  # include boundary for first half
    inside_second = (scalars[:, half:] <= 0).all(dim=1)  # exclude boundary for second half
    return inside_first & inside_second  # (N,)


# %%
