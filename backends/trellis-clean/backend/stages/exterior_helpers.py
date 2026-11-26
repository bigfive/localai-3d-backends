"""Exterior shell selection and source eligibility, independent of inference."""
import numpy as np
import trimesh
from scipy.ndimage import binary_propagation, binary_dilation

def contains(shell,point):
    if np.any(point<shell.bounds[0]) or np.any(point>shell.bounds[1]):return False
    total=0.
    for first in range(0,len(shell.faces),8192):
        vectors=shell.vertices[shell.faces[first:first+8192]]-point
        a,b,c=vectors[:,0],vectors[:,1],vectors[:,2]
        la,lb,lc=np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.linalg.norm(c,axis=1)
        numerator=np.einsum('ij,ij->i',a,np.cross(b,c))
        denominator=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
        total+=float((2*np.arctan2(numerator,denominator)).sum())
    return abs(total)>2*np.pi

def select_outer(mesh):
    parts=mesh.split(only_watertight=False,repair=False)
    if not parts or any(not p.is_watertight or not p.is_winding_consistent for p in parts):raise ValueError('Exterior selection requires closed consistently oriented voxel shells')
    positive=[p for p in parts if p.volume>0]
    outer=[p for p in positive if not any(q is not p and q.volume>p.volume and contains(q,p.vertices[0]) for q in positive)]
    if not outer:raise ValueError('No exterior shells found')
    return trimesh.util.concatenate(outer),{'input_components':len(parts),'exterior_components':len(outer),'removed_components':len(parts)-len(outer)}

def eligible_faces(vertices,faces,pitch):
    if not np.isfinite(pitch) or pitch<=0:raise ValueError('Invalid exterior voxel pitch')
    shape=np.ceil(np.ptp(vertices,axis=0)/pitch).astype(int)+5
    if np.prod(shape,dtype=np.int64)>256_000_000:raise ValueError('Exterior grid exceeds 256 million cells; reduce voxel_resolution')
    grid=trimesh.Trimesh(vertices,faces,process=False).voxelized(pitch,method='subdivide')
    surface=np.pad(grid.matrix,2);seed=np.zeros_like(surface)
    for a in range(3):
        for edge in [0,-1]:
            sl=[slice(None)]*3;sl[a]=edge;seed[tuple(sl)]=True
    air=binary_propagation(seed,mask=~surface)
    accessible=binary_dilation(air,iterations=2)
    keep=np.zeros(len(faces),dtype=bool)
    for weights in [[1,0,0],[0,1,0],[0,0,1],[1/3,1/3,1/3],[.5,.5,0],[0,.5,.5],[.5,0,.5]]:
        for start in range(0,len(faces),32768):
            f=faces[start:start+32768];points=(vertices[f]*np.asarray(weights)[None,:,None]).sum(axis=1)
            ids=np.rint((points-grid.transform[:3,3])/pitch).astype(int)+2
            ids=np.clip(ids,0,np.asarray(surface.shape)-1)
            keep[start:start+len(f)]|=accessible[tuple(ids.T)]
    if not keep.any():raise ValueError('Exterior filtering removed all source triangles')
    return keep,{'source_triangles':len(faces),'eligible_source_triangles':int(keep.sum()),'excluded_source_triangles':int((~keep).sum()),'enclosed_air_voxels':int((~air & ~surface).sum()),'grid_shape':list(surface.shape)}
