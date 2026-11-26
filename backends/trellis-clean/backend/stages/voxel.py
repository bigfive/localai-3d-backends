from pathlib import Path
import time,json,numpy as np,trimesh,gpytoolbox as g
import meshlib.mrmeshpy as mr,meshlib.mrmeshnumpy as mn
settings=__import__('json').loads((Path(__file__).parent/'settings.json').read_text())
out=Path(__file__).parent/'outputs';t=time.monotonic()
m=trimesh.load(out/'dense-geometry.glb',force='mesh',process=False)
mesh=mn.meshFromFacesVerts(np.asarray(m.faces,dtype=np.int32),np.asarray(m.vertices,dtype=np.float32));pitch=float(np.ptp(m.vertices,axis=0).max()/384);del m
print('Loaded MeshLib mesh; voxel size',pitch,flush=True)
p=mr.OffsetParameters();p.voxelSize=pitch;p.signDetectionMode=mr.SignDetectionMode.Unsigned
print('Starting unsigned-distance OpenVDB remesh, 384 cells across longest axis',flush=True)
t1=time.monotonic();result=mr.offsetMesh(mesh,pitch*settings["shell_thickness_voxels"],p);remesh_seconds=time.monotonic()-t1
v=mn.getNumpyVerts(result);f=mn.getNumpyFaces(result.topology);m=trimesh.Trimesh(v,f,process=False);m.remove_unreferenced_vertices();v=np.asarray(m.vertices);f=np.asarray(m.faces,dtype=np.int32)
m.export(out/'unsigned-384.glb');report={'voxel_size':pitch,'resolution':384,'remesh_seconds':remesh_seconds,'total_before_qslim_seconds':time.monotonic()-t,'vertices':len(v),'triangles':len(f),'watertight':bool(m.is_watertight),'results':{}};print(report,flush=True);(out/'unsigned-report.json').write_text(json.dumps(report,indent=2))
if not m.is_watertight:raise RuntimeError('Voxel output not watertight')
