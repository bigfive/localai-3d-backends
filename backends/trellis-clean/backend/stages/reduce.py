from pathlib import Path
import json,time,numpy as np,trimesh,gpytoolbox as g
out=Path(__file__).parent/'outputs';m=trimesh.load(out/'unsigned-main.glb',force='mesh',process=False);report={'input_triangles':len(m.faces),'component_policy':'largest connected component; full voxel result retained','results':{}}
for n in [10000]:
 t=time.monotonic();v,f,*_=g.decimate(np.asarray(m.vertices),np.asarray(m.faces,dtype=np.int32),num_faces=n,method='qslim')
 if not len(f):raise RuntimeError('Empty QSlim output')
 q=trimesh.Trimesh(v,f,process=False);q.export(out/f'unsigned-main-qslim-{n}.glb',include_normals=True);report['results'][str(n)]={'triangles':len(f),'watertight':bool(q.is_watertight),'seconds':time.monotonic()-t};print(n,report['results'][str(n)],flush=True);(out/'unsigned-main-report.json').write_text(json.dumps(report,indent=2))
