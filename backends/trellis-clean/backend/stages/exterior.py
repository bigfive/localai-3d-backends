from pathlib import Path
import json,time,struct,numpy as np,trimesh
from exterior_helpers import select_outer,eligible_faces
run=Path(__file__).parent;out=run/'outputs';settings=json.loads((run/'settings.json').read_text());started=time.monotonic()
if settings['exterior_only'] is not True:raise ValueError('Exterior stage requires exterior_only=true')
mesh=trimesh.load(out/'unsigned-384.glb',force='mesh',process=False)
outer,report=select_outer(mesh)
outer.export(out/'unsigned-exterior.glb')
# Keep original source indexing and colours; filter only the faces used by the
# colour projector. No generated image, original GLB or old run is rewritten.
raw=(out/'trellis-1024.glb').read_bytes();length=struct.unpack_from('<I',raw,12)[0];doc=json.loads(raw[20:20+length]);blob=raw[28+length:]
if len(doc['meshes'])!=1 or len(doc['meshes'][0]['primitives'])!=1:raise ValueError('Exterior source requires the native single-primitive TRELLIS GLB')
p=doc['meshes'][0]['primitives'][0]
def accessor(i):
    a=doc['accessors'][i];v=doc['bufferViews'][a['bufferView']]
    if a.get('sparse') or v.get('byteStride') or v.get('buffer',0)!=0:raise ValueError('Unsupported exterior source accessor layout')
    dtype={5126:'<f4',5123:'<u2',5125:'<u4'}[a['componentType']];size={'VEC3':3,'SCALAR':1}[a['type']]
    return np.frombuffer(blob,dtype=dtype,count=a['count']*size,offset=v.get('byteOffset',0)+a.get('byteOffset',0)).reshape(-1,size)
vertices=accessor(p['attributes']['POSITION']);faces=accessor(p['indices']).reshape(-1,3).astype(np.int64)
pitch=float(np.ptp(vertices,axis=0).max())/settings['voxel_resolution']
keep,source_report=eligible_faces(vertices,faces,pitch);np.save(out/'exterior-source-face-mask.npy',keep,allow_pickle=False)
report.update(source_report);report.update(triangles=len(outer.faces),watertight=bool(outer.is_watertight),seconds=time.monotonic()-started)
(out/'exterior-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
