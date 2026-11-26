from pathlib import Path
import json,struct,io,time
import numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from PIL import Image
settings=__import__('json').loads((Path(__file__).parent/'settings.json').read_text())
out=Path(__file__).parent/'outputs';start=time.monotonic();size=2048

def read(path):
 raw=path.read_bytes();jl=struct.unpack_from('<I',raw,12)[0];return json.loads(raw[20:20+jl]),raw[28+jl:]
def acc(d,b,i):
 a=d['accessors'][i];w=d['bufferViews'][a['bufferView']];dtype={5126:'f4',5123:'u2',5125:'u4',5121:'u1'}[a['componentType']];dim={'VEC3':3,'VEC4':4,'VEC2':2,'SCALAR':1}[a['type']];x=np.frombuffer(b,dtype=dtype,count=a['count']*dim,offset=w.get('byteOffset',0)+a.get('byteOffset',0)).reshape(-1,dim)
 if a.get('normalized'):x=x.astype('f4')/np.iinfo(dtype).max
 return x
sd,sb=read(out/'trellis-1024.glb');attrs=sd['meshes'][0]['primitives'][0]['attributes'];sv=acc(sd,sb,attrs['POSITION']);sc=acc(sd,sb,attrs['COLOR_0'])[:,:3];sn=acc(sd,sb,attrs['NORMAL']);tree=cKDTree(sv)
d,b=read(out/'resolution-2048-normal-10000.glb');pr=d['meshes'][0]['primitives'][0];a=pr['attributes'];v=acc(d,b,a['POSITION']);n=acc(d,b,a['NORMAL']);uv=acc(d,b,a['TEXCOORD_0']);f=acc(d,b,pr['indices']).reshape(-1,3)
tri=np.full((size,size),-1,np.int32);bary=np.zeros((size,size,3),np.float32)
for i,q in enumerate(uv[f]*size-.5):
 lo=np.maximum(np.ceil(q.min(axis=0)).astype(int),0);hi=np.minimum(np.floor(q.max(axis=0)).astype(int),size-1)
 if np.any(hi<lo):continue
 xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1));xy=np.column_stack([xx.ravel(),yy.ravel()]);a=q[0];e=q[1]-a;h=q[2]-a;det=e[0]*h[1]-e[1]*h[0];r=xy-a
 w1=(r[:,0]*h[1]-r[:,1]*h[0])/det;w2=(e[0]*r[:,1]-e[1]*r[:,0])/det;w=np.column_stack([1-w1-w2,w1,w2]);keep=(w>=-1e-7).all(axis=1);x=xy[keep,0];y=xy[keep,1];tri[y,x]=i;bary[y,x]=w[keep]
mask=tri>=0;ids=f[tri[mask]];w=bary[mask];pos=np.sum(v[ids]*w[:,:,None],axis=1);ln=np.sum(n[ids]*w[:,:,None],axis=1);ln/=np.linalg.norm(ln,axis=1,keepdims=True)
# Build a projection BVH from source triangles; MeshLib repairs topology on import.
# Recover source colour per resulting vertex by exact position correspondence.
import meshlib.mrmeshpy as mr,meshlib.mrmeshnumpy as mn
sourcefaces=acc(sd,sb,sd['meshes'][0]['primitives'][0]['indices']).reshape(-1,3)
if settings['exterior_only']:
 face_mask=np.load(out/'exterior-source-face-mask.npy',allow_pickle=False)
 if face_mask.dtype!=np.bool_ or face_mask.shape!=(len(sourcefaces),) or not face_mask.any():raise ValueError('Invalid exterior source-face mask')
 sourcefaces=sourcefaces[face_mask]
reference=mn.meshFromFacesVerts(sourcefaces.astype('i4'),sv.astype('f4'))
rv=mn.getNumpyVerts(reference);rf=mn.getNumpyFaces(reference.topology)
lookupdist,lookup=tree.query(rv)
if np.max(lookupdist)>1e-7:raise RuntimeError('Projection mesh changed source vertex positions')
rc=sc[lookup];projector=mr.PointsToMeshProjector();projector.updateMeshData(reference)
proj=mr.std_vector_MeshProjectionResult();projector.findProjections(proj,mn.fromNumpyArray(pos.astype('f4')),None,None,1e30,0.)
faceids=np.fromiter((int(r.proj.face) for r in proj),dtype=np.int64,count=len(pos))
if np.any(faceids<0):raise RuntimeError('Source projection missed')
closest=np.array([[r.proj.point.x,r.proj.point.y,r.proj.point.z] for r in proj])
validfaces=mn.getNumpyBitSet(reference.topology.getValidFaces());face_lookup=np.full(len(validfaces),-1,dtype=np.int64);face_lookup[np.flatnonzero(validfaces)]=np.arange(len(rf));faceids=face_lookup[faceids]
if np.any(faceids<0):raise RuntimeError('Projection returned invalid source face')
weights=trimesh.triangles.points_to_barycentric(rv[rf[faceids]],closest)
color=np.sum(rc[rf[faceids]]*weights[:,:,None],axis=1)
dist=np.linalg.norm(pos-closest,axis=1)[:,None]
# glTF vertex colours are linear; base colour textures are encoded as sRGB.
color=np.where(color<=.0031308,color*12.92,1.055*np.maximum(color,0)**(1/2.4)-.055)
image=np.zeros((size,size,3),np.uint8);image[mask]=np.clip(np.round(color*255),0,255).astype('u1');distance,nearest=distance_transform_edt(~mask,return_indices=True);pad=(~mask)&(distance<=settings["texture_padding"]);image[pad]=image[nearest[0][pad],nearest[1][pad]]
png=io.BytesIO();Image.fromarray(image).save(png,format='PNG');data=png.getvalue();(out/'triangle-2048-albedo.png').write_bytes(data)
blob=b+b'\0'*((-len(b))%4);offset=len(blob);blob+=data;blob+=b'\0'*((-len(blob))%4);vi=len(d['bufferViews']);d['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':len(data)});imageid=len(d['images']);d['images'].append({'bufferView':vi,'mimeType':'image/png'});tex=len(d['textures']);d['textures'].append({'source':imageid,'sampler':0});mat=d['materials'][0]['pbrMetallicRoughness'];mat['roughnessFactor']=settings['roughness'];mat['metallicFactor']=settings['metallic'];mat['baseColorTexture']={'index':tex};mat['baseColorFactor']=[1,1,1,1];d['buffers'][0]['byteLength']=len(blob);j=json.dumps(d,separators=(',',':')).encode();j+=b' '*((-len(j))%4);(out/'triangle-2048-textured.glb').write_bytes(struct.pack('<III',0x46546c67,2,28+len(j)+len(blob))+struct.pack('<II',len(j),0x4e4f534a)+j+struct.pack('<II',len(blob),0x004e4942)+blob)
print({'seconds':time.monotonic()-start,'texels':len(pos),'p95_vertex_distance':float(np.quantile(dist[:,0],.95))},flush=True)
