from pathlib import Path
import time,json,struct,io,sys
import numpy as np,trimesh,xatlas
from PIL import Image
from scipy.ndimage import distance_transform_edt
import meshlib.mrmeshpy as mr,meshlib.mrmeshnumpy as mn
settings=__import__('json').loads((Path(__file__).parent/'settings.json').read_text())
out=Path(__file__).parent/'outputs';size=2048

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-20)
def write_glb(path,v,n,uv,t,f,img):
 chunks=[];views=[];access=[]
 def data(b):
  offset=sum(len(x) for x in chunks);padded=b+b'\0'*((-len(b))%4);chunks.append(padded);views.append({'buffer':0,'byteOffset':offset,'byteLength':len(b)});return len(views)-1
 def attr(a,kind,component):
  a=np.ascontiguousarray(a);vi=data(a.tobytes());item={'bufferView':vi,'componentType':component,'count':len(a),'type':kind}
  if kind=='VEC3':item.update(min=a.min(axis=0).tolist(),max=a.max(axis=0).tolist())
  access.append(item);return len(access)-1
 attrs={'POSITION':attr(v.astype('f4'),'VEC3',5126),'NORMAL':attr(n.astype('f4'),'VEC3',5126),'TEXCOORD_0':attr(uv.astype('f4'),'VEC2',5126),'TANGENT':attr(t.astype('f4'),'VEC4',5126)}
 ind=attr(f.astype('u4').ravel(),'SCALAR',5125);imview=data(img)
 doc={'asset':{'version':'2.0'},'buffers':[{'byteLength':sum(len(x) for x in chunks)}],'bufferViews':views,'accessors':access,'images':[{'bufferView':imview,'mimeType':'image/png'}],'samplers':[{'magFilter':9729,'minFilter':9987,'wrapS':33071,'wrapT':33071}],'textures':[{'source':0,'sampler':0}],'materials':[{'normalTexture':{'index':0,'scale':1},'pbrMetallicRoughness':{'baseColorFactor':[.65,.68,.73,1],'metallicFactor':0,'roughnessFactor':1}}],'meshes':[{'primitives':[{'attributes':attrs,'indices':ind,'material':0}]}],'nodes':[{'mesh':0}],'scenes':[{'nodes':[0]}],'scene':0}
 if not settings['normal_map']:doc['materials'][0].pop('normalTexture')
 j=json.dumps(doc,separators=(',',':')).encode();j+=b' '*((-len(j))%4);b=b''.join(chunks);path.write_bytes(struct.pack('<III',0x46546c67,2,12+8+len(j)+8+len(b))+struct.pack('<II',len(j),0x4e4f534a)+j+struct.pack('<II',len(b),0x004e4942)+b)

if settings["normal_map"]:
 high=trimesh.load(out/'unsigned-main.glb',force='mesh',process=False);hv=np.asarray(high.vertices);hf=np.asarray(high.faces);hn=high.vertex_normals
 reference=mn.meshFromFacesVerts(hf.astype('i4'),hv.astype('f4'));projector=mr.PointsToMeshProjector();projector.updateMeshData(reference)
for budget in map(int,sys.argv[1:]):
 start=time.monotonic();low=trimesh.load(out/f'unsigned-main-qslim-{budget}.glb',force='mesh',process=False)
 # xatlas uses absolute geometric tolerances. Give it a 1000-unit bounding box
 # so small, valid QSlim triangles are not discarded as degenerate. Output
 # positions below still come from the original mesh, without this transform.
 unwrap_positions=np.asarray(low.vertices,dtype=np.float64)
 unwrap_extent=float(np.ptp(unwrap_positions,axis=0).max())
 if not np.isfinite(unwrap_positions).all() or unwrap_extent<=0:raise RuntimeError('Invalid mesh coordinates for UV unwrap')
 unwrap_positions=(unwrap_positions-unwrap_positions.mean(axis=0))*(1000.0/unwrap_extent)
 atlas=xatlas.Atlas();atlas.add_mesh(np.asarray(unwrap_positions,dtype='f4'),np.asarray(low.faces,dtype='u4'));pack=xatlas.PackOptions();pack.resolution=size;pack.padding=settings["texture_padding"];atlas.generate(pack_options=pack)
 mapping,faces,uv=atlas[0];v=np.asarray(low.vertices)[mapping];n=np.asarray(low.vertex_normals)[mapping];f=faces.astype('i4');uv=np.asarray(uv,dtype=np.float64)
 # Smooth tangent basis, retained explicitly in GLB for a consistent bake/view basis.
 p=v[f];duv=uv[f];e1=p[:,1]-p[:,0];e2=p[:,2]-p[:,0];d1=duv[:,1]-duv[:,0];d2=duv[:,2]-duv[:,0];den=d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0]
 if np.any(np.abs(den)<1e-15):raise RuntimeError('Degenerate UV triangle')
 ft=(e1*d2[:,1,None]-e2*d1[:,1,None])/den[:,None];fb=(-e1*d2[:,0,None]+e2*d1[:,0,None])/den[:,None]
 ta=np.zeros_like(v);ba=np.zeros_like(v)
 for k in range(3):np.add.at(ta,f[:,k],ft);np.add.at(ba,f[:,k],fb)
 ta=unit(ta-n*np.sum(n*ta,axis=1,keepdims=True));hand=np.where(np.sum(np.cross(n,ta)*ba,axis=1)<0,-1.,1.);tang=np.column_stack([ta,hand])
 tri=np.full((size,size),-1,np.int32);bary=np.zeros((size,size,3),np.float32)
 for i,q in enumerate(uv[f]*size-.5):
  lo=np.maximum(np.ceil(q.min(axis=0)).astype(int),0);hi=np.minimum(np.floor(q.max(axis=0)).astype(int),size-1)
  if np.any(hi<lo):continue
  xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1));xy=np.column_stack([xx.ravel(),yy.ravel()]);a=q[0];d=q[1]-a;e=q[2]-a;det=d[0]*e[1]-d[1]*e[0];r=xy-a
  b=(r[:,0]*e[1]-r[:,1]*e[0])/det;c=(d[0]*r[:,1]-d[1]*r[:,0])/det;w=np.column_stack([1-b-c,b,c]);keep=(w>=-1e-7).all(axis=1);x=xy[keep,0];y=xy[keep,1];tri[y,x]=i;bary[y,x]=w[keep]
 mask=tri>=0;ids=f[tri[mask]];w=bary[mask];positions=np.sum(v[ids]*w[:,:,None],axis=1);ln=unit(np.sum(n[ids]*w[:,:,None],axis=1));lt=unit(np.sum(ta[ids]*w[:,:,None],axis=1));lt=unit(lt-ln*np.sum(ln*lt,axis=1,keepdims=True));lh=np.where(np.sum(hand[ids]*w,axis=1)<0,-1.,1.);lb=np.cross(ln,lt)*lh[:,None]
 if settings["normal_map"]:
  print(budget,'UV pixels',len(positions),'projecting',flush=True)
  proj=mr.std_vector_MeshProjectionResult();projector.findProjections(proj,mn.fromNumpyArray(positions.astype('f4')),None,None,1e30,0.)
  faceids=np.fromiter((int(r.proj.face) for r in proj),dtype=np.int64,count=len(positions));closest=np.array([[r.proj.point.x,r.proj.point.y,r.proj.point.z] for r in proj])
  if np.any(faceids<0):raise RuntimeError('Projection missed source')
  hw=trimesh.triangles.points_to_barycentric(hv[hf[faceids]],closest);sn=unit(np.sum(hn[hf[faceids]]*hw[:,:,None],axis=1));# Correct ambiguous nearest-surface matches using orientation-compatible local candidates.
  bad=np.sum(sn*ln,axis=1)<0.2
  from scipy.spatial import cKDTree
  tree=cKDTree(hv[hf].mean(axis=1))
  badidx=np.flatnonzero(bad)
  for startidx in range(0,len(badidx),2048):
   ix=badidx[startidx:startidx+2048];_,candidates=tree.query(positions[ix],k=32)
   pts=trimesh.triangles.closest_point(hv[hf[candidates]].reshape(-1,3,3),np.repeat(positions[ix],32,axis=0))
   weights=trimesh.triangles.points_to_barycentric(hv[hf[candidates]].reshape(-1,3,3),pts)
   normals=unit(np.sum(hn[hf[candidates]].reshape(-1,3,3)*weights[:,:,None],axis=1)).reshape(-1,32,3)
   distance=np.linalg.norm(pts.reshape(-1,32,3)-positions[ix,None,:],axis=2)
   compatible=np.sum(normals*ln[ix,None,:],axis=2)>0.2
   distance[~compatible]=np.inf;choice=distance.argmin(axis=1);valid=np.isfinite(distance[np.arange(len(ix)),choice])
   sn[ix[valid]]=normals[np.arange(len(ix))[valid],choice[valid]]
  print('Orientation correction candidates',len(badidx),flush=True)
  tn=unit(np.column_stack([np.sum(sn*lt,axis=1),np.sum(sn*lb,axis=1),np.sum(sn*ln,axis=1)]))
 else:
  tn=np.tile([0.,0.,1.],(len(positions),1));closest=positions.copy()
 image=np.zeros((size,size,3),np.uint8);image[:]=[128,128,255];image[mask]=np.clip(np.round((tn*.5+.5)*255),0,255).astype(np.uint8)
 dist,nearest=distance_transform_edt(~mask,return_indices=True);pad=(~mask)&(dist<=settings["texture_padding"]);image[pad]=image[nearest[0][pad],nearest[1][pad]]
 png=io.BytesIO();Image.fromarray(image).save(png,format='PNG');(out/f'resolution-2048-normal-{budget}.png').write_bytes(png.getvalue());write_glb(out/f'resolution-2048-normal-{budget}.glb',v,n,uv,tang,f,png.getvalue())
 report={'budget':budget,'texture_resolution':size,'covered_texels':int(mask.sum()),'seconds':time.monotonic()-start,'projection_distance_p95':float(np.quantile(np.linalg.norm(positions-closest,axis=1),.95))};(out/f'resolution-2048-normal-{budget}.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
