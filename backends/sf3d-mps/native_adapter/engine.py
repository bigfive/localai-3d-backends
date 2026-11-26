"""SF3D FP32/MPS geometry with albedo, predicted normals and matte material."""
import gc
import logging
import os
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image,ImageOps
import torch
import trimesh
import rembg
import xatlas
from scipy import ndimage
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'experiment'))
from mps_attention import install
install()
from sf3d.system import SF3D
from sf3d.utils import resize_foreground,default_cond_c2w,create_intrinsic_from_fov_deg
LOG=logging.getLogger('sf3d-native')

class Engine:
 def __init__(self,root):
  self.root=Path(root).resolve(strict=True)
  if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
  if not (Path(os.environ['U2NET_HOME'])/'u2net.onnx').is_file():raise FileNotFoundError('Downloaded u2net.onnx required')
  self.background_session=rembg.new_session('u2net',providers=['CPUExecutionProvider'])
  self.model=SF3D.from_pretrained(str(self.root),'config.yaml','model.safetensors').to('mps').eval()
  self.model.image_estimator=None
  self.model.global_estimator=None

 def prepare(self,src,background):
  with Image.open(src) as opened:image=ImageOps.exif_transpose(opened).convert('RGBA')
  rgba=np.asarray(image).copy()
  if background=='auto':
   if rgba[:,:,3].min()==255:image=rembg.remove(image,session=self.background_session)
  elif background in ('black','white'):
   rgb=rgba[:,:,:3];candidate=rgb.max(2)<=12 if background=='black' else rgb.min(2)>=243
   labels,_=ndimage.label(candidate)
   border=np.unique(np.concatenate((labels[0],labels[-1],labels[:,0],labels[:,-1])))
   rgba[:,:,3][np.isin(labels,border[border!=0])]=0;image=Image.fromarray(rgba)
  elif background!='keep':raise ValueError(f'Unsupported background: {background}')
  if not (np.asarray(image)[:,:,3]>0).any():raise ValueError('No visible foreground')
  return resize_foreground(image,.85)

 @torch.inference_mode()
 def generate(self,src,dst,texture=1024,background='auto',*,target_triangles=None,allow_larger_target=False):
  try:
   start=time.monotonic();image=self.prepare(src,background);preprocess=time.monotonic()-start
   torch.mps.synchronize();begin=time.monotonic()
   mask_cond,rgb=self.model.prepare_image(image)
   intrinsic,normed=create_intrinsic_from_fov_deg(self.model.cfg.default_fovy_deg,512,512)
   batch={'rgb_cond':rgb[None],'mask_cond':mask_cond[None],
    'c2w_cond':default_cond_c2w(self.model.cfg.default_distance).to('mps').view(1,1,4,4),
    'intrinsic_cond':intrinsic.to('mps').view(1,1,3,3),
    'intrinsic_normed_cond':normed.to('mps').view(1,1,3,3)}
   for key in ('rgb_cond','mask_cond'):batch[key]=self.model.image_processor(batch[key],512)
   codes,_=self.model.get_scene_codes(batch)
   surface=self.model.triplane_to_meshes(codes)[0]
   if not len(surface.t_pos_idx):raise RuntimeError('SF3D generated an empty mesh')
   source_triangles=len(surface.t_pos_idx)
   if target_triangles is not None:
    if type(target_triangles) is not int or target_triangles<1 or (target_triangles>=source_triangles and not allow_larger_target):
     raise ValueError('target_triangles must be a positive integer below the source triangle count')
   if target_triangles is not None and target_triangles<source_triangles:
    import gpytoolbox
    vertices,triangles,*_=gpytoolbox.decimate(surface.v_pos.cpu().numpy().astype('float64'),surface.t_pos_idx.cpu().numpy().astype('int32'),num_faces=target_triangles,method='qslim')
    if not len(triangles) or not np.isfinite(vertices).all():raise RuntimeError('QSlim returned an invalid mesh')
    surface=type(surface)(torch.as_tensor(vertices,dtype=torch.float32,device='mps'),torch.as_tensor(triangles,dtype=torch.int64,device='mps'))
   source_positions=surface.v_pos.cpu().numpy()
   source_normals=surface.v_nrm.cpu().numpy()
   extent=float(np.ptp(source_positions,axis=0).max())
   if not np.isfinite(source_positions).all() or extent<=0:raise RuntimeError('Invalid mesh coordinates for UV unwrap')
   unwrap_positions=(source_positions-source_positions.mean(0))*(1000./extent)
   atlas=xatlas.Atlas()
   atlas.add_mesh(unwrap_positions.astype('float32'),surface.t_pos_idx.cpu().numpy().astype('uint32'))
   pack=xatlas.PackOptions();pack.resolution=texture;pack.padding=8
   atlas.generate(pack_options=pack)
   mapping,faces,uv=atlas[0]
   # Measure UV orientation in double precision. Small nonzero UV triangles
   # still define a tangent frame and should survive export.
   uv_triangles=uv[faces].astype(np.float64)
   d1=uv_triangles[:,1]-uv_triangles[:,0]
   d2=uv_triangles[:,2]-uv_triangles[:,0]
   det=d1[:,0]*d2[:,1]-d1[:,1]*d2[:,0]
   if not np.isfinite(det).all():raise RuntimeError('Non-finite UV tangent frame')
   geometry=source_positions[mapping[faces]].astype(np.float64)
   e1=geometry[:,1]-geometry[:,0]
   e2=geometry[:,2]-geometry[:,0]
   if (det==0).any():raise RuntimeError('UV unwrap produced a zero-area triangle')
   face_tangents=(e1*d2[:,1,None]-e2*d1[:,1,None])/det[:,None]
   if not np.isfinite(face_tangents).all():raise RuntimeError('Non-finite UV tangent vectors')
   # Keep corner-indexed storage for baking; chart-shared tangents are joined below.
   corner_indices=mapping[faces].reshape(-1)
   surface.v_pos=torch.as_tensor(source_positions[corner_indices],device='mps')
   surface.t_pos_idx=torch.arange(len(corner_indices),device='mps',dtype=surface.t_pos_idx.dtype).reshape(-1,3)
   surface._v_tex=torch.as_tensor(uv[faces].reshape(-1,2),device='mps')
   smooth=source_normals[corner_indices]
   surface._v_nrm=torch.as_tensor(smooth,device='mps')
   surface._v_tng=None
   baker=self.model.baker
   # Metal samples at integer texel coordinates. Offset the UVs so the
   # resulting barycentrics describe the pixel centers used by GL texture filtering.
   raster_uv=(surface.v_tex+torch.tensor([-.5/texture,.5/texture],device='mps')).contiguous()
   rast=baker.rasterize(raster_uv,surface.t_pos_idx,texture);mask=baker.get_mask(rast)
   if not mask.any():raise RuntimeError('UV rasterizer covered no texels')
   positions=baker.interpolate(surface.v_pos,rast,surface.t_pos_idx)[mask]
   gb_nrm=torch.nn.functional.normalize(baker.interpolate(surface.v_nrm,rast,surface.t_pos_idx)[mask],dim=-1)
   normals=surface.v_nrm
   # Handedness is constant across all three corners of each triangle.
   signs=torch.as_tensor(np.repeat(np.sign(det),3)[:,None],dtype=normals.dtype,device='mps')
   # UV unwrap duplicates every face corner. Accumulate tangents across
   # corners that share position, UV, smooth normal and UV orientation.
   # Otherwise tangent-space normal values jump across internal chart edges.
   keys=np.column_stack((surface.v_pos.cpu().numpy(),surface.v_tex.cpu().numpy(),normals.cpu().numpy(),signs.cpu().numpy()))
   _,groups=np.unique(keys,axis=0,return_inverse=True)
   accumulated=np.zeros((int(groups.max())+1,3),dtype=np.float64)
   np.add.at(accumulated,groups,np.repeat(face_tangents,3,axis=0))
   shared_t=torch.as_tensor(accumulated[groups].astype('float32'),device='mps')
   tangents=torch.nn.functional.normalize(shared_t-(shared_t*normals).sum(-1,keepdim=True)*normals,dim=-1)

   gb_tng=baker.interpolate(tangents,rast,surface.t_pos_idx)[mask]
   gb_tng=torch.nn.functional.normalize(gb_tng-(gb_tng*gb_nrm).sum(-1,keepdim=True)*gb_nrm,dim=-1)
   # Handedness is constant per UV triangle. The Metal interpolator only
   # accepts float3 attributes; never pass this scalar through that kernel.
   face_signs=signs.reshape(-1,3,1)
   if not (face_signs==face_signs[:,0:1]).all():raise RuntimeError('Inconsistent tangent handedness within a triangle')
   gb_sign=face_signs[:,0][rast[mask][:,3].long()]
   gb_btng=torch.cross(gb_nrm,gb_tng,dim=-1)*gb_sign
   predicted=[]
   for points in positions.split(262144):
    features=self.model.query_triplane(points,codes[0])[0]
    decoded=self.model.decoder(features,include=['perturb_normal'])
    predicted.append(decoded['perturb_normal'])
   predicted=torch.nn.functional.normalize(torch.cat(predicted),dim=-1)
   if not torch.isfinite(predicted).all():raise RuntimeError('Non-finite predicted normals')
   tangent_normal=torch.stack(((predicted*gb_tng).sum(-1),(predicted*gb_btng).sum(-1),(predicted*gb_nrm).sum(-1)),dim=-1)
   # Export inverts normals and flips UV V; the final frame negates X only.
   tangent_normal[:,0]*=-1
   normal_tex=torch.zeros((texture,texture,3),device='mps');normal_tex[mask]=(tangent_normal*.5+.5).clamp(0,1)
   normal_tex=normal_tex.cpu().numpy()
   # Each empty texel belongs to its nearest covered chart texel. Copy its
   # vector exactly instead of averaging unrelated tangent frames in the gutters.
   nearest=ndimage.distance_transform_edt(~mask.cpu().numpy(),return_distances=False,return_indices=True)
   normal_tex=normal_tex[tuple(nearest)]
   normal_image=Image.fromarray(np.rint(np.clip(normal_tex,0,1)*255).astype('uint8'))
   albedo_size=texture
   albedo_uv=(surface.v_tex+torch.tensor([-.5/albedo_size,.5/albedo_size],device='mps')).contiguous()
   albedo_rast=baker.rasterize(albedo_uv,surface.t_pos_idx,albedo_size)
   albedo_mask=baker.get_mask(albedo_rast)
   if not albedo_mask.any():raise RuntimeError('Albedo UV rasterizer covered no texels')
   albedo_positions=baker.interpolate(surface.v_pos,albedo_rast,surface.t_pos_idx)[albedo_mask]
   colors=[]
   for points in albedo_positions.split(262144):
    features=self.model.query_triplane(points,codes[0])[0]
    colors.append(self.model.decoder(features,include=['features'])['features'].cpu().numpy())
   albedo_occupied=albedo_mask.cpu().numpy()
   tex=np.zeros((albedo_size,albedo_size,3),dtype='float32')
   tex[albedo_occupied]=np.concatenate(colors)
   albedo_nearest=ndimage.distance_transform_edt(~albedo_occupied,return_distances=False,return_indices=True)
   tex=tex[tuple(albedo_nearest)]
   albedo=Image.fromarray(np.rint(np.clip(tex,0,1)*255).astype('uint8'))
   material=trimesh.visual.material.PBRMaterial(baseColorTexture=albedo,normalTexture=normal_image,
    metallicRoughnessTexture=Image.new('RGB',(1,1),(255,255,0)),roughnessFactor=1.,metallicFactor=0.)
   mesh=trimesh.Trimesh(vertices=surface.v_pos.cpu().numpy(),faces=surface.t_pos_idx.cpu().numpy(),vertex_normals=smooth,
    visual=trimesh.visual.texture.TextureVisuals(uv=surface.v_tex.cpu().numpy(),material=material),process=False)
   mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90),[1,0,0]));mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(90),[0,1,0]));mesh.invert()
   torch.mps.synchronize();generation=time.monotonic()-begin
   target=Path(dst);target.parent.mkdir(parents=True,exist_ok=True)
   rotation=trimesh.transformations.rotation_matrix(np.radians(90),[0,1,0])[:3,:3]@trimesh.transformations.rotation_matrix(np.radians(-90),[1,0,0])[:3,:3]
   mesh.vertex_attributes['TANGENT']=np.column_stack((tangents.cpu().numpy()@rotation.T,signs.cpu().numpy())).astype('float32')
   def tangent_semantic(tree):
    for item in tree['meshes']:
     for primitive in item['primitives']:
      attrs=primitive['attributes']
      attrs['TANGENT']=attrs.pop('_TANGENT')
   mesh.export(target,file_type='glb',include_normals=True,tree_postprocessor=tangent_semantic)
   result={'runtime':'pytorch-mps','faces':len(mesh.faces),'source_triangles':source_triangles,'target_triangles':target_triangles,'texture_size':texture,'albedo_size':albedo_size,
    'normal_map':True,'uv_unwrap':'xatlas','normal_padding':'nearest-covered-texel','uv_padding_pixels':8,'covered_texels':int(mask.sum()),'roughness':float(material.roughnessFactor),'metallic':float(material.metallicFactor),
    'timings':{'preprocess_s':preprocess,'generation_s':generation,'total_s':time.monotonic()-start},'bytes':target.stat().st_size}
   LOG.info('Generated native SF3D: %s',result)
   return result
  finally:gc.collect();torch.mps.empty_cache()

 def close(self):
  self.model=self.background_session=None;gc.collect();torch.mps.empty_cache()
