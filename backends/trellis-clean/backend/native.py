import ctypes as c
import os
from pathlib import Path
import numpy as np
class Native:
 def __init__(self,root):
  self.root=Path(root);self.lib=c.CDLL(os.environ['TRELLIS_LIBRARY']);l=self.lib;fp=c.POINTER(c.c_float);ip=c.POINTER(c.c_int32);self.fp=fp;self.ip=ip
  l.t2_abi_version.restype=c.c_int32
  if l.t2_abi_version()!=11:raise RuntimeError('Expected TRELLIS ABI 11')
  l.t2_pipeline_load.argtypes=[c.c_char_p]*10+[c.c_int32,c.c_char_p,c.c_int32];l.t2_pipeline_load.restype=c.c_void_p
  l.t2_generate.argtypes=[c.c_void_p,c.c_void_p,c.c_int32,c.c_int32,c.c_int32,c.c_uint64,c.c_int32,c.c_float,c.c_int32]+[c.c_void_p]*4+[c.c_char_p,c.c_int32];l.t2_generate.restype=c.c_void_p
  for name,typ in [('t2_mesh_n_verts',c.c_int32),('t2_mesh_n_tris',c.c_int32),('t2_mesh_verts',fp),('t2_mesh_tris',ip),('t2_mesh_pbr',fp),('t2_mesh_has_pbr',c.c_int32)]:
   f=getattr(l,name);f.argtypes=[c.c_void_p];f.restype=typ
  for name in ['t2_mesh_free','t2_pipeline_free','t2_free_buffer']:getattr(l,name).argtypes=[c.c_void_p];getattr(l,name).restype=None
  l.t2_bake_glb.argtypes=[fp,c.c_int32,ip,c.c_int32,fp,c.c_int32,c.c_int32,c.POINTER(c.c_int32),c.c_char_p,c.c_int32];l.t2_bake_glb.restype=c.c_void_p
  names=['dino_f16.gguf','ss_flow_f16.gguf','ss_dec_f16.gguf','slat_flow_f16.gguf','slat_flow_1024_f16.gguf','shape_dec_f16.gguf','shape_enc_f16.gguf','tex_dec_f16.gguf','tex_slat_flow_512_f16.gguf','tex_slat_flow_1024_f16.gguf']
  paths=[str((self.root/n).resolve(strict=True)).encode() for n in names];err=c.create_string_buffer(2048);self.p=l.t2_pipeline_load(*paths,0,err,len(err))
  if not self.p:raise RuntimeError('TRELLIS load: '+err.value.decode())
 def close(self):
  if self.p:self.lib.t2_pipeline_free(self.p);self.p=None
 def generate(self,src,dst,quality,background,seed,steps,guidance,texture_steps=12):
  l=self.lib;data=Path(src).read_bytes();buf=c.create_string_buffer(data);err=c.create_string_buffer(2048)
  mesh=l.t2_generate(self.p,buf,len(data),{'512':2,'1024':3}[quality],{'auto':0,'keep':1,'black':2,'white':3}[background],seed,steps,guidance,texture_steps,None,None,None,None,err,len(err))
  if not mesh:raise RuntimeError('TRELLIS generation: '+err.value.decode())
  try:
   nv=l.t2_mesh_n_verts(mesh);nf=l.t2_mesh_n_tris(mesh)
   if not nv or not nf:raise RuntimeError('Empty TRELLIS mesh')
   if not l.t2_mesh_has_pbr(mesh):raise RuntimeError('TRELLIS did not generate colour')
   length=c.c_int32();glb=l.t2_bake_glb(l.t2_mesh_verts(mesh),nv,l.t2_mesh_tris(mesh),nf,l.t2_mesh_pbr(mesh),0,2,c.byref(length),err,len(err))
   if not glb or length.value<=0:raise RuntimeError('TRELLIS GLB export: '+err.value.decode())
   try:Path(dst).write_bytes(c.string_at(glb,length.value))
   finally:l.t2_free_buffer(glb)
   return {'source_vertices':nv,'source_triangles':nf}
  finally:l.t2_mesh_free(mesh)
