import math,re
DEFAULTS={'target_triangles':None,'exterior_only':False,'sampling_steps':12,'texture_sampling_steps':12,'voxel_resolution':384,'shell_thickness_voxels':1.5,'components':'tiny','minimum_component_area_ratio':0.0001,'normal_map':True,'normal_resolution':None,'colour_resolution':None,'texture_padding':8,'roughness':1.0,'metallic':0.0}
def parse_params(params,texture):
 unknown=set(params)-set(DEFAULTS)
 if unknown:raise ValueError('Unknown parameters: '+str(sorted(unknown)))
 out=dict(DEFAULTS);out.update(normal_resolution=texture,colour_resolution=texture)
 for key,value in params.items():
  if key in {'target_triangles','sampling_steps','texture_sampling_steps','voxel_resolution','normal_resolution','colour_resolution','texture_padding'}:
   if not re.fullmatch(r'[0-9]+',value):raise ValueError(key+' must be an integer')
   number=int(value)
   if number < (0 if key=='texture_padding' else 1):raise ValueError(key+' is out of range')
   out[key]=number
  elif key in {'shell_thickness_voxels','roughness','metallic','minimum_component_area_ratio'}:
   try:number=float(value)
   except ValueError:raise ValueError(key+' must be numeric')
   if not math.isfinite(number) or (number<=0 if key=='shell_thickness_voxels' else not 0<=number<=1):raise ValueError(key+' is out of range')
   out[key]=number
  elif key in {'normal_map','exterior_only'}:
   if value not in {'true','false'}:raise ValueError(key+' must be true or false')
   out[key]=value=='true'
  elif key=='components':
   if value not in {'tiny','largest','all'}:raise ValueError('components must be tiny, largest or all')
   out[key]=value
 return out
