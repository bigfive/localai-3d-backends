from pathlib import Path
import json,trimesh,numpy as np
settings=json.loads((Path(__file__).parent/'settings.json').read_text())
out=Path(__file__).parent/'outputs'
m=trimesh.load(out/('unsigned-exterior.glb' if settings['exterior_only'] else 'unsigned-384.glb'),force='mesh',process=False)
parts=trimesh.graph.connected_components(m.face_adjacency,nodes=np.arange(len(m.faces)),min_len=1)
if not parts:raise RuntimeError('Voxel mesh has no components')
areas=np.array([m.area_faces[p].sum() for p in parts]);total=float(areas.sum())
if not np.isfinite(total) or total<=0:raise RuntimeError('Invalid component surface area')
mode=settings['components']
if mode=='largest':selected=[int(np.argmax(areas))]
elif mode=='all':selected=list(range(len(parts)))
elif mode=='tiny':
 selected=np.flatnonzero(areas>=total*settings['minimum_component_area_ratio']).tolist()
 if not selected:raise RuntimeError('Component threshold removes every component')
else:raise ValueError('Unsupported component mode: '+mode)
q=m.submesh([np.concatenate([parts[i] for i in selected])],append=True)
report={'mode':mode,'input_components':len(parts),'retained_components':len(selected),'removed_components':len(parts)-len(selected),'retained_area_ratio':float(areas[selected].sum()/total),'triangles':len(q.faces),'watertight':bool(q.is_watertight)}
q.export(out/'unsigned-main.glb');(out/'components.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
