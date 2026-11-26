from pathlib import Path
import trimesh
out=Path(__file__).parent/'outputs';m=trimesh.load(out/'trellis-1024.glb',force='mesh',process=False);m.merge_vertices();m.remove_unreferenced_vertices();print('Source triangles',len(m.faces),flush=True);trimesh.Trimesh(m.vertices,m.faces,process=False).export(out/'dense-geometry.glb')
