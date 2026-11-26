from pathlib import Path
import json,logging,secrets,shutil,subprocess,sys,time,uuid
from native import Native
ROOT=Path(__file__).resolve().parent.parent
TEMPLATES=ROOT/'backend/stages'
LOG=logging.getLogger('trellis-clean')
class Engine:
 def __init__(self,root):
  self.root=Path(root).resolve();self.native=Native(self.root)
 def close(self):self.native.close()
 def generate(self,src,dst,quality='512',background='auto',seed=0,steps=12,texture=2048,guidance=7.5,voxel=384,target=10000,options=None):
  from settings import parse_params
  options=parse_params({},texture) if options is None else options
  run=ROOT/'runs'/f'{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:8]}';out=run/'outputs';out.mkdir(parents=True);started=time.monotonic();seed=seed if seed>0 else secrets.randbits(63)
  report={'status':'running','quality':quality,'seed':seed,'shape_steps':steps,'texture_size':texture,'voxel_resolution':voxel,'target_triangles':target,'stages':{},'run_directory':str(run)}
  shutil.copyfile(src,run/'input.png')
  (run/'settings.json').write_text(json.dumps(options))
  report['options']=options
  def save():(run/'report.json').write_text(json.dumps(report,indent=2))
  stage='generation';save()
  try:
   t=time.monotonic();LOG.info('Run %s stage generation',run.name);report.update(self.native.generate(src,out/'trellis-1024.glb',quality,background,seed,steps,guidance,options["texture_sampling_steps"]));report['stages'][stage]=time.monotonic()-t
   save()
   scripts={}
   scripts['prepare']=(TEMPLATES/'prepare.py').read_text()
   scripts['voxel']=(TEMPLATES/'voxel.py').read_text().replace('/384',f'/{voxel}')
   if options['exterior_only']:
    shutil.copyfile(TEMPLATES/'exterior_helpers.py',run/'exterior_helpers.py')
    scripts['exterior']=(TEMPLATES/'exterior.py').read_text()
   scripts['component']=(TEMPLATES/'component.py').read_text()
   scripts['reduce']=(TEMPLATES/'reduce.py').read_text().replace('[10000]',f'[{target}]')
   scripts['normal']=(TEMPLATES/'normal.py').read_text().replace('size=2048',f'size={options["normal_resolution"]}')
   scripts['color']=(TEMPLATES/'color.py').read_text().replace('size=2048',f'size={options["colour_resolution"]}').replace('10000',str(target))
   for stage,code in scripts.items():
    t=time.monotonic();LOG.info('Run %s stage %s',run.name,stage);path=run/f'{stage}.py';path.write_text(code)
    with (run/f'{stage}.log').open('w') as log:
     result=subprocess.run([sys.executable,str(path)]+([str(target)] if stage=='normal' else []),stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{stage} process exited {result.returncode}; see {run/stage}.log')
    report['stages'][stage]=time.monotonic()-t
    if stage=='exterior':report['exterior']=json.loads((out/'exterior-report.json').read_text())
    save()
   final=out/'triangle-2048-textured.glb'
   if not final.is_file() or final.stat().st_size<1024:raise RuntimeError('Final textured GLB missing or invalid')
   shutil.copyfile(final,dst);report.update(status='complete',total_seconds=time.monotonic()-started);save();return report
  except Exception as error:
   report.update(status='failed',failed_stage=stage,error=str(error),total_seconds=time.monotonic()-started);save();raise RuntimeError(f'TRELLIS clean stage {stage} failed: {error}; run {run}') from error
