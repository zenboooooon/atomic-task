from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,re,collections
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/gaze_v10_2000'
JST=timezone(timedelta(hours=9))
def stamp(p):
 return datetime.fromtimestamp(p.stat().st_mtime,JST).isoformat(timespec='seconds') if p.exists() else None
def small(p,limit=1800):
 if not p.exists():return None
 return p.read_text(errors='replace')[-limit:]
def load_json(p):
 if not p.exists():return None
 try:return json.loads(p.read_text())
 except json.JSONDecodeError:return {'read_error':'JSON incomplete'}
def process_file(p):
 s=small(p,100)
 if not s:return None
 try:pid=int(s.strip())
 except ValueError:return {'pid_text':s.strip(),'alive':False}
 q=Path('/proc')/str(pid)
 try:
  state=(q/'stat').read_text().rpartition(')')[2].split()[0]
  cmd=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  cwd=str((q/'cwd').resolve())
  return {'pid':pid,'alive':state!='Z' and bool(cmd),'state':state,'cmd':cmd[:700],'cwd':cwd}
 except OSError:return {'pid':pid,'alive':False}
def status_files(d):
 names=['pipeline.status','pipeline.pid','pipeline.log','parallelism','run.plan.json']
 return {n:{'exists':(d/n).exists(),'mtime':stamp(d/n),'tail':small(d/n,800) if n not in ['run.plan.json'] else None} for n in names}
def training(d,tasks,multiplier):
 records={}; parse_errors=0; log=d/'train.log'; report=load_json(d/'report.json')
 if log.exists():
  with log.open(errors='replace') as f:
   for line in f:
    if not line.startswith('{'):continue
    try:x=json.loads(line)
    except json.JSONDecodeError:continue
    if isinstance(x,dict) and isinstance(x.get('update'),int):records[x['update']]=x
 if report and isinstance(report.get('updates'),list):
  for row in report['updates']:
   if isinstance(row,dict) and isinstance(row.get('update'),int):records[row['update']]=row
 totals={}
 for task in tasks:
  s=f=i=0
  for row in records.values():
   r=row.get('tasks',{}).get(task,{}) if task is not None else row
   s+=int(r.get('successes',r.get('terminal_successes',0)))
   f+=int(r.get('failures',r.get('terminal_failures',0)))
   i+=int(r.get('interruptions',0))
  n=s+f+i;totals[task or 'treasure']={'successes':s,'failures':f,'interruptions':i,'ended':n,'rate':s/n if n else None}
 checkpoints=sorted(d.glob('checkpoint_*.pt'),key=lambda p:int(p.stem.rsplit('_',1)[-1]))
 text_tail=small(log,1200) or ''
 errors=[x[:300] for x in text_tail.splitlines() if any(k in x for k in ['Traceback (most recent call last)','RuntimeError:','ValueError:','CUDA out of memory'])]
 return {'update':max(records,default=0),'unique_updates':len(records),'missing_updates':bool(records) and set(records)!=set(range(1,max(records)+1)),'decisions':max(records,default=0)*multiplier,'totals':totals,'log_mtime':stamp(log),'latest_checkpoint':checkpoints[-1].name if checkpoints else None,'checkpoint_mtime':stamp(checkpoints[-1]) if checkpoints else None,'report_exists':report is not None,'errors':errors}
def evaluation(p):
 x=load_json(p)
 if not x:return None
 if 'read_error' in x:return x
 return {'environment_seed':x.get('environment_seed'),'agent_decisions_trained':x.get('agent_decisions_trained'),'condition':x.get('condition'),'summary':{task:{k:v.get(k) for k in ['episodes','successes','success_rate']} for task,v in x.get('summary',{}).items()}}
result={'time_jst':datetime.now(JST).isoformat(timespec='seconds'),'primitive':{},'downstream':{}}
result['root']={'files':status_files(BASE),'status':small(BASE/'pipeline.status'),'process':process_file(BASE/'pipeline.pid'),'plan':load_json(BASE/'run.plan.json')}
for seed in range(3):
 d=BASE/f'seed{seed}'; t=d/f'pretrain3_seed{seed}'
 result['primitive'][str(seed)]={'files':status_files(d),'status':small(d/'pipeline.status'),'process':process_file(d/'pipeline.pid'),'training':training(t,['look','approach','jump'],768),'validation':evaluation(d/'pretrain3_validation.json'),'validation_log':{'exists':(d/'pretrain3_validation.log').exists(),'mtime':stamp(d/'pretrain3_validation.log'),'tail':small(d/'pretrain3_validation.log',500)}}
down=BASE/'downstream'
result['downstream']={'files':status_files(down),'status':small(down/'pipeline.status'),'process':process_file(down/'pipeline.pid'),'chain_process':process_file(down/'chain.pid'),'chain_lock':{'exists':(down/'chain.lock').exists(),'content':small(down/'chain.lock',100),'mtime':stamp(down/'chain.lock')},'progress':load_json(down/'progress.json'),'plan':load_json(down/'run.plan.json'),'train_log':{'mtime':stamp(down/'train.log'),'tail':small(down/'train.log',800)},'validation_log':{'mtime':stamp(down/'validation.log'),'tail':small(down/'validation.log',800)},'jobs':{}}
for seed in range(3):
 for condition in ['P','S','F','R']:
  name=f'{condition}_seed{seed}';d=down/name
  validations={p.stem:evaluation(p) for p in sorted(d.glob('validation_*.json'))}
  result['downstream']['jobs'][name]={'exists':d.exists(),'status':small(d/'pipeline.status'),'process':process_file(d/'pipeline.pid'),'training':training(d,[None],256),'validations':validations,'validation_logs':[{'name':p.name,'mtime':stamp(p)} for p in sorted(d.glob('validation_*.log'))]}

# The original supervisor still owns progress.json for its nine P/S/F jobs.
# R has separate control metadata; combined files describe all twelve jobs.
ext=down/'r_extension'
if ext.exists():
 result['downstream']['r_extension']={'files':status_files(ext),'status':small(ext/'pipeline.status'),'process':process_file(ext/'pipeline.pid'),'progress':load_json(ext/'progress.json')}
combined=load_json(down/'combined.progress.json')
if combined and 'read_error' not in combined:
 result['downstream']['original_progress']=result['downstream']['progress']
 result['downstream']['original_status']=result['downstream']['status']
 result['downstream']['progress']=combined
 result['downstream']['status']=small(down/'combined.pipeline.status')
 result['downstream']['plan']=load_json(down/'combined.run.plan.json')
result['downstream']['combined_files']={n:{'exists':(down/n).exists(),'mtime':stamp(down/n),'tail':small(down/n,800) if n=='combined.pipeline.status' else None} for n in ['combined.pipeline.status','combined.progress.json','combined.run.plan.json']}

# Find actual trainers/players as well as the wrapper PIDs, without executing or signaling them.
processes=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=str((p/'cwd').resolve());state=(p/'stat').read_text().rpartition(')')[2].split()[0]
  if (cwd==str(ROOT) or str(ROOT) in cmd) and any(k in cmd for k in ['train_pretraining_pilot.py','train_downstream_pilot.py','evaluate_multi_pilot.py','TpsPretrainJump.x86_64','TpsTreasure.x86_64','run_gaze_v10_','wait_gaze_v10_']):
   processes.append({'pid':int(p.name),'state':state,'cmd':cmd[:700]})
 except OSError:pass
result['processes']=processes
print(json.dumps(result,ensure_ascii=False))
