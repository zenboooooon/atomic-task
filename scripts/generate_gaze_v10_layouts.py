"""Keep look/approach layouts and add explicit jump and elevated-treasure layouts."""
from pathlib import Path
import hashlib,json,math,random
ROOT=Path(__file__).resolve().parents[1]
def main():
 source=json.loads((ROOT/'manifests/gaze_v08_layouts.json').read_text())
 doc={k:v for k,v in source.items() if k not in ('sha256','layouts','appearance_protocol','appearance')}
 doc['protocol_version']='gaze-jump-v0.10-single-color'
 doc['appearance']={'target_and_platform_rgb':[1,.38,.03],'avatar_rgb':[.03,.82,.9],'same_across_all_tasks':True}
 doc['layouts']=[dict(x) for x in source['layouts'] if x['task'] in ('look','approach')]
 for task in ['jump','treasure']:
  for split,count in [('train',128),('validation',32),('test',0 if task=='jump' else 32)]:
   rng=random.Random(f'v10|{task}|{split}|20261019')
   for i in range(count):
    angle=rng.uniform(-math.pi,math.pi); f=(math.sin(angle),math.cos(angle)); side=(f[1],-f[0])
    # Small initial lateral offsets; no privileged states enter policy observations.
    distance=rng.uniform(3.2,4.0) if task=='jump' else rng.uniform(5.5,6.3)
    px,pz=f[0]*distance,f[1]*distance
    camera=math.degrees(angle)+(rng.uniform(-20,20) if task=='jump' or i%2==0 else rng.choice([-1,1])*rng.uniform(80,150))
    obstacles=[]
    if task=='treasure':
     for j,d in enumerate([1.55,2.8,3.55]):
      lateral=([.2,-.35,.15][j])+rng.uniform(-.15,.15)
      obstacles.append({'x':f[0]*d+side[0]*lateral,'z':f[1]*d+side[1]*lateral,'radius':.42,'height':1.0})
    target_distance=distance+.55 if task=='treasure' else distance
    row={'task':task,'split':split,'player_x':0.,'player_z':0.,'target_x':f[0]*target_distance,'target_z':f[1]*target_distance,'target_y':2.0 if task=='treasure' else 1.6,'camera_yaw_degrees':camera,'look_start_viewport_x':0.,'initial_speed_mps':0.,'obstacles':obstacles,'has_platform':True,'platform_x':px,'platform_z':pz,'platform_yaw_degrees':math.degrees(angle)}
    doc['layouts'].append(row)
 for i,row in enumerate(doc['layouts']):
  row['layout_id']=hashlib.sha256(json.dumps({'version':doc['protocol_version'],'i':i,**{k:v for k,v in row.items() if k!='layout_id'}},sort_keys=True).encode()).hexdigest()[:20]
 doc['counts_per_task']={t:{s:sum(x['task']==t and x['split']==s for x in doc['layouts']) for s in ('train','validation','test')} for t in ('look','approach','jump','treasure')}
 doc['platform']={'width':3,'depth':2.4,'height':1,'success':'actual jump followed by top contact; treasure requires subsequent treasure contact'}
 doc['sha256']=hashlib.sha256(json.dumps(doc,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 content=json.dumps(doc,indent=2,sort_keys=True)+'\n'
 for p in ['manifests/gaze_v10_layouts.json','manifests/pilot_layouts.json','unity/Assets/StreamingAssets/pilot_layouts.json']:(ROOT/p).write_text(content)
 print(doc['counts_per_task'])
if __name__=='__main__':main()
