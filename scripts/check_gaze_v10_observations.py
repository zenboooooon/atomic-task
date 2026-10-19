"""Check the actual five-action, single-camera, single-palette Unity interfaces."""
import json
from pathlib import Path
import cv2
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from tps_transfer.smoke import _rgb_at

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/v10_smoke'


def main():
    frames={}; checks={}
    for player,task,worker in [('TpsPretrainJump','pretrain',614),('TpsTreasure','treasure',615)]:
        env=UnityEnvironment(file_name=str(ROOT/f'unity/Build/{player}.x86_64'),worker_id=worker,seed=9000,no_graphics=False,
                             additional_args=['-logFile',str(OUT/f'{player}_observation.player.log'),f'--tps-task={task}','--tps-split=validation'])
        try:
            env.reset()
            for name,spec in env.behavior_specs.items():
                head=name.split('?',1)[0].removeprefix('TPS_')
                assert spec.action_spec.continuous_size==5 and spec.action_spec.discrete_size==0
                assert len(spec.observation_specs)==1
                decisions,terminals=env.get_steps(name)
                assert len(decisions)==(16 if task=='treasure' else 6)
                assert not len(terminals) and np.all(decisions.reward==0)
                rgb=_rgb_at(decisions.obs,0); assert rgb.shape==(224,224,3)
                r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
                orange=int(((r>.25)&(r>1.3*g)&(g>1.5*b)).sum())
                cyan=int(((g>2*r)&(b>2*r)&(g>.3)).sum())
                assert orange>=5 and cyan>20,(head,orange,cyan)
                checks[head]={'continuous_actions':5,'camera_sensors':1,'initial_reward':0,'orange_pixels':orange,'cyan_pixels':cyan}
                frame=cv2.resize(cv2.cvtColor((np.clip(rgb,0,1)*255).astype('uint8'),cv2.COLOR_RGB2BGR),(448,448))
                cv2.putText(frame,head.upper(),(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
                frames[head]=frame
        finally:env.close()
    cv2.imwrite(str(OUT/'environment_contact.jpg'),np.vstack([np.hstack([frames['look'],frames['approach']]),np.hstack([frames['jump'],frames['treasure']])]))
    (OUT/'observation_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
