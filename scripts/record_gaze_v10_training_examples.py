"""Record first successes and failures from a frozen learned v0.10 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from tps_transfer.build_identity import layout_manifest_sha256, unity_build_sha256
from tps_transfer.depth import DepthEstimator
from tps_transfer.episode import EpisodeTracker
from tps_transfer.model import ActorHead, ModelConfig, SharedEncoder, TanhGaussian
from tps_transfer.observation import FrameProcessor, history_batch
from tps_transfer.preprocess import DepthCalibration
from tps_transfer.smoke import _rgb_at, _steps

ROOT=Path(__file__).resolve().parents[1]
TASKS=('look','approach','jump')
SIZE=(448,528)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video_frame(rgb,task,seed,update,number,outcome,reason,final=False,condition=None):
    canvas=np.zeros((SIZE[1],SIZE[0],3),np.uint8)
    canvas[80:]=cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),(448,448))
    color=(80,220,110) if outcome=='success' else (90,120,245)
    task_label=f'{condition} {task.upper()}' if condition else task.upper()
    lines=(f'{task_label} v0.10 | seed {seed} | update {update}',
           'LEARNED POLICY | stochastic | train layouts',
           f'{outcome.upper()} | decision {number:03d} | {number/10:.1f}s')
    for y,label in zip((21,44,68),lines):
        cv2.putText(canvas,label,(8,y),cv2.FONT_HERSHEY_SIMPLEX,.44,color if y==68 else (235,235,235),1,cv2.LINE_AA)
    cv2.drawMarker(canvas,(224,304),(220,220,220),cv2.MARKER_CROSS,12,1)
    if final:
        cv2.rectangle(canvas,(0,470),(447,527),(0,0,0),-1)
        cv2.putText(canvas,reason.replace('_',' ').upper(),(10,504),cv2.FONT_HERSHEY_SIMPLEX,.52,color,1,cv2.LINE_AA)
    return canvas


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--worker-id',type=int,default=630)
    parser.add_argument('--sampling-seed',type=int,default=20261019)
    parser.add_argument('--max-calls',type=int,default=1200)
    parser.add_argument('--max-seconds',type=float,default=0)
    parser.add_argument('--capture-tasks',nargs='+',choices=(*TASKS,'treasure'))
    parser.add_argument('--needed-cases',type=json.loads,
                        help='Task to required outcome list; defaults to success and failure')
    args=parser.parse_args()
    torch.set_num_threads(2)
    path=args.checkpoint.resolve();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    primitive=checkpoint['stage']=='pretraining_pilot'
    if primitive:
        assert tuple(checkpoint['pretraining_tasks'])==TASKS
        tasks=TASKS;condition=None
    else:
        assert checkpoint['stage']=='downstream_pilot' and checkpoint['downstream_task']=='treasure'
        assert checkpoint['condition'] in ('P','S','F')
        tasks=('treasure',);condition=checkpoint['condition']
    capture_tasks=args.capture_tasks or list(tasks)
    assert set(capture_tasks)<=set(tasks)
    needed=args.needed_cases or {task:['success','failure'] for task in capture_tasks}
    assert set(needed)==set(capture_tasks)
    assert all(cases and set(cases)<= {'success','failure'} for cases in needed.values())
    assert checkpoint['model_config']['action_size']==5
    exe=ROOT/('unity/Build/TpsPretrainJump.x86_64' if primitive else 'unity/Build/TpsTreasure.x86_64')
    manifest=ROOT/'manifests/gaze_v10_layouts.json'
    if not primitive:
        exe=ROOT/checkpoint.get('unity_executable','unity/Build/TpsTreasure.x86_64')
        manifest=ROOT/checkpoint.get('layout_manifest','manifests/gaze_v10_layouts.json')
    calibration=ROOT/'manifests/phase1/depth_selected_v02_normalization.json'
    build_hash=unity_build_sha256(exe)
    assert checkpoint['unity_build_sha256']==build_hash
    assert checkpoint['layout_manifest_sha256']==layout_manifest_sha256(exe,manifest)
    assert checkpoint['calibration_sha256']==sha(calibration)
    seed=int(checkpoint['seed'])
    update=int(checkpoint['agent_decisions'])//(768 if primitive else 256)
    device=torch.device('cuda')
    config=ModelConfig(**checkpoint['model_config'])
    shared=SharedEncoder(config).to(device).eval()
    shared.load_state_dict(checkpoint['shared']);shared.requires_grad_(False)
    actors={}
    for task in tasks:
        actor=ActorHead(config.width).to(device).eval()
        actor.load_state_dict(checkpoint['actors'][task] if primitive else checkpoint['actor'])
        actor.requires_grad_(False)
        actors[task]=actor
    processor=FrameProcessor(DepthEstimator(392),DepthCalibration.load(calibration))
    torch.manual_seed(args.sampling_seed);torch.cuda.manual_seed_all(args.sampling_seed)
    environment_seed=seed if primitive else 10000+seed
    env=UnityEnvironment(file_name=str(exe),worker_id=args.worker_id,seed=environment_seed,no_graphics=False,
                         additional_args=['-logFile',str(out/'replay.player.log'),
                                          '--tps-task=pretrain' if primitive else '--tps-task=treasure',
                                          '--tps-split=train'])
    tracker=EpisodeTracker(worker_id=args.worker_id)
    buffers={};actions_by_episode={};returns={};episodes=[];selected={task:{} for task in capture_tasks}
    report={'checkpoint':str(path.relative_to(ROOT)),'checkpoint_sha256':sha(path),'seed':seed,'update':update,
            'build_sha256':build_hash,'layout_manifest_sha256':checkpoint['layout_manifest_sha256'],
            'environment_variant':checkpoint.get('environment_variant'),
            'condition':condition,'split':'train','environment_seed':environment_seed,'sampling_seed':args.sampling_seed,
            'action_policy':'TanhGaussian.sample, unchanged training distribution',
            'recording_type':'fresh rollout from frozen learned checkpoint','scripted':False,
            'weight_updates_during_recording':False,'training_continued_separately':True,
            'selection':'first completed success and first completed failure per task',
            'capture_tasks':capture_tasks,'needed_cases':needed,
            'videos':[],'episodes':episodes,'complete':False}
    def finished(task):
        return all(case in selected[task] for case in needed[task])
    def save_report():
        report['videos']=[dict(task=task,**item) for task,cases in selected.items() for item in cases.values()]
        report['complete']=all(finished(task) for task in selected)
        pending=out/'manifest.json.pending';pending.write_text(json.dumps(report,indent=2)+'\n');pending.replace(out/'manifest.json')
    try:
        def stop_recording(signum,frame):
            raise KeyboardInterrupt('recording interrupted')
        signal.signal(signal.SIGTERM,stop_recording)
        signal.signal(signal.SIGINT,stop_recording)
        started=time.monotonic()
        env.reset()
        behaviors={name.split('?',1)[0].removeprefix('TPS_'):name for name in env.behavior_specs}
        assert set(behaviors)==set(tasks)
        for calls in range(args.max_calls):
            if args.max_seconds and time.monotonic()-started>=args.max_seconds:
                report['stop_reason']='recording_time_budget'
                break
            for task in tasks:
                behavior=behaviors[task]
                decisions,terminals=env.get_steps(behavior)
                terminal_images={int(agent):_rgb_at(terminals.obs,row) for row,agent in enumerate(terminals.agent_id)}
                transitions=tracker.observe(behavior,_steps(decisions,False),_steps(terminals,True))
                for transition in transitions:
                    key=transition.key
                    if task not in selected or finished(task):continue
                    returns[key]=returns.get(key,0.0)+transition.reward
                    if not (transition.terminated or transition.truncated):continue
                    success=bool(transition.terminated and transition.reward>.5)
                    outcome='success' if success else 'failure'
                    reason=({'look':'central_hold_0.5s','approach':'target_contact',
                             'jump':'landed_on_platform_top','treasure':'landed_and_treasure_contact'}[task]
                            if success else 'time_limit' if transition.truncated else 'terminal_failure')
                    images=buffers.pop(key,[]);episode_actions=actions_by_episode.pop(key,[])
                    info={'agent_id':key.agent_id,'episode_ordinal':key.episode_id,'success':success,
                          'interrupted':transition.truncated,'reason':reason,'decision_frames':len(images),
                          'cumulative_reward':returns.pop(key)}
                    episodes.append(dict(task=task,**info))
                    if outcome not in needed[task] or outcome in selected[task] or not images:continue
                    filename=f'{task}_{outcome}_seed{seed}_update{update:04d}.mp4'
                    writer=cv2.VideoWriter(str(out/filename),cv2.VideoWriter_fourcc(*'mp4v'),10,SIZE)
                    if not writer.isOpened():raise RuntimeError('Cannot write video')
                    try:
                        for number,rgb in enumerate(images):
                            writer.write(video_frame(rgb,task,seed,update,number,outcome,reason,condition=condition))
                        last=np.clip(np.rint(terminal_images[key.agent_id]*255),0,255).astype(np.uint8)
                        final=video_frame(last,task,seed,update,len(images),outcome,reason,True,condition)
                        for _ in range(15):writer.write(final)
                    finally:writer.release()
                    np.save(out/f'{task}_{outcome}_actions.npy',np.stack(episode_actions))
                    cv2.imwrite(str(out/f'{task}_{outcome}_final.jpg'),final)
                    selected[task][outcome]=dict(info,outcome=outcome,file=filename)
                    print(json.dumps({'selected_task':task,'selected_outcome':outcome,**info}),flush=True)
                    save_report()
                values=np.zeros((len(decisions),5),np.float32)
                ids=[int(agent) for agent in decisions.agent_id]
                if ids:
                    recording=task in selected and not finished(task)
                    if recording:
                        histories=[tracker.history_for(behavior,agent) for agent in ids]
                        images,previous,valid=history_batch(histories,processor,device)
                        with torch.inference_mode():
                            mean,log_std=actors[task](shared(images,previous,valid))
                            actions,_=TanhGaussian(mean,log_std).sample()
                            values=actions.cpu().numpy().astype(np.float32)
                    for row,agent in enumerate(ids):
                        key=tracker.key_for(behavior,agent)
                        if recording:
                            rgb=np.clip(np.rint(_rgb_at(decisions.obs,row)*255),0,255).astype(np.uint8)
                            buffers.setdefault(key,[]).append(rgb)
                            actions_by_episode.setdefault(key,[]).append(values[row].copy())
                        tracker.record_action(behavior,agent,values[row])
                    env.set_actions(behavior,ActionTuple(continuous=values))
            if all(finished(task) for task in selected):break
            env.step()
            if (calls+1)%25==0:
                processor.clear()
                print(json.dumps({'calls':calls+1,'selected':{t:list(x) for t,x in selected.items()},'episodes':len(episodes)}),flush=True)
        report['env_step_calls']=calls+1
    except KeyboardInterrupt:
        report['stop_reason']='recording_interrupted'
    finally:
        env.close();save_report()
    print(json.dumps({'complete':report['complete'],'videos':report['videos']},indent=2),flush=True)


if __name__=='__main__':main()
