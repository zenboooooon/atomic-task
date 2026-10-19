"""Package learned-policy episodes as side-by-side success/failure videos."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess


def run(args):
    return subprocess.run(args,check=True,capture_output=True,text=True).stdout


def duration(path):
    return float(run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(path)]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--update',type=int)
    parser.add_argument('--group',default='primitives')
    args=parser.parse_args()
    selection=json.loads(args.selection.read_text())
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    files=[];metadata=[]
    for task in selection:
        pair=[]
        for outcome in ['success','failure']:
            item=selection[task].get(outcome)
            target=out/f'{task}_{outcome}.mp4'
            if item:
                if args.update is not None and item['update']!=args.update:
                    raise ValueError('Cannot mix checkpoint updates in one delivery')
                source=Path(item['local_file'])
                run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(source),'-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(target)])
                metadata.append(dict(task=task,outcome=outcome,file=target.name,seed=item['seed'],update=item['update'],reason=item['reason'],checkpoint=item['checkpoint'],checkpoint_sha256=item['checkpoint_sha256']))
            else:
                labels=[f'{args.group} {task} | update {args.update}',
                        f'{outcome.upper()} NOT CAPTURED',
                        'No example found within recording budget',
                        'This is not a success-rate estimate']
                filters=','.join(f"drawtext=text='{label}':fontcolor=white:fontsize=15:x=12:y={100+45*i}" for i,label in enumerate(labels))
                run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i','color=c=black:s=448x528:r=10','-vf',filters,'-t','3','-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(target)])
                metadata.append(dict(task=task,outcome=outcome,file=target.name,update=args.update,missing=True))
            pair.append(target)
        seconds=max(map(duration,pair))
        paired=out/f'{task}_success_and_failure.mp4'
        run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(pair[0]),'-i',str(pair[1]),
             '-filter_complex','[0:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=30[left];[1:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=30[right];[left][right]hstack=inputs=2[v]',
             '-map','[v]','-t',f'{seconds:.3f}','-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(paired)])
        files.append(paired)
    concat=out/'clips.txt';concat.write_text(''.join(f"file '{p.name}'\n" for p in files))
    combined=out/'all_tasks_success_and_failure.mp4'
    run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(concat),'-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(combined)])
    run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(combined),'-f','null','-'])
    report={'recording_type':'learned checkpoint, fresh stochastic rollouts on train layouts','not_formal_validation':True,
            'scripted':False,'layout':'independent episodes; success left, failure right; finished panel held at its final frame',
            'group':args.group,'update':args.update,'videos':metadata,'combined':combined.name,'combined_seconds':duration(combined),
            'playback_verified':True,
            'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.mp4')}}
    (out/'delivery.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
