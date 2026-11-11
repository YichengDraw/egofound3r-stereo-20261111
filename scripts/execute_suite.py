#!/usr/bin/env python3
"""Serial, locked execution of the prepared official reproduction commands."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

def stamp():
    return datetime.now(timezone.utc).isoformat()

def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def select_gpu(root):
    q = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in q.splitlines():
        idx, name, uuid, free, util = [x.strip() for x in line.split(',')]
        rows.append(dict(index=int(idx), name=name, uuid=uuid, free_mib=int(free), utilization=int(util)))
    mem = {s.split(':')[0]: int(s.split()[1]) for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')}
    snapshot = dict(time_utc=stamp(), gpus=rows, mem_available_kib=mem['MemAvailable'])
    with (root/'reports/resource_snapshots.jsonl').open('a') as f:
        f.write(json.dumps(snapshot) + '\n')
    eligible = [r for r in rows if 'L40S' in r['name'] and r['free_mib'] >= 22528]
    if not eligible or mem['MemAvailable'] < 32 * 1024 ** 2:
        return None
    return max(eligible, key=lambda r: (r['free_mib'], -r['utilization']))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--run-id', action='append')
    args=ap.parse_args()
    root=args.root.resolve()
    lock=(root/'reports/execution.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(f'{os.getpid()} {stamp()}\n'); lock.flush()
    upstream=root/'upstream'
    sha=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    if sha!='264f4df9fd42536aae4dba982d18272cc3334077': raise RuntimeError('Unexpected upstream revision')
    if subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True).strip(): raise RuntimeError('Upstream has unrecorded changes')
    manifest=json.loads((args.manifest or root/'data/input_manifest.json').read_text())
    report_path=root/'reports/inference_runs.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {'official_commit':sha,'runs':{}}
    from run_official import validate_result
    for seq in manifest['sequences']:
        for view in seq['views']:
            rid=f"{seq['id']}__{view['id']}"
            if args.run_id and rid not in args.run_id: continue
            target=root/'outputs'/rid
            payload=target/'inference_output.pt'
            if payload.exists() and (target/'summary.json').exists():
                check=validate_result(payload,seq['frames'])
                report['runs'].setdefault(rid,{}).update(status='complete',validation=check,checked_utc=stamp())
                save(report_path,report); print('REUSED',rid,flush=True); continue
            if target.exists() and any(target.iterdir()): raise RuntimeError(f'Inspect partial output before retry: {target}')
            gpu=select_gpu(root)
            while gpu is None:
                print('WAITING for >=22 GiB available GPU memory and >=32 GiB RAM',flush=True)
                time.sleep(30)
                gpu=select_gpu(root)
            command=[sys.executable,str(root/'scripts/gpu_entry.py'),str(upstream/'infer_marker_video.py'),
                '--config',str(root/'configs/runtime.toml'),'--checkpoint',str(root/'weights/EgoFound3R.pt'),
                '--video-path',view['video'],'--output-dir',str(target),'--device','cuda:0',
                '--upsample-mano','--no-overlay-frame-images','--high-fps','10','--global-stride','5',
                '--chunk-seconds','2','--overlap-seconds','0.5']
            if view.get('K_file'):
                command += ['--root-k-source','external','--root-k-file',view['K_file']]
            entry={'status':'running','started_utc':stamp(),'command':command,'input_frames':seq['frames'],
                'input_fps':seq['fps'],'camera_id':view['camera_id'],'inputs':['RGB']+(['camera intrinsics'] if view.get('K_file') else []),
                'ground_truth_used':False,'gpu':gpu,'allocator_cap_mib':16384,'mps_active_thread_percentage':20,'queue_pid':os.getpid()}
            report['runs'][rid]=entry; save(report_path,report)
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],CUDA_MPS_ACTIVE_THREAD_PERCENTAGE='20',
                OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',EGO_GPU_CAP_MIB='16384',PYTHONUNBUFFERED='1')
            print('START',rid,gpu,flush=True)
            with (root/'logs'/f'{rid}.log').open('a') as log:
                log.write(json.dumps({'started_utc':entry['started_utc'],'command':command,'gpu':gpu})+'\n');log.flush()
                p=subprocess.Popen(command,cwd=upstream,env=env,stdout=log,stderr=subprocess.STDOUT)
                entry['pid']=p.pid;save(report_path,report)
                rc=p.wait()
            entry.update(finished_utc=stamp(),exit_code=rc)
            if rc:
                entry['status']='failed';save(report_path,report);raise RuntimeError(f'Inference failed: {rid}, exit {rc}')
            entry.update(status='complete',validation=validate_result(payload,seq['frames']))
            save(report_path,report);print('COMPLETE',rid,flush=True)

if __name__=='__main__':main()
