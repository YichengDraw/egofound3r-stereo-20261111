#!/usr/bin/env python3
"""Wait for the single inference queue, then evaluate and build real artifacts."""
import fcntl,json,subprocess,sys,time,os
from pathlib import Path
r=Path('/nvme-ssd2/zhaoyicheng/egofound3r-reproduction-20261111')
lock=(r/'reports/finalization.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
inputs=json.loads((r/'data/input_manifest_expanded.json').read_text());ids=[s['id']+'__'+v['id'] for s in inputs['sequences'] for v in s['views']]
while True:
    runs=json.loads((r/'reports/inference_runs.json').read_text())['runs'];states={k:runs.get(k,{}).get('status','queued') for k in ids}
    if any(x=='failed' for x in states.values()):raise RuntimeError(states)
    if all(x=='complete' for x in states.values()):break
    print('WAIT',sum(x=='complete' for x in states.values()),'/',len(ids),flush=True);time.sleep(20)
for script in ['evaluate_suite.py','build_complete_site.py','validate_site.py']:
    print('RUN',script,flush=True);subprocess.run([sys.executable,str(r/'scripts'/script),'--root',str(r)],cwd=r,check=True)
print('FINALIZATION_COMPLETE',flush=True)
