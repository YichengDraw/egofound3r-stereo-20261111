#!/usr/bin/env python3
"""Stage the validated server artifacts into the existing authorized Mac repo."""
from pathlib import Path
import os,subprocess,json,sys
BASE=Path('/Users/yiche/26fall/Ego WM/egofound3r_reproduction_20261111')
REPO=BASE/'github_pages'
REMOTE='/nvme-ssd2/zhaoyicheng/egofound3r-reproduction-20261111'
E=os.environ.copy()
for k in list(E):
    if k.lower() in ['http_proxy','https_proxy','all_proxy']:E.pop(k)
E['GODEBUG']='http2client=0'
GIT=['git','-c','http.proxy=','-c','http.https://github.com.proxy=','-c','http.version=HTTP/1.1']
def run(cmd,**kw):
    print('RUN',cmd[0],cmd[1:4],flush=True);return subprocess.run(cmd,cwd=REPO,env=E,check=True,**kw)
status=subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True)
assert not any(line[:2]!='??' for line in status.splitlines()),'Tracked local changes require review before staging'
proof=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','l40s','cat '+REMOTE+'/reports/site_asset_validation.json'],text=True));assert proof['status']=='passed' and proof['prediction_geometry_exports']==26
run(GIT+['fetch','origin','main']);run(GIT+['merge','--ff-only','origin/main'])
run(['rsync','-a','l40s:'+REMOTE+'/docs/',str(REPO/'docs')+'/'])
run(['rsync','-a','--include=*.py','--include=*.md','--exclude=*','l40s:'+REMOTE+'/scripts/',str(REPO/'scripts')+'/'])
run(['rsync','-a','--include=*/','--include=*.json','--include=*.jsonl','--include=*.csv','--include=*.sha256','--include=*.txt','--include=*.jpg','--exclude=*','l40s:'+REMOTE+'/reports/',str(REPO/'reports')+'/'])
for name in ['README.md','PROTOCOL.md','status.md']:
    run(['scp','-q','l40s:'+REMOTE+'/'+name,str(REPO/name)])
(REPO/'configs').mkdir(exist_ok=True);run(['scp','-q','l40s:'+REMOTE+'/configs/runtime.toml',str(REPO/'configs/runtime.toml')])
(BASE/'raw_results').mkdir(exist_ok=True);run(['rsync','-a','l40s:'+REMOTE+'/artifacts/raw/',str(BASE/'raw_results')+'/'])
meta=json.loads((REPO/'reports/raw_artifacts.json').read_text())
import hashlib
for x in meta['artifacts']:
    p=BASE/'raw_results'/(x['run_id']+'.tar.gz');assert p.stat().st_size==x['archive_bytes'];h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==x['archive_sha256'],p
assert all((REPO/'docs/vendor'/n).is_file() for n in ['three.module.js','OrbitControls.js','LICENSE.three.txt'])
large=[(str(p.relative_to(REPO)),p.stat().st_size) for p in (REPO/'docs').rglob('*') if p.is_file() and p.stat().st_size>99*1024**2];assert not large,large
receipt={'status':'staged_and_archive_hashes_verified','sequences':11,'predictions':13,'evaluations':13,'prediction_geometry_exports':26,'GTreferences':13,'raw_archives':13,'largest_docs_file':max((p.stat().st_size,str(p.relative_to(REPO))) for p in (REPO/'docs').rglob('*') if p.is_file())}
(BASE/'staging_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
run(['git','diff','--stat'])
