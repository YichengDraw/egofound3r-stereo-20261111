#!/usr/bin/env python3
"""Publish already staged/validated results to the existing GitHub project."""
from pathlib import Path
from datetime import datetime,timezone
import os,subprocess,json,hashlib,shutil
BASE=Path('/Users/yiche/26fall/Ego WM/egofound3r_reproduction_20261111');REPO=BASE/'github_pages'
NAME='YichengDraw/egofound3r-stereo-20261111';TAG='reproduction-20261111';GH='/Users/yiche/.local/bin/gh'
E=os.environ.copy()
for k in list(E):
    if k.lower() in ['http_proxy','https_proxy','all_proxy']:E.pop(k)
E['GODEBUG']='http2client=0'
GIT=['git','-c','http.proxy=','-c','http.https://github.com.proxy=','-c','http.version=HTTP/1.1']
def run(cmd,**kw):
    print('RUN',cmd[:4],flush=True);return subprocess.run(cmd,cwd=REPO,env=E,check=True,**kw)
assert json.loads((BASE/'staging_receipt.json').read_text())['status']=='staged_and_archive_hashes_verified'
assert json.loads((REPO/'reports/site_asset_validation.json').read_text())['status']=='passed'
for name in ['stage_publication.py','publish_verified_results.py']:
    src=BASE/name
    if src.exists():shutil.copy2(src,REPO/'scripts'/name)
selected=['docs','README.md','PROTOCOL.md','status.md','configs/runtime.toml']
selected += [str(p.relative_to(REPO)) for p in (REPO/'scripts').glob('*.py')]
selected += [str(p.relative_to(REPO)) for p in (REPO/'scripts').glob('*.md')]
reports=['data_validation.json','expanded_data_validation.json','expanded_selection.json','weight_transfer_verified_20261111.json','weight_access.json','inference_runs.json','runtime_environment.json','frozen_anatomical_mapping.json','anatomical_slot_audit.json','anatomical_slot_audit.jpg','site_build.json','site_asset_validation.json','suite_metrics.json','metrics.csv','CHECKSUMS.sha256','raw_artifacts.json','resource_snapshots.jsonl','contract_test.txt','publication_receipt.json']
reports += [p.name for p in (REPO/'reports').glob('*_metrics.json')]
selected += ['reports/'+name for name in sorted(set(reports))]
run(['git','add','--',*selected])
names=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).splitlines()
assert names and not any(x.startswith(('scripts/data/','weights/','outputs/','upstream/','evaluation_ref/')) for x in names)
for f in names:
    p=REPO/f
    if p.is_file():assert p.stat().st_size < 100*1024**2,(f,p.stat().st_size)
run(['git','-c','user.name=YichengDraw','-c','user.email=202728613+YichengDraw@users.noreply.github.com','commit','-m','Publish real EgoFound3R results: 13 RGB streams, independent scoring and interactive 3D'])
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
(BASE/'publication_progress.json').write_text(json.dumps({'phase':'committed','commit':sha},indent=2)+'\n')
run(GIT+['push','origin','main'])
run([GH,'repo','edit',NAME,'--description','EgoFound3R official reproduction: 13 calibrated RGB streams, independent evaluation, synchronized RGB and interactive hand/scene/contact 3D.'])
notes='''# Verified official inference outputs

13/13 official runs; 11 sequences; 2,080 RGB frames. Official clean source commit: 264f4df9fd42536aae4dba982d18272cc3334077.

Each archive contains the original inference_output.pt, official summary, independent metrics and the original run log. It contains no model checkpoint or GT annotations. Archive and prediction SHA256 values are in reports/raw_artifacts.json and CHECKSUMS.sha256.

Dataset-specific primary results (official valid, finite, excluding official missing-fill):

| Dataset | Valid / GT hand-frames | Raw MPJPE mm |
|---|---:|---:|
| DexYCB two independent RGB cameras | 1 / 292 | 360.99 (one valid hand-frame only) |
| HOT3D-Aria single RGB | 1680 / 1780 | 27.97 |
| HO3D-v3 single RGB | 430 / 868 | 86.47 |

External known K constrains the official hand root/projection path. Scene K remains predicted. This is not joint stereo. HO3D timing is explicitly assumed 10 FPS; no physical motion-rate claim. GT is isolated from inference. Missing predictions remain in all-GT PCK denominators.

Interactive results: https://yichengdraw.github.io/egofound3r-stereo-20261111/
'''
(BASE/'release_notes.md').write_text(notes)
assets=json.loads((REPO/'reports/raw_artifacts.json').read_text())['artifacts']
files=[str(BASE/'raw_results'/(x['run_id']+'.tar.gz')) for x in assets]
run([GH,'release','create',TAG,*files,'--repo',NAME,'--target',sha,'--title','EgoFound3R: 13 official RGB runs and independently verified outputs','--notes-file',str(BASE/'release_notes.md')])
release=json.loads(subprocess.check_output([GH,'api','repos/'+NAME+'/releases/tags/'+TAG],cwd=REPO,env=E,text=True))
remote={x['name']:x for x in release['assets']}
for x in assets:
    y=remote[x['run_id']+'.tar.gz'];assert y['size']==x['archive_bytes'];assert y.get('state')=='uploaded'
    if y.get('digest'):assert y['digest']=='sha256:'+x['archive_sha256'],y['name']
receipt={'phase':'pushed_and_raw_assets_published','commit':sha,'repository':'https://github.com/'+NAME,'release_url':release['html_url'],'release_id':release['id'],'archive_count':len(remote),'asset_sizes_and_digests_verified':True,'finished_utc':datetime.now(timezone.utc).isoformat()}
(BASE/'publication_progress.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
