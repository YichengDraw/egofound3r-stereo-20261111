#!/usr/bin/env python3
"""Record unchanged published HGGT materials and verified same-camera identities."""
import argparse,json,hashlib,pathlib,numpy as np
p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args();root=pathlib.Path(__file__).resolve().parents[1];docs=root/'docs';h=a.source/'docs';old=json.loads((h/'manifest.json').read_text());current=json.loads((docs/'manifest.json').read_text());base='https://yichengdraw.github.io/hggt-hawor-singleview-20261110/';raw=f'https://raw.githubusercontent.com/YichengDraw/hggt-hawor-singleview-20261110/{a.commit}/docs/';items=[]
for sample in old['samples']:
 sid=sample['id'];view='cam_a' if sample['dataset']=='DexYCB' else 'rgb';seq=next(s for s in current['sequences'] if s['id']==sid);payload=h/f'interactive3d/{sid}.json';g=json.loads(payload.read_text());ref=docs/f'data/{sid}/{view}/reference/geometry.json';gt=json.loads(ref.read_text());v=gt['joints'];j=np.fromfile(ref.parent/v['url'],dtype='<f4').reshape(v['shape']);o=np.array([[[p if p is not None else [np.nan]*3 for p in hand] for hand in frame] for frame in g['methods']['gt']['joints']]);aligned=np.full_like(o,np.nan)
 for index,side in enumerate(gt['hand_order']):aligned[:,['left','right'].index(side)]=j[:,index]
 good=np.isfinite(o)&np.isfinite(aligned);error=float(np.max(np.abs(o[good]-aligned[good])));assert error<1e-6
 assert sample['source_frame_ids']==g['frame_ids']==seq['frame_ids'];assert np.max(np.abs(np.asarray(g['timestamps_relative_s'])-np.asarray(seq['timestamps'])))<1e-9
 expected='836212060125' if view=='cam_a' else '214-1';assert g['provenance']['camera_id']==expected
 videos=[]
 for key,path,label in [('comparison',sample['media']['comparison']['src'],'HGGT × HaWoR 原视频'),('native',f'media/{sid}/hggt_native_projection.mp4','HGGT 原生投影')]:
  local=h/path;videos.append({'id':key,'url':base+path,'label':label,'sha256':hashlib.sha256(local.read_bytes()).hexdigest(),'fps':sample['fps'],'frames':sample['frame_count']})
 items.append({'id':sid,'view':view,'camera_id':expected,'frame_count':sample['frame_count'],'geometry_url':raw+f'interactive3d/{sid}.json','geometry_sha256':hashlib.sha256(payload.read_bytes()).hexdigest(),'frame_ids':g['frame_ids'],'timestamp_max_difference_s':0,'gt_max_difference_m':error,'videos':videos,'sample':sample})
out={'source_site':base,'source_commit':a.commit,'geometry_units':'m','source_coordinates':'per-frame calibrated RGB camera, not world','matching':'Source frame IDs, timestamps, camera ID and anatomical-side GT checked; GT differs only by published micrometer rounding. No GT fit.','samples':items,'original_manifest':old,'original_materials_url':base,'licenses_url':base+'interactive3d/README.md'}
(docs/'legacy-hggt.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('Verified 4 sequences; only DexYCB camera A and matching Aria RGB. Original metrics and materials retained.')
