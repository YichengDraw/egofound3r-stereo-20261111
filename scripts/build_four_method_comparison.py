#!/usr/bin/env python3
"""Cached Base16 rescoring in explicitly distinct DexYCB and Aria groups.

No inference or output fitting. Does not modify the earlier three-method report.
No all-four ranking is produced: the four-way common valid set is empty.
"""
from __future__ import annotations
import argparse, copy, csv, json
from pathlib import Path
import numpy as np
from build_cross_method_comparison import BASE16, SIDES, average, metric_arrays, read_geometry, sha

METHODS=['hawor','egofound3r','hggt','egohsi3r']
LABELS={'hawor':'HaWoR','egofound3r':'EgoFound3R','hggt':'HGGT','egohsi3r':'EgoHSI3R'}
FIELDS=['mpjpe16_mm','rr_mpjpe16_mm','pa_mpjpe16_mm','wrist_error_mm','rr_acceleration16_error_m_s2']
GROUPS=[('dexycb','DexYCB · 箱子与香蕉',['dexycb_box','dexycb_banana'],['hawor','hggt','egohsi3r']),
        ('aria','Aria · 1976与3209',['aria_1976','aria_3209'],['hawor','egofound3r','hggt'])]

def missing_row(method,gt_count):
    return {'label':LABELS[method], 'available':False,'included_in_common_comparison':False,
            'reason':'此片段没有已有的 EgoHSI3R 默认预测，未重跑；不是检测失败。',
            'gt_hand_frames':gt_count,'gt_joint_count':gt_count*16,
            **{k:None for k in ['strict_hand_frames','coverage_fraction','excluded_fill_hand_frames','displayable_hand_frames',
                               'common_hand_frames','common_acceleration_triplets','pck50_all_gt','correct_joints_50mm',*FIELDS]}}

def geometry_arrays(path):
    meta,arrays=read_geometry(path)
    if 'display_valid' in meta:
        spec=meta['display_valid'];arrays['display_valid']=np.fromfile(path.parent/spec['url'],dtype='u1').reshape(spec['shape'])
    return meta,arrays

def summarize_group(samples,gid,label,scored_methods):
    result={'id':gid,'label':label,'sample_ids':[s['id'] for s in samples], 'scored_methods':scored_methods,
            'rgb_frames':sum(s['rgb_frames'] for s in samples),'gt_hand_frames':sum(s['gt_hand_frames'] for s in samples),
            'common_hand_frames':sum(s['common_hand_frames'] for s in samples),
            'common_acceleration_triplets':sum(s['common_acceleration_triplets'] for s in samples),
            'all_four_common_hand_frames':sum(s['all_four_common_hand_frames'] for s in samples),'methods':{}}
    for method in METHODS:
        rows=[s['methods'][method] for s in samples]
        if not any(r['available'] for r in rows):
            result['methods'][method]=missing_row(method,result['gt_hand_frames']);continue
        assert all(r['available'] for r in rows), 'Do not aggregate incomplete method coverage as full group coverage'
        counts=['gt_hand_frames','gt_joint_count','strict_hand_frames','excluded_fill_hand_frames','displayable_hand_frames',
                'common_hand_frames','common_acceleration_triplets','correct_joints_50mm']
        row={'label':LABELS[method],'available':True,'included_in_common_comparison':method in scored_methods,
             **{k:sum(r[k] for r in rows) for k in counts}}
        row['coverage_fraction']=row['strict_hand_frames']/row['gt_hand_frames']
        row['pck50_all_gt']=row['correct_joints_50mm']/row['gt_joint_count']
        for key in FIELDS+(['adapted_mpjpe16_mm','adapted_wrist_error_mm'] if method=='hggt' else []):
            count='common_acceleration_triplets' if key=='rr_acceleration16_error_m_s2' else 'common_hand_frames'
            used=[r for r in rows if r[key] is not None and r[count]>0]
            row[key]=sum(r[key]*r[count] for r in used)/sum(r[count] for r in used) if used else None
        if method=='hggt':
            row['adapted_correct_joints_50mm']=sum(r['adapted_correct_joints_50mm'] for r in rows)
            row['adapted_pck50_all_gt']=row['adapted_correct_joints_50mm']/row['gt_joint_count']
        if method not in scored_methods:
            row['reason']='已有预测运行，但该组对应已标注手没有严格有效预测；覆盖率及全GT PCK为0，误差不可评价。'
        result['methods'][method]=row
    return result

def build(root,legacy):
    docs=root/'docs';ldocs=legacy/'docs'
    manifest=json.loads((docs/'manifest.json').read_text())
    old=json.loads((docs/'cross-method-comparison.json').read_text())
    hsi_materials=json.loads((docs/'egohsi3r-materials.json').read_text())
    matching=json.loads((docs/'legacy-hggt.json').read_text())
    protocol=copy.deepcopy(old['protocol'])
    protocol.update({'coordinate_frame':'Original RGB camera axes (OpenCV xyz), meters; errors in mm. Model-specific intrinsic estimation/calibration is retained.',
                     'hand_mapping':'Fixed left/right anatomical slots. EgoHSI3R source right,left is permuted once to left,right. No GT-error matching.',
                     'common_mask':'Intersection of strict method masks for each explicitly named group; all its scored methods use the same hand/frame set.',
                     'strict_validity':'GT valid AND finite Base16 AND model valid AND no explicit missing-hand fill. EgoHSI3R uses original detector hand_mask, without generating missing hands.',
                     'coverage':'Method strict available hand-frames / every annotated hand-frame in its group. An absent experiment is null, not a failure.',
                     'group_policy':'DexYCB compares HaWoR/HGGT/EgoHSI3R on 146 right-hand frames. Aria compares HaWoR/EgoFound3R/HGGT on 565 hand-frames. No pooled cross-group ranking.',
                     'four_way_comparison':'No all-four common valid hand-frame exists. EgoFound3R failed on the matched Dex camera A, and EgoHSI3R was not run on these Aria samples.',
                     'limitations':['Four fixed development clips, not an official benchmark or a guaranteed training-disjoint test.',
                                    'Only DexYCB camera A 836212060125 and Aria RGB 214-1 are matched.',
                                    'Same source recording/camera/frame IDs and GT; different video encodings, model detectors, crops, calibration and temporal processing are retained.',
                                    'EgoHSI3R default RGB predicts its own K; no GT calibration, GT boxes, later trained ablation, focal or reset variant is substituted.',
                                    'Base16 excludes all five fingertip definitions. Original 21-point tables and the previous three-method report remain unchanged.',
                                    'Lower scores across different groups cannot be ranked against one another.']})
    result={'schema':'egofound3r.four_method_grouped_base16.v1','methods':METHODS,'scope':'Four cached models in two different comparable groups; no model rerun.',
            'protocol':protocol,'sources':{'previous_three_method_report':'cross-method-comparison.json','previous_report_sha256':sha(docs/'cross-method-comparison.json'),
                                         'egohsi3r_materials':'egohsi3r-materials.json','egohsi3r_materials_sha256':sha(docs/'egohsi3r-materials.json'),
                                         'model_inference_rerun':False,'old_metric_files_modified':False},'samples':[],'groups':[],
            'all_four_common_hand_frames':0,'aggregate':None,
            'aggregate_unavailable_reason':'没有四方法共同有效手帧；DexYCB与Aria分组报告，不能拼成四方法总排名。'}
    for gid,label,sids,scored_methods in GROUPS:
        for sid in sids:
            seq=next(s for s in manifest['sequences'] if s['id']==sid);match=next(s for s in matching['samples'] if s['id']==sid)
            view=match['view'];base=docs/'data'/sid/view;ep=base/'prediction/geometry.json';gp=base/'reference/geometry.json'
            em,ea=geometry_arrays(ep);gm,ga=geometry_arrays(gp);oldpath=ldocs/f'interactive3d/{sid}.json';published=json.loads(oldpath.read_text())
            assert sha(oldpath)==match['geometry_sha256']
            ids=np.asarray(seq['frame_ids']);times=np.asarray(gm['timestamps_s'],float);times-=times[0];n=len(ids)
            assert em['frame_indices']==gm['frame_indices']==seq['frame_ids']==published['frame_ids']
            assert np.all(np.diff(ids)==1) and np.allclose(times,published['timestamps_relative_s'],atol=1e-9,rtol=0)
            assert em['coordinate_space']==gm['coordinate_space']=='camera' and em['units']==gm['units']=='m'
            truth=np.full((n,2,21,3),np.nan);gvalid=np.zeros((n,2,21),bool)
            for h,side in enumerate(gm['hand_order']):
                truth[:,SIDES.index(side)]=ga['joints'][:,h];gvalid[:,SIDES.index(side)]=ga['joint_valid'][:,h]
            eligible=gvalid[:,:,BASE16].all(-1)
            assert np.array_equal(gvalid.any(-1),gvalid.all(-1)) and np.array_equal(eligible,published['methods']['gt']['valid'])
            preds={'egofound3r':{'joints':ea['joints'].astype(float),'valid':ea['valid'].astype(bool),
                               'filled':ea['fill_mask'].astype(bool),'display':ea['display_valid'].astype(bool)}}
            source_hashes={}
            for name,meta,path in [('egofound3r',em,ep),('reference',gm,gp)]:
                source_hashes[str(path.relative_to(docs))]=sha(path)
                for field in ['joints','valid','display_valid','fill_mask','joint_valid']:
                    if field in meta:
                        p=path.parent/meta[field]['url'];source_hashes[str(p.relative_to(docs))]=sha(p)
            source_hashes[f'hggt_hawor/interactive3d/{sid}.json']=sha(oldpath)
            for method in ['hawor','hggt']:
                p=ldocs/f'downloads/predictions/{sid}_{method}.npz';z=np.load(p,allow_pickle=False)
                assert sha(p)==published['provenance']['source_sha256'][method]
                assert z['side_order'].tolist()==SIDES and np.array_equal(z['frame_ids'],ids)
                assert np.allclose(z['timestamps_s']-z['timestamps_s'][0],times,atol=1e-9,rtol=0)
                field='joints_cam_native' if method=='hggt' else 'joints_cam'
                fill=np.asarray(published['methods']['hawor']['infilled'],bool) if method=='hawor' else np.zeros((n,2),bool)
                display=np.asarray(published['methods']['hggt_native' if method=='hggt' else 'hawor']['valid'],bool)
                preds[method]={'joints':z[field].astype(float),'valid':display,'filled':fill,'display':display}
                if method=='hggt':preds[method].update(adapted=z['joints_cam'].astype(float),adapter_valid=z['adapter_valid'].astype(bool))
                source_hashes[f'hggt_hawor/downloads/predictions/{sid}_{method}.npz']=sha(p)
            if gid=='dexycb':
                hp=base/'egohsi3r/geometry.json';hm,ha=geometry_arrays(hp)
                assert hm['hand_order']==SIDES and hm['frame_indices']==ids.tolist()
                assert np.array_equal(hm['timestamps_s'],times)
                evidence=next(x for x in hsi_materials['samples'] if x['id']==sid)
                assert evidence['gt_joint_arrays_exactly_equal'] and evidence['gt_masks_exactly_equal']
                assert evidence['reference_joints_sha256']==sha(gp.parent/gm['joints']['url'])
                preds['egohsi3r']={'joints':ha['joints'].astype(float),'valid':ha['valid'].astype(bool),'display':ha['display_valid'].astype(bool),'filled':np.zeros((n,2),bool)}
                source_hashes[str(hp.relative_to(docs))]=sha(hp)
                for field in ['joints','valid','display_valid','joint_valid']:
                    p=hp.parent/hm[field]['url'];source_hashes[str(p.relative_to(docs))]=sha(p)
            for p in preds.values():
                finite=np.isfinite(p['joints'][:,:,BASE16]).all((-1,-2))
                p['strict']=eligible&p['valid']&finite&~p['filled'];p['display']=eligible&p['display']&finite
                p['metrics']=metric_arrays(p['joints'],truth,times)
            common=np.logical_and.reduce([preds[k]['strict'] for k in scored_methods])
            triplets=np.zeros_like(common);triplets[1:-1]=common[:-2]&common[1:-1]&common[2:]
            four=np.logical_and.reduce([preds[k]['strict'] for k in METHODS]) if all(k in preds for k in METHODS) else np.zeros_like(common)
            assert not four.any()
            hg=preds['hggt'];adapted=metric_arrays(hg['adapted'],truth,times)
            assert (hg['adapter_valid'][hg['strict']]).all()
            assert np.isfinite(hg['adapted'][:,:,BASE16][hg['strict']]).all()
            item={'id':sid,'group_id':gid,'label':seq['label'],'dataset':seq['dataset'],'camera':match['camera_id'],'view':view,
                  'rgb_frames':n,'gt_hand_frames':int(eligible.sum()),'common_hand_frames':int(common.sum()),
                  'common_acceleration_triplets':int(triplets.sum()),'all_four_common_hand_frames':int(four.sum()),
                  'scored_methods':scored_methods,'common_pairs':[{'source_frame_id':int(ids[f]),'side':SIDES[h]} for f,h in np.argwhere(common)],
                  'source_sha256':source_hashes,'methods':{}}
            for method in METHODS:
                if method not in preds:
                    item['methods'][method]=missing_row(method,int(eligible.sum()));continue
                p=preds[method];strict=p['strict'];included=method in scored_methods
                mask=common if included else np.zeros_like(common);tm=triplets if included else np.zeros_like(triplets)
                if not included: assert not strict.any(), 'Unscored existing output needs an explicit comparison policy'
                row={'label':LABELS[method],'available':True,'included_in_common_comparison':included,
                     'strict_hand_frames':int(strict.sum()),'gt_hand_frames':int(eligible.sum()),'gt_joint_count':int(eligible.sum()*16),
                     'coverage_fraction':float(strict.sum()/eligible.sum()),'excluded_fill_hand_frames':int((p['filled']&eligible).sum()),
                     'displayable_hand_frames':int(p['display'].sum()),'common_hand_frames':int(mask.sum()),'common_acceleration_triplets':int(tm.sum()),
                     'correct_joints_50mm':int(p['metrics']['correct50'][strict].sum()),
                     'pck50_all_gt':float(p['metrics']['correct50'][strict].sum()/(eligible.sum()*16))}
                for field in FIELDS:row[field]=average(p['metrics'][field],tm if field=='rr_acceleration16_error_m_s2' else mask)
                if not included:row['reason']='已有运行但已标注手严格有效预测为0；仅保留覆盖率和全GT PCK失败计数。'
                if method=='hggt':
                    row.update({'adapted_mpjpe16_mm':average(adapted['mpjpe16_mm'],mask),
                                'adapted_wrist_error_mm':average(adapted['wrist_error_mm'],mask),
                                'adapted_correct_joints_50mm':int(adapted['correct50'][strict].sum()),
                                'adapted_pck50_all_gt':float(adapted['correct50'][strict].sum()/(eligible.sum()*16))})
                    assert np.max(np.abs(p['metrics']['rr_mpjpe16_mm'][mask]-adapted['rr_mpjpe16_mm'][mask]))<.001
                item['methods'][method]=row
            result['samples'].append(item)
        result['groups'].append(summarize_group([x for x in result['samples'] if x['group_id']==gid],gid,label,scored_methods))
    assert [(g['gt_hand_frames'],g['common_hand_frames'],g['common_acceleration_triplets']) for g in result['groups']]==[(146,146,142),(600,565,555)]
    return result

def write(root,result):
    path=root/'docs/four-method-comparison.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    rows=[]
    for kind,scopes in [('group',result['groups']),('sample',result['samples'])]:
        for scope in scopes:
            for method,row in scope['methods'].items():rows.append({'scope_kind':kind,'scope':scope['id'],'method':method,'scored_methods':','.join(scope['scored_methods']),**row})
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.with_suffix('.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--legacy-root',required=True,type=Path)
    args=ap.parse_args();result=build(args.root,args.legacy_root);write(args.root,result)
    print(json.dumps(result['groups'],ensure_ascii=False,indent=2))
