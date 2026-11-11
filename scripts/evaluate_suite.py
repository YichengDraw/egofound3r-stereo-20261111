#!/usr/bin/env python3
"""Independent evaluation of all actual official outputs under a frozen mapping.

Primary scores exclude every official fill and retain all annotated failures in
the coverage/PCK denominators. Display scores are a separately named diagnostic.
No calibration, pose alignment or side reassignment is fitted to lower error.
"""
import argparse,copy,csv,json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from official_contract import load_prediction,hand_contract,camera_contract,array,probability,contact_distance,finite_json,sha256
from evaluate_predictions import evaluate,average

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(finite_json(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def summarize(rows):
    c={k:sum(x['primary']['coverage'][k] for x in rows) for k in ['gt_joint_count','evaluated_joint_count','gt_hand_frames','evaluated_hand_frames','filled_annotated_hand_frames','pa_evaluated_hand_frames']}
    c['fraction']=c['evaluated_joint_count']/c['gt_joint_count'] if c['gt_joint_count'] else None
    m={}
    for key in ['mpjpe_raw_mm','mpjpe_root_relative_mm','mpjpe_pa_mm','wrist_position_error_mm']:
        weight_key={'mpjpe_pa_mm':'pa_evaluated_joint_count','mpjpe_root_relative_mm':'root_relative_evaluated_joint_count','wrist_position_error_mm':'evaluated_hand_frames'}.get(key,'evaluated_joint_count')
        weights=[x['primary']['coverage'][weight_key] for x in rows]
        pairs=[(x['primary']['metrics'][key],w) for x,w in zip(rows,weights) if x['primary']['metrics'][key] is not None and w]
        m[key]=sum(v*w for v,w in pairs)/sum(w for v,w in pairs) if pairs else None
    return {'streams':len(rows),'rgb_frames':sum(x['input_frames'] for x in rows),'coverage':c,'metrics':m,
       'pck50_all_gt':sum(x['primary']['pck']['50mm']['correct_joint_count'] for x in rows)/c['gt_joint_count'] if c['gt_joint_count'] else None,
       'display_hand_frames':sum(x['display_including_fill']['coverage']['evaluated_hand_frames'] for x in rows),
       'native_official_fill_hand_frames':sum(sum(x['validity_counts']['official_fill_by_native_slot']) for x in rows)}

def extra_metrics(p,h,gt,seq,view,hmap):
    n=len(h['joints']);K=array(gt['K'],dtype=float)
    if K.shape==(3,3):K=np.repeat(K[None],n,axis=0)
    known=array(p['hand_projection_K_full'],dtype=float).copy();mh,mw=p['image_size_hw'];ih,iw=view['image_size_hw']
    known[:,0]*=iw/mw;known[:,1]*=ih/mh
    maxerr=float(np.max(np.abs(known-K)));assert maxerr<0.01,(seq['id'],maxerr)
    c=camera_contract(p,n);predK=array(p['metric_predictions']['intrinsics']['K_full'],dtype=float).copy();predK[:,0]*=iw/mw;predK[:,1]*=ih/mh
    kv=array(p['metric_predictions']['intrinsics']['K_full_valid']).astype(bool)
    focal=(np.abs(predK[:,[0,1],[0,1]]-K[:,[0,1],[0,1]])/K[:,[0,1],[0,1]])*100
    principal=np.linalg.norm(predK[:,:2,2]-K[:,:2,2],axis=1)
    out={'known_K_projection_max_abs_error_px':maxerr,'scene_intrinsics':{'valid_frames':int(kv.sum()),'mean_focal_relative_error_pct':average(focal,np.broadcast_to(kv[:,None],focal.shape)),'mean_principal_point_error_px':average(principal,kv),
       'note':'Scene intrinsics remain independently predicted. External K is only used by the official hand root/projection path.'}}
    # First-camera rigid gauge only; no fitted Sim(3) or trajectory alignment.
    if seq['dataset']=='DexYCB':G=np.repeat(np.eye(4)[None],n,axis=0)
    elif 'camera_to_world' in gt:G=gt['camera_to_world']
    else:G=None
    if G is not None:
        good=c['valid']&np.isfinite(G).all(axis=(1,2));ids=np.flatnonzero(good)
        if len(ids):
            anchor=int(ids[0]);Pr=np.linalg.inv(c['c2w'][anchor])@c['c2w'];Gr=np.linalg.inv(G[anchor])@G
            dt=np.linalg.norm(Pr[:,:3,3]-Gr[:,:3,3],axis=1)*1000
            E=np.swapaxes(Gr[:,:3,:3],-1,-2)@Pr[:,:3,:3]
            angle=np.degrees(np.arccos(np.clip((np.trace(E,axis1=1,axis2=2)-1)/2,-1,1)))
            out['camera_first_frame_aligned']={'anchor_frame':anchor,'valid_frames':int(good.sum()),'translation_rmse_mm':float(np.sqrt(np.mean(dt[good]**2))),'rotation_mean_deg':average(angle,good),'alignment':'relative to first valid camera; rigid coordinate gauge only, no scale fitting','gt_source':'fixed dataset camera' if seq['dataset']=='DexYCB' else 'timestamp-matched HOT3D camera_to_world'}
    # Reprojection is computed only on valid, not filled, anatomically matched hands.
    P=h['joints'];truth=gt['joints'];gvalid=gt['valid'].astype(bool)
    if gvalid.ndim==2:gvalid=np.broadcast_to(gvalid[...,None],truth.shape[:-1])
    errors=[]
    for slot,side in enumerate(hmap):
        if side<0:continue
        x=np.einsum('tij,tnj->tni',K,P[:,slot]);y=np.einsum('tij,tnj->tni',K,truth[:,side]);good=h['valid'][:,slot,None]&gvalid[:,side]&(P[:,slot,:,2]>0)&(truth[:,side,:,2]>0)
        with np.errstate(divide='ignore',invalid='ignore'):
            uvx=x[...,:2]/x[...,2:];uvy=y[...,:2]/y[...,2:]
        e=np.linalg.norm(uvx-uvy,axis=-1);good &= np.isfinite(e);errors.extend(e[good].tolist())
    out['reprojection']={'mean_error_px':float(np.mean(errors)) if errors else None,'evaluated_joints':len(errors),'image_size_hw':view['image_size_hw']}
    return out

def contact_metrics(p,h,gt,asset,hmap):
    if 'contact' not in gt:return {'available':False,'reason':'No matched vertex-contact labels in this scoring cache'}
    slot=hmap.index(0);ids=asset['source_vertex_ids'];target=gt['contact'][:,ids];dist=gt['contact_distance'][:,ids]
    prob=probability(h['hand'],'vertex_contact');pd=contact_distance(h['hand'])
    if prob is None:return {'available':False,'reason':'No native contact prediction'}
    prob=prob[:,slot];eligible=np.isfinite(target)&np.isin(target,(0,1));valid=eligible&h['valid'][:,slot,None]&np.isfinite(prob)
    pos=target>.5;pred=prob>=.5;tp=int((valid&pos&pred).sum());fp=int((valid&~pos&pred).sum());fn=int((valid&pos&~pred).sum())
    positives=int((eligible&pos).sum());missing_pos=int((eligible&pos&~valid).sum())
    return {'available':True,'space':'195 native MANO vertex IDs; no 778 interpolation scored','threshold':0.5,
        'gt_vertex_frames':int(eligible.sum()),'evaluated_vertex_frames':int(valid.sum()),'tp':tp,'fp':fp,'fn_on_valid':fn,'missing_gt_positives':missing_pos,
        'precision':tp/(tp+fp) if tp+fp else None,'recall_on_valid':tp/(tp+fn) if tp+fn else None,
        'f1_on_valid':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
        'recall_all_gt_positives_including_failures':tp/positives if positives else None,
        'distance_mae_mm':average(np.abs(pd[:,slot]-dist)*1000,valid&np.isfinite(dist)) if pd is not None else None,
        'distance_gt_unit':'meters (HO3D handVertDist as stored)','gt_positive_vertex_frames':positives,
        'label_source':'HO3D-v3 handVertContact and handVertDist, validated finite and 0/1; labels are not recomputed or optimized',
        'definition_caution':'Uses released HO3D binary labels; their geometry-based contact rule need not equal EgoFound3R training contact semantics. Treat as cross-dataset diagnostic.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    manifest=json.loads((r/'data/input_manifest_expanded.json').read_text());evaluation=json.loads((r/'evaluation_ref/expanded_evaluation_manifest.json').read_text());gpaths={v['run_id']:Path(v['gt_path']) for v in evaluation['samples']}
    side_order=['left','right'];audit=json.loads((r/'reports/anatomical_slot_audit.json').read_text())
    mapping_evidence={'native_hand_order':side_order,'joint_order':'OpenPose21: wrist, thumb, index, middle, ring, little; four joints per finger','joint_mapping':list(range(21)),
        'evidence':'Fixed before quantitative suite scoring: visual RGB audit of HOT3D-Aria 1976 at frames 0,74,149; native slot 0 projects onto anatomical left hand, slot 1 onto right. Internal predicted joint-tip vs MANO-tip distance matrix confirms five finger order. No per-frame swapping or best-error selection.',
        'audit_run':audit['audit_run'],'audit_image':'anatomical_slot_audit.jpg','upstream_limitation':'Released inference schema does not explicitly name the two anatomical slots; mapping is an empirical audit of this checkpoint, not an asserted upstream guarantee.'}
    save(r/'reports/frozen_anatomical_mapping.json',mapping_evidence)
    asset=np.load(r/'upstream/egohandmetric_prompt/data/mano_upsampling/mano_195_to_778.npz');rows=[]
    for seq in manifest['sequences']:
      for view in seq['views']:
        rid=seq['id']+'__'+view['id'];path=r/'outputs'/rid/'inference_output.pt';assert path.exists(),rid
        gp=gpaths.get(rid,r/'evaluation_ref'/seq['id']/view['camera_id']/'gt.npz')
        with np.load(gp,allow_pickle=False) as z:gt={k:z[k] for k in z.files}
        p=load_prediction(path);h=hand_contract(p);sides=gt['hand_sides'].tolist();hmap=[sides.index(s) if s in sides else -1 for s in side_order]
        mapping={'pred_to_gt_hand':hmap,'pred_to_gt_joint':list(range(21)),'evidence':mapping_evidence['evidence']}
        primary=evaluate(p,gt,mapping)
        q=dict(p);q['metric_predictions']=dict(p['metric_predictions']);q['metric_predictions']['hand']=dict(h['hand']);q['metric_predictions']['hand']['valid']=h['display_valid'];q['hand_missing_fill_full']=np.zeros_like(h['fill_mask'])
        display=evaluate(q,gt,mapping);display['validity_policy']='SECONDARY diagnostic: official display-valid finite output INCLUDING official fills; not accepted as native valid coverage'
        row={'run_id':rid,'dataset':seq['dataset'],'input_group':seq['input_group'],'input_frames':seq['frames'],'primary':primary,'display_including_fill':display,
          'validity_counts':{'slot_order':side_order,'strict_by_native_slot':h['valid'].sum(0).tolist(),'official_valid_by_native_slot':h['official_valid'].sum(0).tolist(),'official_fill_by_native_slot':h['fill_mask'].sum(0).tolist(),'display_by_native_slot':h['display_valid'].sum(0).tolist(),
             'frames_with_any_strict_hand':int(h['valid'].any(1).sum()),'frames_with_no_strict_hand':int((~h['valid'].any(1)).sum()),
             'network_H_anchor_frames':int(np.isin(array(p['source_frame_indices']),array(p['high_frame_indices'])).sum()),
             'non_anchor_R_frames':int((~np.isin(array(p['source_frame_indices']),array(p['high_frame_indices']))).sum()),
             'note':'A valid R-frame can be temporal interpolation between network H anchors. This is distinct from official missing-hand fill.'},
          'source':{'inference_sha256':sha256(path),'gt_sha256':sha256(gp),'gt_used_only_for_scoring':True,'gt_path':str(gp),'prediction_path':str(path)},
          'additional_metrics':extra_metrics(p,h,gt,seq,view,hmap),'contact':contact_metrics(p,h,gt,asset,hmap)}
        row['runtime']=json.loads((path.parent/'summary.json').read_text())
        save(r/'reports'/f'{rid}_metrics.json',row);rows.append(row)
        print(rid,primary['coverage'],primary['metrics'],flush=True)
    groups={d:summarize([x for x in rows if x['dataset']==d]) for d in ['DexYCB','HOT3D-Aria','HO3D-v3']}
    summary={'schema_version':'egofound3r.reproduction.suite.v1','completed_utc':datetime.now(timezone.utc).isoformat(),'streams':len(rows),'rgb_frames':sum(x['input_frames'] for x in rows),'datasets':groups,
       'primary_policy':'Official canonical valid AND finite AND NOT missing-fill. Denominator is every valid annotated hand/joint. No-annotation frames are excluded from accuracy only, retained in inference.',
       'pooling':'Report each dataset/input group separately; do not pool monocular supplements into dual-RGB scores. Within each group, average over evaluated joints (wrist over hand frames).',
       'scene_depth_accuracy':'Not scored: no independently verified depth labels in the selected evaluation caches. Scene is actual predicted depth/K lifted into 3D.',
       'timing':'HO3D source FPS unknown: ordered sampled frames supplied at assumed 10 FPS. No physical velocity/acceleration scores reported.',
       'runs':[{k:v for k,v in x.items() if k!='runtime'} for x in rows]}
    save(r/'reports/suite_metrics.json',summary)
    with (r/'reports/metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['run_id','dataset','input_frames','gt_hand_frames','strict_hand_frames','fill_annotated_hand_frames','coverage_pct','raw_mpjpe_mm','root_relative_mpjpe_mm','pa_mpjpe_mm','pck50_all_gt_pct'])
        for x in rows:
            c=x['primary']['coverage'];m=x['primary']['metrics'];w.writerow([x['run_id'],x['dataset'],x['input_frames'],c['gt_hand_frames'],c['evaluated_hand_frames'],c['filled_annotated_hand_frames'],100*c['fraction'],m['mpjpe_raw_mm'],m['mpjpe_root_relative_mm'],m['mpjpe_pa_mm'],100*x['primary']['pck']['50mm']['all_gt_including_failures']])
    print(json.dumps(groups,indent=2))

if __name__=='__main__':main()
