#!/usr/bin/env python3
"""Score cached predictions only. Never runs inference or overwrites old metrics.

Base16 uses the wrist and 3 non-tip joints per finger. All methods are evaluated
on exactly the same anatomical sides/source frames and held-out camera-space GT.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

BASE16 = [i for i in range(21) if i not in (4,8,12,16,20)]
SIDES = ['left','right']
METHODS = ['hawor','egofound3r','hggt']
LABELS = {'hawor':'HaWoR','egofound3r':'EgoFound3R','hggt':'HGGT'}
SAMPLES = ['dexycb_box','dexycb_banana','aria_1976','aria_3209']
EGO_COMMIT = 'b2ffc2c894d5132958e8670a15b1dacefb6446c8'
LEGACY_COMMIT = 'b5da99d28417351a1d1814ffceddcb7a882c326c'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def average(x,mask):
    y=np.asarray(x)[mask]
    assert np.isfinite(y).all()
    return float(y.mean()) if y.size else None

def procrustes(pred,target):
    x,y=pred-pred.mean(0),target-target.mean(0)
    if np.linalg.matrix_rank(x,tol=1e-8)<2 or np.linalg.matrix_rank(y,tol=1e-8)<2: return None
    u,s,vt=np.linalg.svd(x.T@y);sign=np.ones(3);sign[-1]=1 if np.linalg.det(u@vt)>=0 else -1
    scale=(s*sign).sum()/np.square(x).sum()
    if scale<=0 or not np.isfinite(scale): return None
    return scale*x@(u@np.diag(sign)@vt)+target.mean(0)

def metric_arrays(joints,truth,times):
    pred=joints[:,:,BASE16].astype(np.float64);target=truth[:,:,BASE16].astype(np.float64)
    delta=np.linalg.norm(pred-target,axis=-1)*1000
    rel,rel_gt=pred-pred[:,:,:1],target-target[:,:,:1]
    pa=np.full(pred.shape[:2],np.nan)
    finite=np.isfinite(pred).all((-1,-2))&np.isfinite(target).all((-1,-2))
    for f,h in np.argwhere(finite):
        fit=procrustes(pred[f,h],target[f,h])
        if fit is not None: pa[f,h]=np.linalg.norm(fit-target[f,h],axis=-1).mean()*1000
    dt=np.diff(times);assert (dt>0).all()
    pv=np.diff(rel,axis=0)/dt[:,None,None,None];gv=np.diff(rel_gt,axis=0)/dt[:,None,None,None]
    acc=np.full(pred.shape[:2],np.nan)
    acc[1:-1]=np.linalg.norm(2*(np.diff(pv,axis=0)-np.diff(gv,axis=0))/(dt[:-1]+dt[1:])[:,None,None,None],axis=-1).mean(-1)
    return {'mpjpe16_mm':delta.mean(-1),'rr_mpjpe16_mm':np.linalg.norm(rel-rel_gt,axis=-1).mean(-1)*1000,
            'pa_mpjpe16_mm':pa,'wrist_error_mm':delta[:,:,0],'rr_acceleration16_error_m_s2':acc,'correct50':(delta<=50).sum(-1)}

def read_geometry(path):
    data=json.loads(path.read_text());arrays={}
    for key in ['joints','valid','official_valid','fill_mask','joint_valid']:
        if key in data:
            spec=data[key];arrays[key]=np.fromfile(path.parent/spec['url'],dtype={'float32':'<f4','uint8':'u1'}[spec['dtype']]).reshape(spec['shape'])
    return data,arrays

def build(root,legacy):
    docs=root/'docs';source=legacy/'docs';manifest=json.loads((docs/'manifest.json').read_text());matching=json.loads((docs/'legacy-hggt.json').read_text());oldmetrics=json.loads((source/'reports/metrics.json').read_text())
    result={'schema':'egofound3r.cross_method_base16.v1','scope':'Four fixed matched RGB clips; cached outputs only; no model rerun.',
      'protocol':{'joints':BASE16,'joint_count':16,'joint_definition':'Wrist plus MCP/PIP/DIP of thumb,index,middle,ring,little; excludes all five fingertip slots to avoid incompatible native tip markers.',
        'coordinate_frame':'Per-frame calibrated RGB camera, OpenCV xyz, meters; errors in mm. No world-space comparison.',
        'hand_mapping':'Fixed audited Ego slot0=left/slot1=right; explicit NPZ side_order and GT hand_order. No best-error matching.',
        'strict_validity':'All 16 points finite, method valid, GT valid, and no explicit missing-hand fill. Ego excludes fill_mask; HaWoR excludes legacy infilled mask; HGGT uses its valid flag.',
        'common_mask':'Intersection of all three strict method masks and all 16 annotated GT joints. Same hand/frame set for every error column.',
        'coverage':'Strict available hand-frames / every annotated hand-frame in the four matched clips. Missing predictions remain in this denominator.',
        'pck50':'Correct camera-space Base16 joints <=50 mm / all annotated Base16 joints; missing/filled predictions count as failures. This is NOT the common-error subset.',
        'pa':'One proper-rotation positive-scale similarity fit per hand/frame on exactly the same 16 points. No alignment for raw error/PCK.',
        'acceleration':'Wrist-relative Base16 acceleration error with shared actual GT timestamps; only three consecutive source frames valid for all methods, never across clip boundaries.',
        'hggt_absolute':'HGGT native camera output is primary. Existing calibration-only translation adapter is reported separately; no GT fit. Root-relative/PA use native output.',
        'aggregation':'Hand-frame weighted errors, triplet-weighted acceleration, joint-weighted PCK, summed coverage counts. Not a mean of clip means.',
        'limitations':['Development examples, not an official benchmark or a guaranteed training-disjoint test.',
          'Only DexYCB camera A 836212060125 and Aria RGB 214-1 are matched; camera B and the other seven Ego clips are excluded.',
          'Original model-specific detection/cropping/calibration approximations and temporal pipelines are retained; equal source frames do not imply identical model inputs.',
          'Base16 is a new common evaluation protocol, not the old native/harmonized 21-joint metric. Original tables, predictions and GT remain unchanged.',
          'The four-clip common error subset contains no DexYCB hand-frame because Ego has no strict prediction on either matched DexYCB camera A clip. Coverage and all-GT PCK still include those failures.',
          'HaWoR original coverage includes temporal infill; this strict table excludes it consistently with Ego. Both excluded fill and displayable counts are provided.']},
      'sources':{'egofound_commit':EGO_COMMIT,'hggt_hawor_commit':LEGACY_COMMIT,'model_inference_rerun':False,'old_metric_files_modified':False},'samples':[]}
    pooled={k:{'gt':0,'strict':0,'filled':0,'display':0,'correct50':0,'adapted_correct50':0,'values':{},'adapted_values':{},'common':0,'triplets':0} for k in METHODS}
    for sid in SAMPLES:
        seq=next(x for x in manifest['sequences'] if x['id']==sid);match=next(x for x in matching['samples'] if x['id']==sid);view=match['view'];base=docs/'data'/sid/view
        eg,ea=read_geometry(base/'prediction/geometry.json');gt,ga=read_geometry(base/'reference/geometry.json');old=json.loads((source/f'interactive3d/{sid}.json').read_text())
        assert eg['coordinate_space']==gt['coordinate_space']=='camera' and eg['units']==gt['units']=='m'
        assert eg['hand_order']=='left,right' and eg['joint_root_index']==0
        assert eg['frame_indices']==seq['frame_ids']==old['frame_ids']==match['frame_ids']
        frames=len(seq['frame_ids']);truth=np.full((frames,2,21,3),np.nan);gtjoint=np.zeros((frames,2,21),bool)
        for h,side in enumerate(gt['hand_order']):truth[:,SIDES.index(side)]=ga['joints'][:,h];gtjoint[:,SIDES.index(side)]=ga['joint_valid'][:,h].astype(bool)
        eligible=gtjoint[:,:,BASE16].all(-1);assert np.array_equal(gtjoint.any(-1),gtjoint.all(-1))
        assert np.array_equal(eligible,np.asarray(old['methods']['gt']['valid'],bool))
        oldtruth=np.array([[[p if p is not None else [np.nan]*3 for p in hand] for hand in frame] for frame in old['methods']['gt']['joints']]);good=np.isfinite(oldtruth)&np.isfinite(truth)
        assert np.max(np.abs(oldtruth[good]-truth[good]))<1e-6
        times=np.asarray(gt['timestamps_s'],np.float64);times-=times[0]
        assert np.allclose(times,np.asarray(old['timestamps_relative_s']),atol=1e-9,rtol=0)
        preds={};hashes={str((base/'prediction/geometry.json').relative_to(docs)):sha(base/'prediction/geometry.json'),str((base/'reference/geometry.json').relative_to(docs)):sha(base/'reference/geometry.json')}
        for label in ['prediction','reference']:
            meta=eg if label=='prediction' else gt
            for k in ['joints','valid','fill_mask','joint_valid']:
                if k in meta:
                    path=base/label/meta[k]['url'];hashes[str(path.relative_to(docs))]=sha(path)
        for method in ['hggt','hawor']:
            path=source/f'downloads/predictions/{sid}_{method}.npz';p=np.load(path,allow_pickle=False)
            assert p['side_order'].tolist()==SIDES and np.array_equal(p['frame_ids'],seq['frame_ids'])
            assert np.allclose(p['timestamps_s']-p['timestamps_s'][0],times,atol=1e-9,rtol=0)
            fill=np.asarray(old['methods']['hawor']['infilled'],bool) if method=='hawor' else np.zeros((frames,2),bool)
            if method=='hawor': assert np.array_equal(fill,np.asarray(old['methods']['hawor']['valid'],bool)&~np.asarray(old['input_valid'],bool))
            joints=np.asarray(p['joints_cam_native'] if method=='hggt' else p['joints_cam'],np.float64)
            preds[method]={'joints':joints,'valid':p['valid'].astype(bool),'filled':fill}
            if method=='hggt':preds[method].update(adapted=np.asarray(p['joints_cam'],np.float64),adapter_valid=p['adapter_valid'].astype(bool))
            hashes[f'hggt_hawor/downloads/predictions/{sid}_{method}.npz']=sha(path)
        preds['egofound3r']={'joints':ea['joints'].astype(np.float64),'valid':ea['valid'].astype(bool),'filled':ea['fill_mask'].astype(bool)}
        for p in preds.values():
            finite=np.isfinite(p['joints'][:,:,BASE16]).all((-1,-2));p['strict']=eligible&p['valid']&finite&~p['filled'];p['display']=eligible&(p['valid']|p['filled'])&finite
            p['metrics']=metric_arrays(p['joints'],truth,times)
        common=np.logical_and.reduce([p['strict'] for p in preds.values()]);triplets=np.zeros_like(common);triplets[1:-1]=common[:-2]&common[1:-1]&common[2:]
        assert np.all(preds['hggt']['adapter_valid'][common])
        assert np.isfinite(preds['hggt']['adapted'][:,:,BASE16][common]).all()
        adapted=metric_arrays(preds['hggt']['adapted'],truth,times)
        # Check unchanged legacy Base16 RR on its original two-method mask.
        original_mask=eligible&np.asarray(old['input_valid'],bool)&preds['hggt']['valid']&preds['hawor']['valid']
        oldsample=next(x for x in oldmetrics['samples'] if x['id']==sid)
        for key,title in [('hggt','HGGT'),('hawor','HaWoR')]:
            expected=next(x for x in oldsample['methods'] if x['method']==title)['base16_rr_mpjpe_mm']
            assert abs(average(preds[key]['metrics']['rr_mpjpe16_mm'],original_mask)-expected)<.002
        item={'id':sid,'label':seq['label'],'dataset':seq['dataset'],'camera':match['camera_id'],'view':view,'rgb_frames':frames,
              'gt_hand_frames':int(eligible.sum()),'common_hand_frames':int(common.sum()),'common_acceleration_triplets':int(triplets.sum()),
              'common_pairs':[{'source_frame_id':int(seq['frame_ids'][f]),'side':SIDES[h]} for f,h in np.argwhere(common)],
              'source_sha256':hashes,'source_timestamp_policy':'GT/reference original timestamps; not nominal playback FPS.','methods':{}}
        for method in METHODS:
            p=preds[method];strict=p['strict'];metrics=p['metrics'];row={'label':LABELS[method],'strict_hand_frames':int(strict.sum()),'gt_hand_frames':int(eligible.sum()),
                'coverage_fraction':float(strict.sum()/eligible.sum()),'excluded_fill_hand_frames':int((p['filled']&eligible).sum()),'displayable_hand_frames':int(p['display'].sum()),
                'common_hand_frames':int(common.sum()),'common_acceleration_triplets':int(triplets.sum()),'pck50_all_gt':float(metrics['correct50'][strict].sum()/(eligible.sum()*16)),
                'correct_joints_50mm':int(metrics['correct50'][strict].sum()),'gt_joint_count':int(eligible.sum()*16)}
            pool=pooled[method]
            for k in ['mpjpe16_mm','rr_mpjpe16_mm','pa_mpjpe16_mm','wrist_error_mm','rr_acceleration16_error_m_s2']:
                mask=triplets if k=='rr_acceleration16_error_m_s2' else common;row[k]=average(metrics[k],mask);pool['values'].setdefault(k,[]).extend(metrics[k][mask].tolist())
            if method=='hggt':
                amask=strict&p['adapter_valid']&np.isfinite(p['adapted'][:,:,BASE16]).all((-1,-2));assert np.array_equal(amask,strict)
                row['adapted_mpjpe16_mm']=average(adapted['mpjpe16_mm'],common);row['adapted_wrist_error_mm']=average(adapted['wrist_error_mm'],common)
                row['adapted_pck50_all_gt']=float(adapted['correct50'][amask].sum()/(eligible.sum()*16));pool['adapted_correct50']+=int(adapted['correct50'][amask].sum())
                for k in ['mpjpe16_mm','wrist_error_mm']:pool['adapted_values'].setdefault(k,[]).extend(adapted[k][common].tolist())
                if common.any():assert np.max(np.abs(metrics['rr_mpjpe16_mm'][common]-adapted['rr_mpjpe16_mm'][common]))<.001
            for k,val in [('gt',eligible.sum()),('strict',strict.sum()),('filled',(p['filled']&eligible).sum()),('display',p['display'].sum()),('correct50',metrics['correct50'][strict].sum()),('common',common.sum()),('triplets',triplets.sum())]:pool[k]+=int(val)
            item['methods'][method]=row
        result['samples'].append(item)
    agg={'id':'all','label':'四段匹配样本汇总','rgb_frames':sum(s['rgb_frames'] for s in result['samples']),'gt_hand_frames':sum(s['gt_hand_frames'] for s in result['samples']),
         'common_hand_frames':sum(s['common_hand_frames'] for s in result['samples']),'common_acceleration_triplets':sum(s['common_acceleration_triplets'] for s in result['samples']),'methods':{}}
    for method,p in pooled.items():
        row={'label':LABELS[method],'strict_hand_frames':p['strict'],'gt_hand_frames':p['gt'],'coverage_fraction':p['strict']/p['gt'],
             'excluded_fill_hand_frames':p['filled'],'displayable_hand_frames':p['display'],'common_hand_frames':p['common'],'common_acceleration_triplets':p['triplets'],
             'pck50_all_gt':p['correct50']/(p['gt']*16),'correct_joints_50mm':p['correct50'],'gt_joint_count':p['gt']*16}
        row.update({k:float(np.mean(v)) if len(v) else None for k,v in p['values'].items()})
        if method=='hggt':row.update({f'adapted_{k}':float(np.mean(v)) if len(v) else None for k,v in p['adapted_values'].items()});row['adapted_pck50_all_gt']=p['adapted_correct50']/(p['gt']*16)
        agg['methods'][method]=row
    result['aggregate']=agg
    assert agg['rgb_frames']==446 and agg['gt_hand_frames']==746 and agg['common_hand_frames']==565
    assert [agg['methods'][k]['strict_hand_frames'] for k in METHODS]==[715,565,715]
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--legacy-root',type=Path,required=True);a=p.parse_args()
    result=build(a.root,a.legacy_root);out=a.root/'docs/cross-method-comparison.json';out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    path=a.root/'docs/cross-method-comparison.csv';rows=[]
    for scope in [result['aggregate'],*result['samples']]:
        for method,data in scope['methods'].items():rows.append({'scope':scope['id'],'method':method,**data})
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(json.dumps(result['aggregate'],ensure_ascii=False,indent=2))
if __name__=='__main__':main()
