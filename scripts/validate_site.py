#!/usr/bin/env python3
"""Validate actual website assets against original inference tensors, once."""
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from official_contract import load_prediction,hand_contract,dense_hand,camera_contract,camera_to_world,sha256

DT={'float32':'<f4','float64':'<f8','uint8':'u1','uint16':'<u2','uint32':'<u4','int32':'<i4'}
def arrays(x,base,seen):
    if isinstance(x,dict):
        if {'url','dtype','shape'}.issubset(x):
            p=base/x['url'];assert p.is_file(),p;assert p.stat().st_size==np.prod(x['shape'])*np.dtype(DT[x['dtype']]).itemsize,p
            assert p.stat().st_size==x.get('byte_length',p.stat().st_size);seen.add(str(p));return
        for v in x.values():arrays(v,base,seen)
    elif isinstance(x,list):
        for v in x:arrays(v,base,seen)
def arr(g,key,base):
    x=g[key];return np.fromfile(base/x['url'],dtype=DT[x['dtype']]).reshape(x['shape'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;d=r/'docs';m=json.loads((d/'manifest.json').read_text());seen=set();checks=[];preds=refs=0
    assert len(m['sequences'])==11 and m['status']=='complete'
    for s in m['sequences']:
      for view,rgb in s['rgb'].items():
        for k in ['video','poster']:assert (d/rgb[k]).is_file(),rgb[k]
      for method in s['methods']:
        path=d/method['geometry'];g=json.loads(path.read_text());base=path.parent;arrays(g,base,seen)
        assert g['frame_count']==s['frames'] and g['units']=='m'
        if method['kind']=='reference':
            assert g['data_role']=='ground_truth_reference' and g['is_model_prediction'] is False and not g.get('vertices');refs+=1;continue
        preds+=1;rid=s['id']+'__'+method['view'];original=r/'outputs'/rid/'inference_output.pt';p=load_prediction(original);h=hand_contract(p);dense=dense_hand(p);J=h['joints'];V=dense['vertices'];C=camera_contract(p,len(J))
        if g['coordinate_space']=='world':J=camera_to_world(J,C['c2w']);V=camera_to_world(V,C['c2w'])
        np.testing.assert_allclose(arr(g,'joints',base),J,rtol=1e-6,atol=1e-6,equal_nan=True)
        np.testing.assert_allclose(arr(g,'vertices',base),V,rtol=1e-6,atol=1e-6,equal_nan=True)
        valid=arr(g,'valid',base).astype(bool);fill=arr(g,'fill_mask',base).astype(bool);assert not (valid&fill).any()
        np.testing.assert_array_equal(fill,h['fill_mask']);assert g['hand_order']=='left,right'
        assert g['source']['inference_sha256']==sha256(original)
        assert np.array_equal(g['frame_indices'],np.arange(s['frames']))
        checks.append({'method':method['id'],'sequence':s['id'],'coordinate':g['coordinate_space'],'native_tensor_equality':True,'strict_fill_excluded':True,'finite_valid_hand_frames':int(valid.sum())})
    for link in [*m['downloads'],*[x for s in m['sequences'] for x in s['downloads']]]:
        if not link['url'].startswith('https://'):assert (d/link['url']).is_file(),link
    assert preds==26 and refs==13
    report={'status':'passed','checked_utc':datetime.now(timezone.utc).isoformat(),'sequence_count':11,'prediction_geometry_exports':preds,'ground_truth_reference_exports':refs,'verified_binary_files':len(seen),'checks':checks,
      'scope':'File-level validation of real final assets; browser playback/drag/interaction acceptance is separately recorded.'}
    (r/'reports/site_asset_validation.json').write_text(json.dumps(report,indent=2)+'\n');(d/'downloads/site_asset_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='checks'}))

if __name__=='__main__':main()
