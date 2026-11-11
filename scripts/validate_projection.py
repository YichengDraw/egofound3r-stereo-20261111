#!/usr/bin/env python3
"""Audit existing exported calibration/projection, without new model inference."""
import json,pathlib,numpy as np
root=pathlib.Path(__file__).resolve().parents[1];docs=root/'docs';manifest=json.loads((docs/'manifest.json').read_text());checks=[]
def arr(g,k,base):
 v=g.get(k)
 if v is None:return None
 if isinstance(v,dict):return np.fromfile(base/v['url'],dtype={'float32':'<f4','uint8':'u1','uint32':'<u4'}[v['dtype']]).reshape(v['shape'])
 return np.asarray(v)
for seq in manifest['sequences']:
 for m in seq['methods']:
  if m['id'].endswith('_world'):continue
  path=docs/m['geometry'];g=json.loads(path.read_text());b=path.parent;J=arr(g,'joints',b);K=arr(g,'projection_K',b);uv=arr(g,'joints2d',b);valid=arr(g,'valid',b).astype(bool)
  if K.ndim==2:K=np.broadcast_to(K,(len(J),3,3))
  q=np.einsum('fij,fhvj->fhvi',K,J);proj=q[...,:2]/np.maximum(q[...,2:],1e-12)
  mask=np.isfinite(uv).all(-1)&np.isfinite(J).all(-1)&(J[...,2]>1e-6)&valid[:,:,None]
  jv=arr(g,'joint_valid',b)
  if jv is not None:mask&=jv.astype(bool)
  err=np.linalg.norm(proj[mask]-uv[mask],axis=-1);mx=float(err.max()) if len(err) else None
  assert mx is None or mx<.03,(seq['id'],m['id'],mx)
  if m['kind']!='reference':
   V=arr(g,'vertices',b);faces=arr({'f':g['faces']['left']},'f',b);assert faces.min()>=0 and faces.max()<V.shape[2]
   assert V.shape[:2]==valid.shape
  checks.append({'sequence':seq['id'],'method':m['id'],'valid_projected_joints':int(mask.sum()),'max_reprojection_error_px':mx})
result={'status':'passed','exports_checked':len(checks),'max_reprojection_error_px':max(x['max_reprojection_error_px'] or 0 for x in checks),'checks':checks,'scope':'Known camera K applied to real camera-space geometry; world views load the corresponding camera-space projection payload. GT is real joints only, no invented mesh.'}
(root/'reports/projection_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
