#!/usr/bin/env python3
"""Audit fixed native slot/finger semantics; no per-frame metric assignment."""
import argparse,json
from pathlib import Path
import numpy as np
import cv2
from official_contract import load_prediction,hand_contract,dense_hand,array,JOINT_EDGES,finite_json

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
rid='aria_1976__rgb';payload=load_prediction(r/'outputs'/rid/'inference_output.pt');h=hand_contract(payload)
d=dense_hand(payload);gt=np.load(r/'evaluation_ref/aria_1976/rgb/gt.npz');K=gt['K'];J=h['joints']
tips=[745,317,444,556,673]
D=np.linalg.norm(J[:,:,[4,8,12,16,20],None,:]-d['vertices'][:,:,None,tips,:],axis=-1)
out={'audit_run':rid,'frames':[0,74,149],'native_tip_joints':[4,8,12,16,20],'anatomical_tip_vertices':tips,
 'finger_order':['thumb','index','middle','ring','little'],'median_internal_tip_distance_matrix_mm':np.median(D,axis=(0,1))*1000,
 'note':'This compares joints and vertices within one prediction, not to GT; it tests finger order without optimizing accuracy.'}
rows=[]
cap=cv2.VideoCapture(str(r/'data/aria_1976/rgb_214_1/input.mp4'))
for f in out['frames']:
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read();assert ok
 img=cv2.resize(img,(512,512));cols=[]
 for label,xyz,names in [('PRED native slots',J[f],['slot 0','slot 1']),('GT anatomy',gt['joints'][f],['LEFT','RIGHT'])]:
  canvas=img.copy();uv=np.einsum('ij,hnj->hni',K[f],xyz);uv=uv[...,:2]/uv[...,2:]*512/768
  for s in range(2):
   color=[(0,230,255),(255,70,230)][s]
   for x,y in JOINT_EDGES:
    if np.isfinite(uv[s,[x,y]]).all():cv2.line(canvas,tuple(np.rint(uv[s,x]).astype(int)),tuple(np.rint(uv[s,y]).astype(int)),color,2)
   x,y=np.rint(uv[s,0]).astype(int);cv2.putText(canvas,names[s],(int(np.clip(x,5,430)),int(np.clip(y-15,30,490))),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
   for j in [4,8,12,16,20]:
    x,y=np.rint(uv[s,j]).astype(int);cv2.putText(canvas,str(j),(x,y),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1)
  cv2.rectangle(canvas,(0,0),(512,28),(15,20,30),-1);cv2.putText(canvas,f'{label} | f={f}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);cols.append(canvas)
 rows.append(np.concatenate(cols,axis=1))
out['wrist_camera_xyz']={'pred':J[out['frames'],:,0].tolist(),'gt_left_right':gt['joints'][out['frames'],:,0].tolist()}
cv2.imwrite(str(r/'reports/anatomical_slot_audit.jpg'),np.concatenate(rows,axis=0),[cv2.IMWRITE_JPEG_QUALITY,78])
(r/'reports/anatomical_slot_audit.json').write_text(json.dumps(finite_json(out),indent=2)+'\n')
print(json.dumps(finite_json(out)))
