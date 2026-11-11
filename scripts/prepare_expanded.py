#!/usr/bin/env python3
"""Freeze and prepare the nine additional previously used calibrated RGB clips.

Only images, calibration and the declared input timeline go into data/.
Hand/object labels are prepared separately under evaluation_ref/.
"""
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
from datetime import datetime, timezone
import cv2
import numpy as np

ROOT=Path('/nvme-ssd2/zhaoyicheng/egofound3r-reproduction-20261111')
BASE=ROOT.parent
CLIPS=[1976,2022,3070,3209,3302,3353]
HO=['MC1','SMu1','SM2']
HO_ORDER=[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]
JOINT_NAMES=['wrist']+[f'{finger}_{part}' for finger in ['thumb','index','middle','ring','little'] for part in ['mcp','pip','dip','tip']]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def video_from_images(images,folder,fps):
    folder.mkdir(parents=True,exist_ok=True)
    rgb=folder/'rgb';rgb.mkdir(exist_ok=True)
    dims=set();channel=[]
    for i,src in enumerate(images):
        a=cv2.imread(str(src));assert a is not None,(i,src)
        dims.add(tuple(a.shape[:2]));channel.append(float(np.mean(np.abs(a[:,:,0].astype(float)-a[:,:,1]))))
        dest=rgb/f'{i:06d}.png'
        if not dest.exists():dest.symlink_to(src.resolve())
    assert len(dims)==1
    size=next(iter(dims));path=folder/'input.mp4'
    if not path.exists():
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','2','-framerate',str(fps),'-i',str(rgb/'%06d.png'),
            '-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(path)],check=True)
    cap=cv2.VideoCapture(str(path));n=0;shapes=set()
    while True:
        ok,im=cap.read()
        if not ok:break
        n+=1;shapes.add(im.shape[:2])
    cap.release();assert n==len(images) and shapes==dims,(path,n,len(images),shapes,dims)
    cv2.imwrite(str(folder/'poster.jpg'),cv2.imread(str(images[0])))
    return path,size,{'decoded_frames':n,'size_hw':list(size),'min_mean_BG_channel_difference':min(channel),'video_sha256':sha(path),
        'encoding':'H264 CRF16 yuv420p; original images retained through symlinks; no spatial resize',
        'source_first_sha256':sha(images[0]),'source_last_sha256':sha(images[-1])}

def calibration(folder,video,size,K):
    n=len(K)
    p=folder/'root_intrinsics.json'
    save(p,{'schema_version':'egofound3r.root_intrinsics_input.v1','camera_model':'pinhole','coordinate_frame':'rgb_camera',
        'video_sha256':sha(video),'image_size_hw':list(size),'source_frame_indices':list(range(n)),'K':np.asarray(K).tolist(),'valid':[True]*n})
    return p

def main():
    manifest=json.loads((ROOT/'data/input_manifest.json').read_text())
    manifest['selection']={'frozen_utc':datetime.now(timezone.utc).isoformat(),'rule':'All six available calibrated Aria RGB clips and all three recent HO3D clips; selected by data availability before expanded inference, no outcome filtering',
        'excluded':[{'dataset':'ARCTIC','sequences':['s03_box_grab_01_1','s03_mixer_grab_01_1','s03_waffleiron_grab_01_1'],
            'reason':'Recent RGB images exist, but no verified source camera calibration found in supplied project roots; do not label estimated SfM intrinsics as known K'}]}
    for s in manifest['sequences']:
        s.update(dataset='DexYCB',input_group='dual_rgb_independent',source_timing='dataset 30 FPS; identical source frame IDs across two RGB cameras')
    save(ROOT/'reports/expanded_selection.json',manifest['selection'])
    report=[];evaluation=[]
    for clip in CLIPS:
        sid=f'aria_{clip}';src=BASE/'aria-calibration-audit-20261030/hot3d_official'/f'{clip:06d}'
        meta=json.loads((src/'manifest.json').read_text());frames=meta['frames'];n=len(frames);assert n>0
        source_ids=np.array([f['source_frame'] for f in frames]);ts=np.array([f['timestamp_ns'] for f in frames],np.int64)
        K=np.array([f['K_768'] for f in frames]);images=[src/'images'/f'{i:04d}.png' for i in range(n)]
        folder=ROOT/'data'/sid/'rgb_214_1';video,size,val=video_from_images(images,folder,30)
        kf=calibration(folder,video,size,K)
        view={'id':'rgb','camera_id':'214-1','video':str(video),'K_file':str(kf),'rgb_dir':str(folder/'rgb'),'image_size_hw':list(size),'poster':str(folder/'poster.jpg')}
        seq={'id':sid,'label':f'HOT3D-Aria {clip:06d}','dataset':'HOT3D-Aria','input_group':'monocular_rgb_supplement','frames':n,'fps':30.0,
            'frame_ids':list(range(n)),'source_frame_ids':source_ids.tolist(),'timestamps_s':((ts-ts[0])*1e-9).tolist(),'views':[view],
            'source_timing':'Original hardware timestamps retained; official video uses nominal 30 FPS; no reordered or duplicated frames',
            'pixel_contract':meta['tool'],'source_manifest':str(src/'manifest.json')}
        manifest['sequences'].append(seq)
        # Independent scoring source, matching the prior HGGT/HaWoR comparison.
        gp=Path('/nvme-ssd1/wlq/doas-reproduction-20261018/aria/eval_gt/trajectory_gt')/f'clip-{clip:06d}'/'gt_hand_object.npz'
        z=np.load(gp,allow_pickle=False)
        lookup={int(t):i for i,t in enumerate(z['timestamp_ns'])}
        gi=np.array([lookup[int(t)] for t in ts]);assert np.array_equal(ts,z['timestamp_ns'][gi])
        c2w=z['c2w'][gi];w2c=np.linalg.inv(c2w)
        joints=np.stack([np.einsum('tij,tkj->tki',w2c[:,:3,:3],z[side+'_joints_world'][gi])+w2c[:,:3,3,None].transpose(0,2,1) for side in ['left','right']],axis=1)
        valid=np.stack([z[side+'_valid'][gi] for side in ['left','right']],axis=1).astype(bool)
        gtpath=ROOT/'evaluation_ref'/sid/'rgb'/'gt.npz';gtpath.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(gtpath,joints=joints.astype(np.float32),valid=valid,source_frame_indices=np.arange(n),
            units=np.array('m'),coordinate_frame=np.array('camera'),hand_sides=np.array(['left','right']),joint_order=np.array('OpenPose21'),joint_names=np.array(JOINT_NAMES),
            timestamps_s=(ts-ts[0])*1e-9,hardware_timestamps_ns=ts,camera_to_world=c2w,K=K,eval_only=np.array(True))
        if clip in [1976,3209]:
            prior=np.load(BASE/'hggt-reproduction-20261110/data'/sid/'evaluation_gt.npz')
            err=float(np.max(np.abs(joints[valid]-prior['joints_camera'][valid])));assert err<1e-6
            val['prior_scoring_cache_max_difference_m']=err
        seq['available_source_frame_range']=[int(source_ids[0]),int(source_ids[-1])]
        seq['selection_note']='Use every frame in the existing audited pinhole cache; Aria 2022 cache starts at original frame 10 (140 frames), other clips have 150.'
        val.update(id=sid,dataset='HOT3D-Aria',input_group=seq['input_group'],gt_hand_frames=int(valid.sum()),
            calibration_sha256=sha(kf),gt_sha256=sha(gtpath),timestamp_spacing_min_s=float(np.diff(ts).min()*1e-9),timestamp_spacing_max_s=float(np.diff(ts).max()*1e-9))
        report.append(val);evaluation.append({'run_id':sid+'__rgb','gt_path':str(gtpath),'source':str(gp),'hand_sides':['left','right'],'joint_order':'OpenPose21','source_gt_sha256':sha(gp)})
        print('PREPARED',sid,n,flush=True)
    for name in HO:
        sid='ho3d_'+name.lower();src=BASE/'ghost-reproduction-20261026/data'/f'hold_{name}_ho3d'
        meta=json.loads((src/'source_frame_map.json').read_text());rows=meta['source_frame_map'];n=len(rows)
        images=[src/'images'/row['input_file'] for row in rows]
        labels=[pickle.load((src/row['source_meta_in_sequence']).open('rb')) for row in rows]
        # Calibration is the sole model input extracted from these metadata records.
        published_K=np.array([x['camMat'] for x in labels if np.shape(x.get('camMat'))==(3,3)])
        assert len(published_K)>0 and np.all(published_K==published_K[0])
        assert all(x['camIDList']==labels[0]['camIDList'] for x in labels)
        K=np.repeat(published_K[:1],n,axis=0)
        missing_label_ids=[rows[i]['source_frame'] for i,x in enumerate(labels) if np.shape(x.get('handJoints3D'))!=(21,3)]
        folder=ROOT/'data'/sid/'rgb';video,size,val=video_from_images(images,folder,10)
        kf=calibration(folder,video,size,K)
        view={'id':'rgb','camera_id':str(labels[0]['camIDList'][0]),'video':str(video),'K_file':str(kf),'rgb_dir':str(folder/'rgb'),'image_size_hw':list(size),'poster':str(folder/'poster.jpg')}
        seq={'id':sid,'label':f'HO3D-v3 {name}','dataset':'HO3D-v3','input_group':'monocular_rgb_supplement','frames':n,'fps':10.0,
            'frame_ids':list(range(n)),'source_frame_ids':[row['source_frame'] for row in rows],'timestamps_s':(np.arange(n)/10).tolist(),'views':[view],
            'source_timing':'Source acquisition FPS is not established in cached provenance. Ordered sampled frames are supplied at an explicitly assumed 10 FPS; do not interpret time derivatives as physical motion.',
            'source_stride':meta['source_frame_stride'],'source_manifest':str(src/'source_frame_map.json'),'object_label':labels[0]['objName'],
            'pixel_contract':'original 640x480 RGB; no crop or spatial resize; H264 CRF16 encoding'}
        manifest['sequences'].append(seq)
        # Official HO3D OpenGL camera -> CV camera; official MANO-to-simple mapping.
        joints=np.array([x['handJoints3D'][HO_ORDER] if np.shape(x.get('handJoints3D'))==(21,3) else np.full((21,3),np.nan) for x in labels],np.float32)*np.array([1,-1,-1],np.float32)
        joints=joints[:,None];valid=np.isfinite(joints).all(-1)&(joints[...,2]>0)
        gtpath=ROOT/'evaluation_ref'/sid/'rgb'/'gt.npz';gtpath.parent.mkdir(parents=True,exist_ok=True)
        objR=np.stack([cv2.Rodrigues(np.asarray(x['objRot']).reshape(3))[0] if np.shape(x.get('objRot'))==(3,) else np.full((3,3),np.nan) for x in labels]);flip=np.diag([1,-1,-1])
        objpose=np.broadcast_to(np.eye(4),(n,4,4)).copy();objpose[:,:3,:3]=flip[None]@objR
        objpose[:,:3,3]=np.array([x['objTrans'] if np.shape(x.get('objTrans'))==(3,) else [np.nan]*3 for x in labels])*np.array([1,-1,-1])
        np.savez_compressed(gtpath,joints=joints,valid=valid,source_frame_indices=np.arange(n),dataset_source_frame_ids=np.array(seq['source_frame_ids']),
            units=np.array('m'),coordinate_frame=np.array('camera'),hand_sides=np.array(['right']),joint_order=np.array('OpenPose21'),joint_names=np.array(JOINT_NAMES),K=K,timestamps_s=np.arange(n)/10,
            contact=np.array([x['handVertContact'] if np.shape(x.get('handVertContact'))==(778,) else np.full(778,np.nan) for x in labels]),
            contact_distance=np.array([x['handVertDist'] if np.shape(x.get('handVertDist'))==(778,) else np.full(778,np.nan) for x in labels]),
            object_pose_camera=objpose,object_name=np.array(labels[0]['objName']),eval_only=np.array(True))
        seq['missing_annotation_source_frame_ids']=missing_label_ids
        seq['calibration_note']='Every published K for this same fixed camera is exactly equal. This static camera calibration also applies to RGB frames whose hand/object metadata are absent.'
        val.update(id=sid,dataset='HO3D-v3',input_group=seq['input_group'],gt_hand_frames=int(valid.any(-1).sum()),calibration_sha256=sha(kf),gt_sha256=sha(gtpath),timing_assumption=seq['source_timing'],missing_annotation_source_frame_ids=missing_label_ids)
        report.append(val);evaluation.append({'run_id':sid+'__rgb','gt_path':str(gtpath),'source':str(src/'gt_meta'),'hand_sides':['right'],'joint_order':'OpenPose21',
            'axis_transform':[1,-1,-1],'source_joint_permutation':HO_ORDER,'official_mapping_source':'https://github.com/shreyashampali/ho3d/blob/master/vis_HO3D.py'})
        print('PREPARED',sid,n,flush=True)
    save(ROOT/'data/input_manifest_expanded.json',manifest)
    save(ROOT/'reports/expanded_data_validation.json',{'status':'passed','added_streams':len(report),'added_rgb_frames':sum(r['decoded_frames'] for r in report),'streams':report,
        'total_sequences':len(manifest['sequences']),'total_streams':sum(len(s['views']) for s in manifest['sequences']),
        'total_rgb_frames':sum(s['frames']*len(s['views']) for s in manifest['sequences']),'input_gt':False,'input_depth':False,'input_extrinsics':False})
    save(ROOT/'evaluation_ref/expanded_evaluation_manifest.json',{'inference_access':False,'samples':evaluation})

if __name__=='__main__':main()
