#!/usr/bin/env python3
"""Import immutable cached EgoHSI3R joints and public materials; no inference.

The source directory contains a minimal extraction of the two historical NPZs,
their private .source.json receipts, and the original public site-data.json.
Private receipt paths are never copied into the website.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from urllib.parse import urljoin
import numpy as np
from build_cross_method_comparison import BASE16, read_geometry

SITE = 'https://egohsi3r-experiments-20261104.yichengdraw.chatgpt.site/'
MATCHES = [('R0', 'dexycb_box'), ('R1', 'dexycb_banana')]
EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]
NAMES = ['wrist'] + [f'{finger}_{part}' for finger in ['thumb','index','middle','ring','little'] for part in ['base','middle','distal','tip']]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def save_array(folder, name, values, dtype='float32'):
    a = np.ascontiguousarray(values, dtype={'float32':'<f4','uint8':'u1'}[dtype])
    path = folder/(name+'.bin'); a.tofile(path)
    return {'url':path.name, 'dtype':dtype, 'shape':list(a.shape), 'byte_length':a.nbytes, 'byte_order':'little', 'sha256':sha(path)}

def public_media(media):
    return {key:{k:(urljoin(SITE,v) if k in ('url','poster') else v) for k,v in item.items() if k in ('url','poster','fps','frameCount','clock','notes')}
            for key,item in media.items()}

def export(root, source):
    docs = root/'docs'; public = json.loads((source/'site-data.json').read_text())
    pixel_evidence = json.loads((source/'model_pixel_evidence.json').read_text())
    originals = []
    for seq in public['sequences']:
        variant = next(v for v in seq['variants'] if v['id']==seq['id']+'_detector')
        originals.append({'source_id':seq['id'], 'label':seq['name'], 'dataset':seq['dataset'], 'frames':seq['frames'],
                          'media':public_media(variant['media']), 'original_scores':variant['metrics'],
                          'original_score_definition':'Historical 21-point marker metrics; not the new Base16 comparison.',
                          'condition':variant['notes']})
    materials = {'schema':'egohsi3r.imported_materials.v1', 'source_site':SITE,
                 'source_manifest_url':urljoin(SITE,'site-data.json'), 'source_manifest_sha256':sha(source/'site-data.json'),
                 'code_commit':public['codeCommit'], 'weight_sha256':public['weightSha256'],
                 'original_sequence_count':public['defaultSequences'], 'original_frame_count':public['defaultFrames'],
                 'original_sequences':originals, 'samples':[],
                 'scope':'Only the default detector runs R0/R1 are imported into matched geometry and Base16 scoring. No GT-box, focal-control, reset or trained ablation variants are substituted.',
                 'no_model_rerun':True}
    evidence = {'schema':'egohsi3r.import_alignment_evidence.v1','remote_files_modified':False,'model_inference_run':False,'samples':[]}
    for rid,sid in MATCHES:
        path=source/(rid+'.npz'); receipt=json.loads((source/(rid+'.source.json')).read_text())
        assert sha(path)==receipt['extracted_npz_sha256']
        z=np.load(path,allow_pickle=False)
        pixel=next(x for x in pixel_evidence['samples'] if x['id']==rid)
        assert pixel['model_rgb_shape']==[384,512,3]
        assert np.array_equal(z['intrinsics'][0],pixel['K'])
        assert z['hand_order'].tolist()==['right','left']
        assert receipt['sample_metadata']['camera_id']=='836212060125'
        base=docs/'data'/sid/'cam_a'; gt,ga=read_geometry(base/'reference/geometry.json')
        assert gt['hand_order']==['right'] and gt['units']=='m' and gt['coordinate_space']=='camera'
        ids=z['source_frame_id'].astype(np.int64); times=z['timestamp_s'].astype(np.float64)
        assert np.array_equal(ids,gt['frame_indices']) and np.array_equal(ids,z['gt_frame_id'])
        assert np.array_equal(times,gt['timestamps_s'])
        assert np.array_equal(z['gt_joints_m'],ga['joints'][:,0])
        assert np.array_equal(z['gt_valid'],ga['joint_valid'][:,0].all(-1))
        assert np.array_equal(z['joints_camera'][:,:,BASE16],z['joints_official21_camera'][:,:,BASE16],equal_nan=True)
        joints=z['joints_camera'][:,[1,0]].copy(); valid=z['hand_mask'][:,[1,0]].astype(bool)
        assert joints.shape==(len(ids),2,21,3)
        joint_valid=np.isfinite(joints).all(-1)&valid[:,:,None]
        assert (joint_valid.all(-1)==valid).all()
        # Both cached model frame-000000 RGB arrays are 384x512; original inputs
        # are 480x640 with the same aspect ratio. No crop or pad translation.
        K=z['intrinsics'].astype(np.float64).copy(); assert K.shape==(len(ids),3,3)
        assert np.isfinite(K).all() and np.allclose(K[:,2],[0,0,1])
        K[:,:2]*=1.25
        homogeneous=np.einsum('tij,thnj->thni',K,joints)
        uv=np.full(joints.shape[:-1]+(2,),np.nan,np.float32)
        projectable=joint_valid&(joints[:,:,:,2]>0)
        uv[projectable]=homogeneous[:,:,:,:2][projectable]/homogeneous[:,:,:,2:][projectable]
        folder=base/'egohsi3r'; folder.mkdir(exist_ok=True)
        geo={'schema_version':'egofound3r.viewer.geometry.v1', 'kind':'prediction', 'data_role':'model_prediction',
             'label':'EgoHSI3R · 默认 RGB', 'model':'EgoHSI3R', 'is_model_prediction':True,
             'frame_count':len(ids),'hand_count':2,'joint_count':21,'vertex_count':0,
             'coordinate_convention':'opencv','coordinate_space':'camera','units':'m','world_scope':'per_frame_camera',
             'hand_order':['left','right'],'hand_labels':['左手','右手'],
             'joint_order':'native_openpose21','joint_root_index':0,'joint_names':NAMES,'joint_edges':EDGES,
             'frame_indices':ids.tolist(),'timestamps_s':times.tolist(),'fps':30.0,
             'joints':save_array(folder,'joints',joints), 'valid':save_array(folder,'valid',valid,'uint8'),
             'display_valid':save_array(folder,'display_valid',valid,'uint8'),
             'joint_valid':save_array(folder,'joint_valid',joint_valid,'uint8'),
             'joints2d':save_array(folder,'joints2d',uv), 'projection_K':save_array(folder,'projection_K',K),
             'projection_space':'input_rgb_pixels','projection_image_size_hw':[480,640],
             'projection_source':'Stored model-estimated intrinsics for 512x384 RGB, rescaled by 1.25 to original 640x480 RGB. Not ground-truth K.',
             'validity_policy':'Original hand_mask and finite joints. No missing-hand geometry is generated or filled.',
             'notice':'真实缓存的21关节骨架；本层不包含手部网格。投影使用模型原始估计内参，无GT拟合。',
             'source':{'model':'EgoHSI3R','source_site':SITE,'source_sample_id':rid,
                       'camera_id':'836212060125','recording_id':receipt['sample_metadata']['recording_id'],
                       'source_prediction_filename':rid+'_detector.npz','inference_sha256':receipt['original_prediction_sha256'],
                       'source_array':'joints_camera','source_hand_order':['right','left'], 'export_hand_permutation':[1,0],
                       'source_input_video_sha256':receipt['sample_metadata']['expected_sha256'],
                       'source_model_image_size_hw':[384,512],'source_input_image_size_hw':[480,640],
                       'checkpoint_code_commit':public['codeCommit'],'checkpoint_sha256':public['weightSha256'],
                       'model_inference_rerun':False,'gt_used_for_inference':False}}
        save_json(folder/'geometry.json',geo)
        current=json.loads((base/'prediction/geometry.json').read_text())
        identity={'source_id':rid,'id':sid,'camera_id':'836212060125','view':'cam_a','frame_count':len(ids),'frame_ids':ids.tolist(),
                  'source_frame_ids_equal':True,'timestamps_equal':True,'timestamp_max_difference_s':0.0,
                  'gt_joint_arrays_exactly_equal':True,'gt_max_difference_m':0.0,'gt_masks_exactly_equal':True,
                  'base16_native_vs_official_max_difference_m':0.0,
                  'original_prediction_sha256':receipt['original_prediction_sha256'],
                  'original_gt_sha256':receipt['original_gt_sha256'],
                  'reference_joints_sha256':sha(base/'reference'/gt['joints']['url']),
                  'source_input_video_sha256':receipt['sample_metadata']['expected_sha256'],
                  'egofound_input_video_sha256':current['source']['rgb_video_sha256'],
                  'input_video_bytes_identical':receipt['sample_metadata']['expected_sha256']==current['source']['rgb_video_sha256'],
                  'image_identity_note':'Same dataset recording, camera and source frame IDs. Video encoding differs; no claim of byte-identical compressed or decoded RGB.',
                  'source_intrinsics':'model estimated, not calibrated K',
                  'source_intrinsics_image_size_hw':[384,512], 'source_intrinsics_rescale_to_input':1.25}
        orig=next(x for x in originals if x['source_id']==rid)
        sample={**identity,'label':orig['label'],'dataset':'DexYCB','fps':30.0,
                'geometry':str((folder/'geometry.json').relative_to(docs)),
                'videos':[{'id':key,**item} for key,item in orig['media'].items()],
                'original_scores':orig['original_scores'],'original_score_definition':orig['original_score_definition'],
                'condition':orig['condition']}
        materials['samples'].append(sample)
        evidence['samples'].append({**identity,'original_prediction_path':receipt['original_prediction_path'],
                                  'original_gt_path':receipt['original_gt_path'],'array_specs':receipt['array_specs'],
                                  'model_pixel_evidence':pixel,
                                  'local_extraction_path':str(path),'local_extraction_sha256':sha(path)})
    save_json(docs/'egohsi3r-materials.json',materials)
    save_json(source/'alignment_evidence.json',evidence)
    return materials,evidence

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--source',type=Path,required=True)
    args=ap.parse_args();m,e=export(args.root,args.source);print(json.dumps({'matched_samples':[x['id'] for x in m['samples']], 'same_gt':True,'source_site':SITE}))
