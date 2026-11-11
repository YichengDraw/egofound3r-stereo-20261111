#!/usr/bin/env python3
"""Prepare two real RGB cameras from previously used DexYCB clips.
No inference, depth input, GT prompts, or fitted calibration. Original JPG bytes
are copied from the official local archive. Annotation assets remain separate.
"""
import argparse, hashlib, json, os, shutil, subprocess, tarfile, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import yaml
from PIL import Image, ImageDraw

DEFAULT_ROOT = "/nvme-ssd2/zhaoyicheng/egofound3r-reproduction-20261111"
DEFAULT_ARCHIVE = "/nvme-ssd1/wlq/doas-reproduction-20261018/benchmarks/dexycb/downloads/20200928-subject-07.tar.gz"
DEFAULT_CALIB = "/nvme-ssd2/zhaoyicheng/egohsi3r-ablation-20261105/data/calibration"
SCENES = [
    ("dexycb_box", "Cracker box", "20200928_143606", 74),
    ("dexycb_banana", "Banana", "20200928_145724", 72),
]
CAMERAS = [("A", "836212060125"), ("B", "932122062010")]
JOINT_NAMES = ["wrist"] + [f"{finger}_{joint}" for finger in ["thumb","index","middle","ring","little"] for joint in ["mcp","pip","dip","tip"]]

def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def save_bytes(path, blob):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != blob:
            raise RuntimeError(f"Existing extracted asset differs: {path}")
        return False
    path.write_bytes(blob)
    return True

def load_K(path):
    d = yaml.load(Path(path).read_text(), Loader=yaml.FullLoader)["color"]
    return np.array([[d["fx"],0,d["ppx"]],[0,d["fy"],d["ppy"]],[0,0,1.]], dtype=np.float64)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--calibration", default=DEFAULT_CALIB)
    args = parser.parse_args()
    root, archive, calibration = Path(args.root), Path(args.archive), Path(args.calibration)
    data, evaluation, reports = root/"data", root/"evaluation_ref", root/"reports"
    for p in [data, evaluation, reports]:
        p.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    counts = {}
    expected = {}
    for sid, title, seq, frames in SCENES:
        expected[(seq, "meta.yml")] = 1
        for view, cam in CAMERAS:
            expected[(seq, cam, "rgb")] = frames
            expected[(seq, cam, "labels")] = frames
    receipt_path = reports/"data_extraction.json"
    prior = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if prior.get("completed") is not True:
        print("Extracting selected RGB and isolated evaluation labels from local archive", flush=True)
        selected_ids = {s[2]: s[0] for s in SCENES}
        total_members = 0
        last_report = started
        with tarfile.open(archive, mode="r|gz", bufsize=1024*1024) as tar:
            for member in tar:
                total_members += 1
                if not member.isfile():
                    continue
                parts = Path(member.name).parts
                if len(parts)<3 or parts[0]!="20200928-subject-07" or parts[1] not in selected_ids:
                    continue
                seq, sid = parts[1], selected_ids[parts[1]]
                output = None; key = None
                if len(parts)==3 and parts[2]=="meta.yml":
                    output = evaluation/sid/"meta.yml"; key=(seq,"meta.yml")
                elif len(parts)==4 and parts[2] in [c[1] for c in CAMERAS]:
                    cam, filename = parts[2], parts[3]
                    if filename.startswith("color_") and filename.endswith(".jpg"):
                        output = data/sid/cam/"rgb"/filename; key=(seq,cam,"rgb")
                    elif filename.startswith("labels_") and filename.endswith(".npz"):
                        output = evaluation/sid/cam/"labels"/filename; key=(seq,cam,"labels")
                if output is not None:
                    blob = tar.extractfile(member).read()
                    if len(blob) != member.size: raise RuntimeError("Short archive read")
                    save_bytes(output, blob)
                    counts[key] = counts.get(key,0)+1
                if time.monotonic()-last_report > 10:
                    print("members",total_members,"selected",sum(counts.values()),"elapsed_s",round(time.monotonic()-started,1), flush=True)
                    last_report = time.monotonic()
                if all(counts.get(k,0)==v for k,v in expected.items()):
                    break
        if any(counts.get(k,0)!=v for k,v in expected.items()):
            raise RuntimeError(f"Incomplete extraction: {counts} expected {expected}")
        dump(receipt_path, {
            "completed":True, "source_archive":str(archive), "archive_size_bytes":archive.stat().st_size,
            "archive_mtime_ns":archive.stat().st_mtime_ns, "source_dataset":"DexYCB",
            "selected_counts":{"/".join(k):v for k,v in counts.items()},
            "scanned_members":total_members, "whole_archive_extracted":False,
            "depth_extracted":False, "rgb_bytes_unchanged":True,
            "source_member_prefix":"20200928-subject-07",
            "elapsed_s":time.monotonic()-started
        })
    else:
        print("Using prior complete selected extraction", flush=True)

    K_by_cam = {}
    for view,cam in CAMERAS:
        K_by_cam[cam] = load_K(calibration/"intrinsics"/f"{cam}_640x480.yml")
    extr_source = calibration/"extrinsics_20200928_082347"/"extrinsics.yml"
    extr = yaml.load(extr_source.read_text(), Loader=yaml.FullLoader)
    T_world_cam = {}
    for view,cam in CAMERAS:
        T = np.eye(4); T[:3] = np.asarray(extr["extrinsics"][cam]).reshape(3,4)
        T_world_cam[cam] = T
    cam_a,cam_b = [c[1] for c in CAMERAS]
    T_B_A = np.linalg.inv(T_world_cam[cam_b]) @ T_world_cam[cam_a]
    baseline = np.linalg.norm(T_world_cam[cam_a][:3,3]-T_world_cam[cam_b][:3,3])
    angle = np.degrees(np.arccos(np.clip((np.trace(T_B_A[:3,:3])-1)/2,-1,1)))
    calib_receipt = {
        "source":str(extr_source),"source_sha256":sha(extr_source),
        "source_convention":"camera_to_world; world = master camera 840412060917",
        "convention_evidence":"https://github.com/NVlabs/dex-ycb-toolkit/blob/master/dex_ycb_toolkit/sequence_loader.py",
        "convention_evidence_functions":["_deproject_depth_and_filter_points","transform_ycb(camera_to_world=True)"],
        "T_world_camera":{c:T.tolist() for c,T in T_world_cam.items()},
        "T_B_from_A":T_B_A.tolist(),"camera_A":cam_a,"camera_B":cam_b,
        "baseline_m":float(baseline),"relative_rotation_deg":float(angle),
        "inference_access":False,
        "intended_use":"evaluation and explicitly labelled calibrated visualization only",
        "rig_scope":"Two synchronized RGB cameras from a fixed wide-baseline multi-camera rig; not a wearable narrow-baseline stereo device."
    }
    dump(evaluation/"calibration"/"stereo_extrinsics.json",calib_receipt)
    shutil.copy2(extr_source,evaluation/"calibration"/"extrinsics.yml")

    manifest = {
        "schema_version":"egofound3r.reproduction_inputs.v1","created_utc":datetime.now(timezone.utc).isoformat(),
        "dataset":"DexYCB","subject":"20200928-subject-07",
        "input_boundary":"Each view is run independently with RGB + known color-camera K. No depth, GT hand, object pose, mesh, or extrinsics enter model inputs.",
        "model_stereo_fusion":False,
        "camera_pair":{"A":cam_a,"B":cam_b},
        "rig_scope":calib_receipt["rig_scope"], "scenes":[]
    }
    evaluation_manifest = {"schema_version":"egofound3r.reproduction_evaluation.v1","inference_access":False,"samples":[]}
    validation = {"status":"checking","scenes":[],"image_contract":{
        "source":"Official DexYCB color JPG files",
        "additional_rectification":False,
        "lens_distortion_coefficients_provided":False,
        "official_sdk_policy":"Uses published color K directly on these RGB images without cv2.undistort; no distortion coefficients are supplied in intrinsics YAML.",
        "pinhole_assumption":"Use the dataset-provided pinhole convention. This does not independently certify physical lens distortion is zero."
    }}
    all_joints = {}
    contact_sheet = Image.new("RGB",(960,2*260),(242,245,248))
    draw = ImageDraw.Draw(contact_sheet)
    for scene_index,(sid,title,seq,n) in enumerate(SCENES):
        scene={"id":sid,"title":title,"sequence_id":seq,"frame_count":n,"fps":30.0,
               "frame_indices":list(range(n)),"timestamps_s":[i/30 for i in range(n)],
               "timestamp_contract":"Shared original DexYCB frame index at nominal30Hz; acquisition PTS are not included in archive labels.",
               "views":[]}
        scene_val={"id":sid,"views":[]}
        meta=yaml.load((evaluation/sid/"meta.yml").read_text(),Loader=yaml.FullLoader)
        assert meta["num_frames"]==n and meta["mano_sides"]==["right"]
        for view_index,(view,cam) in enumerate(CAMERAS):
            p=data/sid/cam;rgb=p/"rgb"
            filenames=sorted(rgb.glob("color_*.jpg"))
            assert [int(f.stem.split("_")[-1]) for f in filenames]==list(range(n))
            labels=sorted((evaluation/sid/cam/"labels").glob("labels_*.npz"))
            assert [int(f.stem.split("_")[-1]) for f in labels]==list(range(n))
            hashes=[]; color_channel_diffs=[]; joints=[];joints2d=[];pose_m=[];pose_y=[]
            for frame,(image_file,label_file) in enumerate(zip(filenames,labels)):
                with Image.open(image_file) as im:
                    assert im.size==(640,480) and im.mode=="RGB"
                    a=np.asarray(im)
                    color_channel_diffs.append(float(np.mean(np.abs(a[...,0].astype(float)-a[...,1].astype(float)))))
                hashes.append({"source_frame_index":frame,"rgb_sha256":sha(image_file),"label_sha256":sha(label_file)})
                with np.load(label_file,allow_pickle=False) as z:
                    joints.append(z["joint_3d"]);joints2d.append(z["joint_2d"]);pose_m.append(z["pose_m"]);pose_y.append(z["pose_y"])
            assert min(color_channel_diffs)>0, "RGB channels unexpectedly identical"
            j=np.array(joints); j2=np.array(joints2d)
            valid=np.isfinite(j).all(-1)&(j[...,2]>0)&(~np.all(j==-1,axis=-1))
            all_joints[(sid,cam)]=(j,valid)
            np.savez_compressed(evaluation/sid/cam/"gt.npz",
                joints=j,valid=valid,joints_2d=j2,source_frame_indices=np.arange(n,dtype=np.int64),
                timestamps_s=np.arange(n)/30.,units=np.array("m"),coordinate_frame=np.array("camera"),
                hand_sides=np.array(["right"]),joint_names=np.array(JOINT_NAMES),
                mano_pose=np.array(pose_m),object_poses=np.array(pose_y),
                K=K_by_cam[cam],T_world_camera=T_world_cam[cam],
                mapping_status=np.array("prediction_side_slot_mapping_requires_verification"))
            projected=j @ K_by_cam[cam].T
            projected=projected[...,:2]/projected[...,2:3]
            eval2d=valid&np.isfinite(j2).all(-1)&(~np.all(j2==-1,axis=-1))
            residual=np.linalg.norm(projected-j2,axis=-1)[eval2d]
            video=p/"input.mp4"
            if not video.exists():
                subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-framerate","30",
                    "-start_number","0","-i",str(rgb/"color_%06d.jpg"),"-frames:v",str(n),
                    "-c:v","libx264","-threads","2","-preset","fast","-crf","18","-pix_fmt","yuv420p",
                    "-movflags","+faststart",str(video)],check=True)
            probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames",
                    "-select_streams","v:0","-show_entries","stream=width,height,nb_read_frames,r_frame_rate,duration",
                    "-of","json",str(video)],text=True))["streams"][0]
            assert probe["width"]==640 and probe["height"]==480 and int(probe["nb_read_frames"])==n and probe["r_frame_rate"]=="30/1"
            shutil.copy2(filenames[0],p/"poster.jpg")
            kpath=p/"root_intrinsics.json"
            sidecar={"schema_version":"egofound3r.root_intrinsics_input.v1","camera_model":"pinhole",
                     "coordinate_frame":"rgb_camera","image_size_hw":[480,640],"K":K_by_cam[cam].tolist(),
                     "valid":True,"video_sha256":sha(video)}
            dump(kpath,sidecar)
            dump(p/"rgb_provenance.json",{"source_frame_indices":list(range(n)),"hashes":hashes,
                "source_member_template":f"20200928-subject-07/{seq}/{cam}/color_%06d.jpg",
                "video_derivation":"H264 CRF18 at nominal30fps; original JPG bytes retained; dimensions unchanged"})
            view_data={"id":f"{sid}_{view.lower()}","camera_id":cam,"view":view,
                "frame_count":n,"fps":30.0,"width":640,"height":480,
                "video_path":str(video),"poster_path":str(p/"poster.jpg"),"frames_path":str(rgb),
                "intrinsics_path":str(kpath),"K":K_by_cam[cam].tolist(),
                "source_frame_indices":list(range(n)),"video_sha256":sidecar["video_sha256"],
                "modality":"RGB","inference_status":"not_run"}
            scene["views"].append(view_data)
            evaluation_manifest["samples"].append({"id":view_data["id"],"gt_path":str(evaluation/sid/cam/"gt.npz"),
                "source_frame_indices":list(range(n)),"hand_sides":["right"],"joint_names":JOINT_NAMES,
                "prediction_mapping_status":"requires_explicit_verification","extrinsics_model_input":False})
            scene_val["views"].append({"id":view_data["id"],"rgb_count":n,"label_count":n,
                "all_rgb_dimensions":[640,480],"rgb_mode":"RGB","min_mean_RG_channel_difference":min(color_channel_diffs),
                "valid_joint_count":int(valid.sum()),"total_joint_count":int(valid.size),
                "K_projection_label_residual_px":{"mean":float(residual.mean()),"max":float(residual.max()),"count":len(residual)},
                "video_probe":probe,"video_sha256":sidecar["video_sha256"]})
            for key_index,idx in enumerate([0,n//2,n-1]):
                im=Image.open(filenames[idx]).resize((160,120))
                x=view_index*480+key_index*160;y=scene_index*260+20
                contact_sheet.paste(im,(x,y))
                draw.text((x+5,y+121),f"{title} {view} frame{idx}",fill=(25,35,45))
                if key_index==0:draw.text((x+5,y+141),cam,fill=(25,35,45))
        ja,va=all_joints[(sid,cam_a)];jb,vb=all_joints[(sid,cam_b)]
        transformed=ja @ T_B_A[:3,:3].T + T_B_A[:3,3]
        both=va&vb
        err=np.linalg.norm(transformed-jb,axis=-1)[both]
        scene_val["cross_camera_gt_consistency_mm"]={"mean":float(err.mean()*1000),"max":float(err.max()*1000),"joint_count":len(err)}
        # GT consistency validates the fixed calibration convention only; it is not model performance.
        manifest["scenes"].append(scene);validation["scenes"].append(scene_val)
    contact_sheet.save(data/"preview_contact_sheet.jpg",quality=92)
    validation.update(status="passed",total_sequences=2,total_camera_streams=4,total_rgb_frames=292,
        input_depth=False,input_gt=False,input_extrinsics=False,baseline_m=float(baseline),
        relative_rotation_deg=float(angle),completed_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_s=time.monotonic()-started)
    # Keep the executable inference manifest strictly limited to RGB/K/time.
    execution_manifest = {
        "schema_version":"egofound3r.inputs.v1",
        "sequences":[{
            "id":s["id"],"label":s["title"],"frames":s["frame_count"],"fps":s["fps"],
            "frame_ids":s["frame_indices"],"timestamps_s":s["timestamps_s"],
            "views":[{
                "id":"cam_" + v["view"].lower(),"camera_id":v["camera_id"],
                "video":v["video_path"],"K_file":v["intrinsics_path"],
                "rgb_dir":v["frames_path"],"image_size_hw":[v["height"],v["width"]]
            } for v in s["views"]]
        } for s in manifest["scenes"]]
    }
    evaluation_manifest["schema_version"]="egofound3r.evaluation.v1"
    evaluation_manifest["sequences"]=[{
        "id":sid,"sequence_id":seq,"frames":n,"views":[{
            "id":"cam_"+view.lower(),"camera_id":cam,
            "gt_npz":str(evaluation/sid/cam/"gt.npz"),
            "labels_dir":str(evaluation/sid/cam/"labels"),
            "hand_sides":["right"],"joint_names":JOINT_NAMES,
            "prediction_mapping_status":"requires_explicit_verification"
        } for view,cam in CAMERAS]
    } for sid,title,seq,n in SCENES]
    dump(data/"input_manifest.json",execution_manifest)
    dump(data/"preview_manifest.json",manifest)
    dump(evaluation/"evaluation_manifest.json",evaluation_manifest)
    dump(reports/"data_validation.json",validation)
    print(json.dumps({"status":"complete","manifest":str(data/"input_manifest.json"),
                      "validation":str(reports/"data_validation.json"),"rgb_frames":292,
                      "baseline_m":float(baseline),"relative_rotation_deg":float(angle)}),flush=True)

if __name__=="__main__":
    main()

