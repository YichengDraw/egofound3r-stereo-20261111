#!/usr/bin/env python3
"""Export explicitly supplied REAL GT joints as an evaluation-only reference.

This is not an EgoFound3R prediction exporter. It uses no checkpoint, never
changes inference inputs, and does not export object poses, MANO parameters,
meshes, or scene geometry. The input NPZ is an evaluation_ref gt.npz prepared
from original dataset annotations. Files are compatible with the joint-only
path of the static comparison viewer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DATA_ROLE = "ground_truth_reference"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_scalar(value, name: str) -> str:
    a = np.asarray(value)
    if a.ndim != 0 or a.dtype.kind not in ("U", "S"):
        raise ValueError(f"{name} must be an explicit scalar string")
    v = a.item()
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def strings(value, size: int, name: str) -> list[str]:
    a = np.asarray(value)
    if a.shape != (size,) or a.dtype.kind not in ("U", "S"):
        raise ValueError(f"{name} must contain {size} explicit strings")
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in a.tolist()]


def named_edges(names: list[str]) -> list[list[int]]:
    if len(names) != 21 or len(set(names)) != 21 or names[0] != "wrist":
        raise ValueError("Expected explicitly named 21-joint GT with wrist at index 0")
    ids = {name: index for index, name in enumerate(names)}
    edges = []
    for finger in ("thumb", "index", "middle", "ring", "little"):
        chain = ["wrist"] + [f"{finger}_{part}" for part in ("mcp", "pip", "dip", "tip")]
        if not all(name in ids for name in chain):
            raise ValueError("GT names do not establish the expected five finger chains")
        edges.extend([[ids[a], ids[b]] for a, b in zip(chain[:-1], chain[1:])])
    return edges


def write_array(output: Path, name: str, value, dtype: str) -> dict:
    encoding = {"float32": "<f4", "uint8": "u1"}[dtype]
    a = np.ascontiguousarray(value, dtype=encoding)
    path = output / (name + ".bin")
    temporary = path.with_suffix(".bin.tmp")
    a.tofile(temporary)
    temporary.replace(path)
    return {"url": path.name, "dtype": dtype, "shape": list(a.shape),
            "byte_length": int(a.nbytes), "byte_order": "little"}


def values_json(values) -> list:
    return [float(x) if np.isfinite(x) else None for x in np.asarray(values).reshape(-1)]


def export_reference(gt_path: Path, output: Path, intrinsics_path: Path) -> dict:
    gt_path, output, intrinsics_path = Path(gt_path), Path(output), Path(intrinsics_path)
    if not gt_path.is_file() or not intrinsics_path.is_file():
        raise FileNotFoundError("Real GT NPZ and the corresponding calibration sidecar are required")
    # Read only annotation fields needed for a skeleton. In particular, do not
    # load the NPZ's MANO pose, object poses, or world-camera extrinsic matrix.
    with np.load(gt_path, allow_pickle=False) as source:
        units = text_scalar(source["units"], "units")
        coordinate = text_scalar(source["coordinate_frame"], "coordinate_frame")
        if units not in ("m", "mm") or coordinate != "camera":
            raise ValueError("GT must explicitly declare camera coordinates and m or mm units")
        joints = np.asarray(source["joints"], dtype=np.float32).copy()
        if units == "mm":
            joints /= 1000
        if joints.ndim != 4 or joints.shape[-2:] != (21, 3) or joints.shape[0] < 1:
            raise ValueError("Real GT joints must have shape [T,H,21,3]")
        frames, hands = joints.shape[:2]
        valid = np.asarray(source["valid"])
        if valid.shape == (frames, hands):
            valid = np.broadcast_to(valid[..., None], (frames, hands, 21))
        if valid.shape != (frames, hands, 21) or not np.isin(valid, (0, 1)).all():
            raise ValueError("GT validity must align with hands or joints")
        joint_valid = valid.astype(bool) & np.isfinite(joints).all(-1) & (joints[..., 2] > 0)
        frame_indices = np.asarray(source["source_frame_indices"])
        if (frame_indices.shape != (frames,) or frame_indices.dtype.kind not in "iu"
                or (np.diff(frame_indices) <= 0).any()):
            raise ValueError("GT source frame indices must be unique increasing integers")
        times = np.asarray(source["timestamps_s"], dtype=np.float64)
        if times.shape != (frames,) or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("GT timestamps must be finite and strictly increasing")
        names = strings(source["joint_names"], 21, "joint_names")
        sides = strings(source["hand_sides"], hands, "hand_sides")
        if any(side not in ("left", "right") for side in sides):
            raise ValueError("GT hand sides must be explicitly left or right")
        K_gt = np.asarray(source["K"], dtype=np.float64) if "K" in source.files else None
        joints2d_annotation = np.asarray(source["joints_2d"], dtype=np.float32).copy() if "joints_2d" in source.files else None
    edges = named_edges(names)
    calibration = json.loads(intrinsics_path.read_text())
    if (calibration.get("schema_version") != "egofound3r.root_intrinsics_input.v1"
            or calibration.get("camera_model") != "pinhole"
            or calibration.get("coordinate_frame") != "rgb_camera"):
        raise ValueError("Expected the verified RGB pinhole calibration sidecar")
    K = np.asarray(calibration["K"], dtype=np.float64)
    if K.shape == (3, 3):
        K = np.broadcast_to(K, (frames, 3, 3)).copy()
    if K.shape != (frames, 3, 3) or not np.isfinite(K).all() or (K[:, 0, 0] <= 0).any() or (K[:, 1, 1] <= 0).any():
        raise ValueError("Calibration must provide one static K or one K per frame")
    if not np.allclose(K[:, 2], [0, 0, 1], atol=1e-8) or (np.abs(K[:, 0, 1]) > 1e-8).any() or (np.abs(K[:, 1, 0]) > 1e-8).any():
        raise ValueError("Calibration is not a zero-skew pinhole matrix")
    if not np.asarray(calibration.get("valid", True), dtype=bool).all():
        raise ValueError("Calibration must be marked valid")
    size = calibration["image_size_hw"]
    if not isinstance(size, list) or len(size) != 2 or any(type(x) is not int or x <= 0 for x in size):
        raise ValueError("Calibration must declare original RGB image_size_hw")
    if K_gt is not None and (K_gt.shape not in ((3, 3), (frames, 3, 3)) or not np.allclose(K, K_gt, atol=1e-6, rtol=1e-7)):
        raise ValueError("Selected RGB intrinsics do not match this camera's GT calibration")
    xyz_pixel = np.einsum("tij,thnj->thni", K, joints)
    projected = np.full(joints.shape[:-1] + (2,), np.nan, dtype=np.float32)
    projected[joint_valid] = xyz_pixel[..., :2][joint_valid] / xyz_pixel[..., 2:3][joint_valid]
    projection_error = None
    if joints2d_annotation is not None:
        if joints2d_annotation.shape != (frames, hands, 21, 2):
            raise ValueError("GT 2D annotations do not align with GT 3D joints")
        two_d_valid = (joint_valid & np.isfinite(joints2d_annotation).all(-1)
                       & ~np.all(joints2d_annotation == -1, axis=-1))
        if two_d_valid.any():
            projection_error = float(np.linalg.norm(projected[two_d_valid] - joints2d_annotation[two_d_valid], axis=-1).max())
            if projection_error > 1.0:
                raise ValueError(f"GT projection and GT 2D disagree by {projection_error:.3f}px: verify camera/calibration")
        joints2d = joints2d_annotation
        joints2d[~two_d_valid] = np.nan
        projection_source = "dataset_2d_ground_truth_verified_against_3d_and_K"
    else:
        joints2d = projected
        projection_source = "derived_projection_of_real_3d_ground_truth"
    joints[~joint_valid] = np.nan
    hand_valid = joint_valid.any(-1)
    output.mkdir(parents=True, exist_ok=True)
    save = lambda name, data, dtype="float32": write_array(output, name, data, dtype)
    labels = [("GT左手" if side == "left" else "GT右手") for side in sides]
    series = []
    for hand in range(hands):
        wrist_valid = joint_valid[:, hand, 0]
        depth = np.where(wrist_valid, joints[:, hand, 0, 2], np.nan)
        speed = np.full(frames, np.nan)
        if frames > 1:
            pair_valid = wrist_valid[1:] & wrist_valid[:-1]
            speed[1:] = np.where(pair_valid, np.linalg.norm(np.diff(joints[:, hand, 0], axis=0), axis=-1) / np.diff(times), np.nan)
        series.extend([
            {"id": f"gt_wrist_depth_{hand}", "label": labels[hand] + "腕部深度", "unit": "m", "values": values_json(depth)},
            {"id": f"gt_wrist_speed_{hand}", "label": labels[hand] + "腕部速度", "unit": "m/s", "values": values_json(speed)}])
    report = {
        "schema_version": "egofound3r.viewer.geometry.v1",
        "kind": "ground_truth", "data_role": DATA_ROLE,
        "label": "GT参考 · 仅评价", "is_model_prediction": False,
        "evaluation_only": True, "used_for_inference": False,
        "notice": "真实数据集关节标注，仅用于评价参考；不是 EgoFound3R 的模型预测。",
        "frame_count": frames, "hand_count": hands, "vertex_count": 0, "joint_count": 21,
        "coordinate_convention": "opencv", "coordinate_space": "camera",
        "coordinate_system": "该路 RGB 相机坐标 · m", "units": "m",
        "hand_order": sides, "hand_labels": labels, "joint_names": names,
        "joint_edges": edges, "joint_root_index": 0,
        "joints": save("gt_joints", joints), "valid": save("gt_hand_valid", hand_valid, "uint8"),
        "joint_valid": save("gt_joint_valid", joint_valid, "uint8"),
        "joints2d": save("gt_joints2d", joints2d),
        "joints2d_source": projection_source,
        "projection_space": "camera", "projection_K": K.tolist(),
        "projection_image_size_hw": size,
        "wrist_trajectory": save("gt_wrist_trajectory", joints[:, :, 0, :]),
        "trajectory_source": "exact GT wrist joint at each annotated timestamp; no interpolation or smoothing",
        "frame_indices": frame_indices.astype(int).tolist(),
        "timestamps_s": times.tolist(),
        "fps": float(1 / np.median(np.diff(times))) if frames > 1 else None,
        "series": series,
        "source": {"type": "ground_truth", "gt_filename": gt_path.name,
                   "gt_sha256": sha256(gt_path), "intrinsics_sha256": sha256(intrinsics_path),
                   "rgb_video_sha256": calibration.get("video_sha256"),
                   "used_fields": ["joints", "valid", "joints_2d", "source_frame_indices", "timestamps_s",
                                   "units", "coordinate_frame", "hand_sides", "joint_names", "K"],
                   "gt_used_only_for_evaluation_reference": True},
        "verification": {"gt_valid_joint_count": int(joint_valid.sum()),
                         "gt_valid_hand_frame_count": int(hand_valid.sum()),
                         "gt_2d_vs_3d_projection_max_error_px": projection_error,
                         "is_model_accuracy_metric": False}
    }
    final = output / "geometry.json"
    temporary = final.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(final)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    args = parser.parse_args()
    report = export_reference(args.gt, args.output, args.intrinsics)
    print(json.dumps({"geometry": str(args.output / "geometry.json"), "data_role": DATA_ROLE,
                      "frame_count": report["frame_count"], "hand_count": report["hand_count"],
                      "is_model_prediction": False, "verification": report["verification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

