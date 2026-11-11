#!/usr/bin/env python3
"""Export REAL EgoFound3R predictions to the static comparison viewer.

No checkpoint, model inference, GT, or synthetic-data path exists here.
Input is the official inference_output.pt. The native 195 predictions remain
available upstream; 778 vertices are a fixed official interpolation, not MANO
parameter regression. Invalid and filled outputs have separate masks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from official_contract import (JOINT_EDGES, UPSTREAM, UPSTREAM_COMMIT, array,
    bool_mask, camera_contract, camera_to_world, contact_distance, dense_hand,
    finite_json, hand_contract, load_prediction, probability, sha256)


DTYPES = {"float32": "<f4", "uint8": "u1", "uint32": "<u4"}


def write_array(directory: Path, name: str, values, dtype: str) -> dict:
    a = np.ascontiguousarray(values, dtype=DTYPES[dtype])
    target = directory / (name + ".bin")
    temporary = target.with_suffix(".bin.tmp")
    a.tofile(temporary)
    temporary.replace(target)
    return {"url": target.name, "dtype": dtype, "shape": list(a.shape),
            "byte_length": int(a.nbytes), "byte_order": "little"}


def video_info(path: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open RGB video: {path}")
        result = {"height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                  "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                  "fps": float(cap.get(cv2.CAP_PROP_FPS))}
        if min(result["height"], result["width"]) <= 0:
            raise ValueError("RGB video has invalid dimensions")
        return result
    finally:
        cap.release()


def read_rgb_frames(path: Path, positions: list[int], image_size_hw: tuple[int, int]) -> dict:
    """Decode sequentially, avoiding inaccurate keyframe seeks."""
    import cv2
    required = set(positions)
    if not required:
        return {}
    cap = cv2.VideoCapture(str(path))
    result = {}
    try:
        if not cap.isOpened():
            raise ValueError("Could not open RGB video")
        for index in range(max(required) + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"RGB video ended before required frame {index}")
            if index in required:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result[index] = cv2.resize(rgb, (image_size_hw[1], image_size_hw[0]),
                                           interpolation=cv2.INTER_LINEAR)
    finally:
        cap.release()
    return result


def project_joints(joints_cam: np.ndarray, K: np.ndarray,
                   valid: np.ndarray, K_valid: np.ndarray) -> np.ndarray:
    if K.shape != (len(joints_cam), 3, 3):
        raise ValueError("Projection K must align with every hand frame")
    homogeneous = np.einsum("tij,thnj->thni", K, joints_cam)
    good = np.isfinite(homogeneous).all(-1) & (joints_cam[..., 2] > 1e-6)
    good &= valid[..., None] & K_valid[:, None, None]
    uv = np.full(joints_cam.shape[:-1] + (2,), np.nan, dtype=np.float32)
    uv[good] = homogeneous[..., :2][good] / homogeneous[..., 2:3][good]
    return uv


def build_scene(payload: dict, cameras: dict, *, coordinate: str,
                stride: int, max_points: int, min_confidence: float,
                max_depth: float, rgb_frames: dict[int, np.ndarray] | None = None) -> dict:
    m = payload["metric_predictions"]
    d = m["depth"]
    if d.get("unit") != "meter":
        raise ValueError("Scene depth must explicitly be in meters")
    depth = array(d["value_metric"], dtype=np.float32)
    confidence = array(d["confidence"], dtype=np.float32)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth, confidence = depth[..., 0], confidence[..., 0]
    if depth.ndim != 3 or confidence.shape != depth.shape:
        raise ValueError("Expected matching depth and confidence [T_G,H,W]")
    ng, height, width = depth.shape
    if tuple(payload["image_size_hw"]) != (height, width):
        raise ValueError("Depth size differs from the pixel frame of native scene K")
    K = array(m["intrinsics"]["K_global"], dtype=np.float64)
    if K.shape != (ng, 3, 3):
        raise ValueError("Scene K does not align with depth anchors")
    frame_ids = array(d["frame_indices"], dtype=np.int64)
    if frame_ids.shape != (ng,):
        raise ValueError("Depth frame IDs do not align with depth anchors")
    frame_lookup = {int(frame): i for i, frame in enumerate(cameras["frame_ids"])}
    if any(int(i) not in frame_lookup for i in frame_ids):
        raise ValueError("Depth anchor is missing from the camera R-frame axis")
    positions = np.array([frame_lookup[int(i)] for i in frame_ids], dtype=np.int64)
    frame_valid = bool_mask(d["valid"], (ng,), "depth.valid")
    frame_valid &= bool_mask(m["intrinsics"]["K_global_valid"], (ng,), "scene K valid")
    if coordinate == "world":
        frame_valid &= cameras["valid"][positions]
    yy, xx = np.meshgrid(np.arange(0, height, stride), np.arange(0, width, stride), indexing="ij")
    yy, xx = yy.ravel(), xx.ravel()
    if len(xx) > max_points:
        selection = np.linspace(0, len(xx) - 1, max_points).round().astype(np.int64)
        yy, xx = yy[selection], xx[selection]
    points = np.full((ng, len(xx), 3), np.nan, dtype=np.float32)
    colors = np.full((ng, len(xx), 3), 180, dtype=np.uint8)
    valid = np.zeros((ng, len(xx)), dtype=bool)
    rays_pixel = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    for i in range(ng):
        if rgb_frames is not None:
            colors[i] = rgb_frames[int(positions[i])][yy, xx]
        if not frame_valid[i]:
            continue
        if not np.isfinite(K[i]).all() or abs(np.linalg.det(K[i])) < 1e-12:
            raise ValueError("A valid scene frame has invalid intrinsics")
        z = depth[i, yy, xx]
        conf = confidence[i, yy, xx]
        good = np.isfinite(z) & (z > 0) & (z <= max_depth) & np.isfinite(conf) & (conf >= min_confidence)
        rays = rays_pixel @ np.linalg.inv(K[i]).T
        xyz = rays * (z / rays[:, 2])[:, None]
        if coordinate == "world":
            c2w = cameras["c2w"][positions[i]]
            xyz = xyz @ c2w[:3, :3].T + c2w[:3, 3]
        good &= np.isfinite(xyz).all(-1)
        points[i, good] = xyz[good]
        valid[i] = good
    return {"points": points, "colors": colors, "valid": valid,
            "frame_indices": positions.tolist(), "source_frame_ids": frame_ids.tolist(),
            "point_count": len(xx), "frame_count": ng,
            "sampling": {"pixel_stride": stride, "max_points_per_anchor": max_points,
                         "min_native_confidence": min_confidence, "max_depth_m": max_depth},
            "color_source": "input_rgb_video" if rgb_frames is not None else "neutral_gray",
            "intrinsics_source": "official_predicted_scene_K_global",
            "interpolation": "none; native G anchors only",
            "invalid_points": "NaN positions and valid=0; renderer must omit"}


def series_data(payload: dict, h: dict, valid: np.ndarray) -> list:
    fps = float(payload.get("r_fps", payload.get("fps", 0)))
    output = []
    for slot in range(2):
        values = np.where(valid[:, slot], h["joints"][:, slot, 0, 2], np.nan)
        output.append({"id": f"root_depth_slot{slot}", "label": f"手槽 {slot} 根部深度", "unit": "m", "values": finite_json(values)})
        root = h["joints"][:, slot, 0]
        speed = np.full(len(root), np.nan)
        if fps > 0 and len(root) > 1:
            good = valid[1:, slot] & valid[:-1, slot]
            speed[1:] = np.where(good, np.linalg.norm(np.diff(root, axis=0), axis=-1) * fps, np.nan)
        output.append({"id": f"root_camera_speed_slot{slot}", "label": f"手槽 {slot} 相机坐标根部速度", "unit": "m/s", "values": finite_json(speed)})
    output.append({"id": "valid_hands", "label": "有效手数量（排除填补）", "unit": "hands", "values": h["valid"].sum(axis=1).tolist()})
    output.append({"id": "filled_hands", "label": "官方填补手数量", "unit": "hands", "values": h["fill_mask"].sum(axis=1).tolist()})
    return output


def export_payload(payload: dict, output: Path, *, coordinate="world", official_repo=None,
                   rgb_video=None, scene_stride=8, max_scene_points=4096,
                   min_depth_confidence=0.0, max_depth=20.0, include_filled=False,
                   hand_order="unknown", no_scene=False, overwrite=False) -> dict:
    output = Path(output)
    if (output / "geometry.json").exists() and not overwrite:
        raise FileExistsError("geometry.json exists; use --overwrite explicitly")
    h = hand_contract(payload)
    dense = dense_hand(payload, official_repo)
    frames = len(h["joints"])
    cameras = camera_contract(payload, frames)
    valid = (h["display_valid"] if include_filled else h["valid"]).copy()
    dense_finite = np.isfinite(dense["vertices"]).all(axis=(-2, -1))
    valid &= dense_finite
    display_valid = h["display_valid"] & dense_finite
    vertices, joints = dense["vertices"], h["joints"]
    if coordinate == "world":
        valid &= cameras["valid"][:, None]
        display_valid &= cameras["valid"][:, None]
        vertices = camera_to_world(vertices, cameras["c2w"])
        joints = camera_to_world(joints, cameras["c2w"])
    elif coordinate != "camera":
        raise ValueError("coordinate must be camera or world")
    model_size = tuple(int(x) for x in payload["image_size_hw"])
    input_info = video_info(Path(rgb_video)) if rgb_video is not None else None
    original_size = (input_info["height"], input_info["width"]) if input_info else tuple(payload.get("provenance", {}).get("source_video_size_hw") or model_size)
    if input_info and input_info["frame_count"] > 0 and input_info["frame_count"] < frames:
        raise ValueError("RGB video has fewer frames than the inference payload")
    projection = payload.get("hand_projection_K_full")
    projection_valid = payload.get("hand_projection_K_valid_full")
    if projection is None:
        raise ValueError("Missing official hand_projection_K_full: do not substitute scene K for hand K")
    K = array(projection, dtype=np.float64).copy()
    K_valid = bool_mask(projection_valid, (frames,), "hand projection K valid")
    K[:, 0] *= original_size[1] / model_size[1]
    K[:, 1] *= original_size[0] / model_size[0]
    joints2d = project_joints(h["joints"], K, display_valid, K_valid)
    scene = None
    if not no_scene:
        depth_ids = array(payload["metric_predictions"]["depth"]["frame_indices"], dtype=np.int64)
        frame_lookup = {int(frame): i for i, frame in enumerate(cameras["frame_ids"])}
        positions = [frame_lookup[int(i)] for i in depth_ids]
        rgb_frames = read_rgb_frames(Path(rgb_video), positions, model_size) if rgb_video is not None else None
        scene = build_scene(payload, cameras, coordinate=coordinate, stride=scene_stride,
                            max_points=max_scene_points, min_confidence=min_depth_confidence,
                            max_depth=max_depth, rgb_frames=rgb_frames)
    output.mkdir(parents=True, exist_ok=True)
    def save(name, value, dtype="float32"):
        return write_array(output, name, value, dtype)
    face_spec = save("hand_faces", dense["faces"], "uint32")
    source_indices = array(payload.get("source_frame_indices", np.arange(frames)), dtype=np.int64)
    if source_indices.shape != (frames,):
        raise ValueError("source_frame_indices do not align with the hand R axis")
    model_anchor_ids = array(payload.get("high_frame_indices", []), dtype=np.int64)
    model_anchor = np.isin(cameras["frame_ids"], model_anchor_ids)
    labels = ["native slot 0", "native slot 1"] if hand_order == "unknown" else hand_order.split(",")
    report = {"schema_version": "egofound3r.viewer.geometry.v1", "frame_count": frames,
        "hand_count": 2, "vertex_count": 778, "joint_count": 21,
        "coordinate_convention": "opencv", "coordinate_space": coordinate, "units": "m",
        "world_scope": cameras["world_scope"] if coordinate == "world" else "per_frame_camera",
        "hand_labels": labels, "hand_order": hand_order,
        "joint_order": "official_native_openpose_skeleton", "joint_root_index": 0,
        "joint_names": ["wrist"] + [f"joint_{i}" for i in range(1, 21)],
        "joint_edges": JOINT_EDGES,
        "joint_order_note": "Native order is preserved. Source names OPENPOSE_HAND_BONES; anatomical finger/side labels are not guessed.",
        "vertices": save("vertices", vertices), "joints": save("joints", joints),
        "valid": save("valid", valid, "uint8"),
        "official_valid": save("official_valid", h["official_valid"], "uint8"),
        "display_valid": save("display_valid", display_valid, "uint8"),
        "fill_mask": save("fill_mask", h["fill_mask"], "uint8"),
        "camera_valid": save("camera_valid", cameras["valid"], "uint8"),
        "network_anchor": save("network_anchor", model_anchor, "uint8"),
        "faces": {"left": face_spec, "right": dict(face_spec)},
        "face_winding": "official common topology; render both sides; keys do not establish anatomical hand-slot order",
        "joints2d": save("joints2d", joints2d),
        "projection_image_size_hw": list(original_size), "projection_space": "input_rgb_pixels",
        "projection_K": save("projection_K", K),
        "projection_source": "official_hand_projection_K_full_rescaled_to_input_rgb",
        "frame_indices": source_indices.tolist(),
        "timestamps_s": finite_json(array(payload.get("timestamps_sec", np.arange(frames) / float(payload.get("fps", 30))), dtype=np.float64)),
        "fps": float(payload.get("r_fps", payload.get("fps", 30))),
        "network_fps": float(payload.get("h_fps", 0)), "scene_fps": float(payload.get("g_fps", 0)),
        "validity_policy": "official_display_including_fill" if include_filled else "official_metric_excluding_fill",
        "upsampling": dense["provenance"], "series": series_data(payload, h, valid),
        "source": {"model": "EgoFound3R", "official_repository": UPSTREAM,
                   "audited_commit": UPSTREAM_COMMIT,
                   "inference_commit": payload.get("provenance", {}).get("git_commit"),
                   "hand_K_source": payload.get("root_intrinsics", {}).get("source", "unspecified")},
        "counts": {"valid_hand_frames": int(h["valid"].sum()), "filled_hand_frames": int(h["fill_mask"].sum()),
                   "trusted_renderable_hand_frames": int(valid.sum()), "displayable_hand_frames": int(display_valid.sum()),
                   "network_anchor_frames": int(model_anchor.sum())}}
    for key, prefix in (("contact", "vertex_contact"), ("visibility", "vertex_visibility")):
        attr = probability(dense["hand"], prefix)
        if attr is not None:
            if attr.shape != (frames, 2, 778):
                raise ValueError(f"{key} attribute does not align with 778 vertices")
            report[key] = save(key, attr)
    distance = contact_distance(dense["hand"])
    if distance is not None:
        if distance.shape != (frames, 2, 778):
            raise ValueError("Contact distance does not align with 778 vertices")
        report["contact_distance"] = save("contact_distance", distance)
    for key in ("joint_contact", "joint_visibility"):
        p = probability(h["hand"], key)
        if p is not None:
            if p.shape != (frames, 2, 21):
                raise ValueError(f"{key} does not align with joints")
            report[key] = save(key, p)
    if scene is not None:
        report["scene"] = {k: v for k, v in scene.items() if k not in ("points", "colors", "valid")}
        report["scene"].update(points=save("scene_points", scene["points"]),
                               colors=save("scene_colors", scene["colors"], "uint8"),
                               valid=save("scene_valid", scene["valid"], "uint8"))
    if rgb_video is not None:
        report["source"]["rgb_video_sha256"] = sha256(Path(rgb_video))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb-video", type=Path)
    parser.add_argument("--coordinate", choices=("camera", "world"), default="world")
    parser.add_argument("--official-repo", type=Path)
    parser.add_argument("--scene-stride", type=int, default=8)
    parser.add_argument("--max-scene-points", type=int, default=4096)
    parser.add_argument("--min-depth-confidence", type=float, default=0.0)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--include-filled", action="store_true")
    parser.add_argument("--hand-order", choices=("unknown", "left,right", "right,left"), default="unknown",
                        help="Only set after independently verifying native model hand slots")
    parser.add_argument("--no-scene", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if min(args.scene_stride, args.max_scene_points) < 1 or args.max_depth <= 0:
        parser.error("Scene sampling and depth limits must be positive")
    if not args.input.is_file():
        parser.error("Official inference_output.pt does not exist; predictions cannot be exported")
    payload = load_prediction(args.input)
    report = export_payload(payload, args.output, coordinate=args.coordinate,
        official_repo=args.official_repo, rgb_video=args.rgb_video, scene_stride=args.scene_stride,
        max_scene_points=args.max_scene_points, min_depth_confidence=args.min_depth_confidence,
        max_depth=args.max_depth, include_filled=args.include_filled, hand_order=args.hand_order,
        no_scene=args.no_scene, overwrite=args.overwrite)
    report["source"].update(inference_sha256=sha256(args.input), inference_filename=args.input.name)
    final = args.output / "geometry.json"
    temporary = final.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(finite_json(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(final)
    print(json.dumps({"geometry": str(final), "frame_count": report["frame_count"], "counts": report["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
