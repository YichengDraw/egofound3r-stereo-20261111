"""Strict readers for the released EgoFound3R inference payload.

Audited against MINT-SJTU/EgoFound3R at
264f4df9fd42536aae4dba982d18272cc3334077. These helpers never run a model.
The upstream canonical hand contract is CAMERA coordinates in METERS;
the camera matrices are world-to-camera. Keep the two intrinsic streams separate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

UPSTREAM_COMMIT = "264f4df9fd42536aae4dba982d18272cc3334077"
UPSTREAM = f"https://github.com/MINT-SJTU/EgoFound3R/tree/{UPSTREAM_COMMIT}"
JOINT_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6),
               (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
               (0, 13), (13, 14), (14, 15), (15, 16), (0, 17),
               (17, 18), (18, 19), (19, 20)]


def array(value: Any, *, dtype=None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if "bfloat16" in str(value.dtype):
            value = value.float()
        value = value.numpy()
    result = np.asarray(value, dtype=dtype)
    if result.dtype == object:
        raise ValueError("Object arrays are not valid numeric inference outputs")
    return result


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def load_prediction(path: Path) -> dict:
    import torch  # Only needed to deserialize actual official output.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("metric_predictions"), dict):
        raise ValueError("Expected official inference_output.pt with metric_predictions")
    return payload


def bool_mask(value, shape, name: str) -> np.ndarray:
    result = array(value)
    if result.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {result.shape}")
    if result.dtype.kind != "b" and not np.isin(result, (0, 1)).all():
        raise ValueError(f"{name} must contain boolean values")
    return result.astype(bool, copy=True)


def hand_contract(payload: dict) -> dict:
    m = payload["metric_predictions"]
    h = m["hand"]
    if h.get("unit") != "meter" or h.get("coordinate_frame") != "camera":
        raise ValueError("Canonical hand must explicitly declare camera-space meters")
    joints = array(h["joint_xyz_metric"], dtype=np.float32)
    vertices = array(h["vertex_xyz_metric"], dtype=np.float32)
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
        raise ValueError(f"Expected [T,2,21,3] joints, got {joints.shape}")
    if vertices.shape != (len(joints), 2, 195, 3):
        raise ValueError(f"Expected native [T,2,195,3] vertices, got {vertices.shape}")
    shape = joints.shape[:2]
    valid = bool_mask(h["valid"], shape, "hand.valid")
    fill = bool_mask(payload.get("hand_missing_fill_full", np.zeros(shape, bool)), shape, "fill_mask")
    display_valid = bool_mask(payload.get("hand_display_valid_full", valid), shape, "display_valid")
    finite = np.isfinite(joints).all(axis=(-2, -1)) & np.isfinite(vertices).all(axis=(-2, -1))
    return {"hand": h, "joints": joints, "native_vertices": vertices,
            "official_valid": valid, "fill_mask": fill,
            "valid": valid & ~fill & finite,
            "display_valid": display_valid & finite}


def probability(hand: dict, prefix: str) -> np.ndarray | None:
    if prefix + "_probability" in hand:
        p = array(hand[prefix + "_probability"], dtype=np.float32)
    elif prefix + "_logits" in hand:
        x = array(hand[prefix + "_logits"], dtype=np.float32)
        p = 1 / (1 + np.exp(-np.clip(x, -80, 80)))
    else:
        return None
    finite = np.isfinite(p)
    if ((p[finite] < -1e-5) | (p[finite] > 1 + 1e-5)).any():
        raise ValueError(f"{prefix} probabilities are outside [0,1]")
    return p


def dense_hand(payload: dict, official_repo: Path | None = None) -> dict:
    """Read actual 778 outputs, or use the official fixed interpolation asset."""
    h = payload["metric_predictions"].get("hand_778", payload.get("hand_778"))
    base = hand_contract(payload)
    if h is not None:
        if h.get("unit") != "meter" or h.get("coordinate_frame") != "camera":
            raise ValueError("hand_778 must explicitly declare camera-space meters")
        vertices = array(h["vertex_xyz_metric"], dtype=np.float32)
        faces = array(h["marker_faces"], dtype=np.int64)
        provenance = "official_payload_hand_778"
    else:
        if official_repo is None:
            raise ValueError("No hand_778: infer with --upsample-mano, or pass --official-repo for its fixed asset")
        asset_path = Path(official_repo) / "egohandmetric_prompt/data/mano_upsampling/mano_195_to_778.npz"
        with np.load(asset_path, allow_pickle=False) as asset:
            neighbors = np.asarray(asset["neighbor_indices"], dtype=np.int64)
            weights = np.asarray(asset["geometry_weights"], dtype=np.float32)
            ids = np.asarray(asset["source_vertex_ids"], dtype=np.int64)
            faces = np.asarray(asset["faces"], dtype=np.int64)
        if neighbors.shape != weights.shape or neighbors.shape[0] != 778 or ids.shape != (195,):
            raise ValueError("Official MANO upsampling asset has incompatible dimensions")
        if neighbors.min() < 0 or neighbors.max() >= 195 or len(set(ids.tolist())) != 195:
            raise ValueError("Invalid official interpolation indices")
        if ids.min() < 0 or ids.max() >= 778:
            raise ValueError("Invalid source MANO vertex IDs")
        if "vertex_ids" in payload and not np.array_equal(array(payload["vertex_ids"]), ids):
            raise ValueError("Prediction marker ordering disagrees with the official upsampling asset")
        native = base["native_vertices"]
        center = native.mean(axis=-2, keepdims=True)
        local = native - center
        vertices = center + (local[..., neighbors, :] * weights[..., None]).sum(axis=-2)
        vertices[..., ids, :] = native
        scalar_weights = np.maximum(weights, 0)
        scalar_weights /= scalar_weights.sum(axis=-1, keepdims=True)
        h = {}
        for attr in ("contact", "visibility"):
            p = probability(base["hand"], "vertex_" + attr)
            if p is not None:
                dense = (p[..., neighbors] * scalar_weights).sum(axis=-1)
                dense[..., ids] = p
                h["vertex_" + attr + "_probability"] = np.clip(dense, 0, 1)
        native_distance = contact_distance(base["hand"])
        if native_distance is not None:
            dense = (native_distance[..., neighbors] * scalar_weights).sum(axis=-1)
            dense[..., ids] = native_distance
            h["vertex_contact_distance_metric"] = np.maximum(dense, 0)
        provenance = "official_fixed_195_to_778_interpolation:" + sha256(asset_path)
    if vertices.shape != (len(base["joints"]), 2, 778, 3):
        raise ValueError(f"Expected [T,2,778,3] hand_778, got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= 778:
        raise ValueError("Invalid 778-vertex face topology")
    return {"vertices": vertices, "faces": faces, "hand": h, "provenance": provenance}


def contact_distance(hand: dict) -> np.ndarray | None:
    if "vertex_contact_distance_metric" in hand:
        return array(hand["vertex_contact_distance_metric"], dtype=np.float32)
    if "vertex_contact_log_distance" in hand:
        return np.expm1(array(hand["vertex_contact_log_distance"], dtype=np.float32)).clip(0) / 1000
    return None


def camera_contract(payload: dict, frame_count: int) -> dict:
    c = payload["metric_predictions"]["camera"]
    if c.get("translation_unit") != "meter":
        raise ValueError("Camera translation must explicitly use meters")
    w2c = array(c["pose_w2c_metric"], dtype=np.float64)
    if w2c.shape != (frame_count, 4, 4):
        raise ValueError("Camera matrix sequence must align exactly with hand frames")
    valid = bool_mask(c["valid"], (frame_count,), "camera.valid")
    frame_ids = array(c["frame_indices"], dtype=np.int64)
    if frame_ids.shape != (frame_count,) or len(np.unique(frame_ids)) != frame_count:
        raise ValueError("Camera frame indices must be unique and match the R axis")
    finite = np.isfinite(w2c).all(axis=(-2, -1))
    if (valid & ~finite).any():
        raise ValueError("A camera marked valid contains nonfinite entries")
    if valid.any() and not np.allclose(w2c[valid, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid homogeneous camera matrix")
    c2w = np.repeat(np.eye(4)[None], frame_count, axis=0)
    c2w[valid] = np.linalg.inv(w2c[valid])
    return {"w2c": w2c, "c2w": c2w, "valid": valid, "frame_ids": frame_ids,
            "world_scope": c.get("world_scope", "unspecified")}


def camera_to_world(points: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    if points.shape[0] != c2w.shape[0] or points.shape[-1] != 3:
        raise ValueError("Point and camera axes do not match")
    return (np.einsum("tij,t...j->t...i", c2w[:, :3, :3], points)
            + c2w[:, :3, 3].reshape((len(points),) + (1,) * (points.ndim - 2) + (3,))).astype(np.float32)


def finite_json(value):
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value
