#!/usr/bin/env python3
"""Evaluate official camera-space hands against explicitly supplied GT.

GT is never searched for or used by inference/export. --gt is a separate NPZ
with joints [T,H,21,3], valid [T,H,21] (or [T,H]), source_frame_indices [T],
units ('m' or 'mm') and coordinate_frame='camera'. Explicit pred_to_gt_hand[2]
and pred_to_gt_joint[21] are required in the NPZ or --mapping-json. A -1 hand
mapping means that prediction slot has no annotated corresponding GT hand.
No best-of-two selection, per-frame reassignment, or guessed semantic order.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from official_contract import array, finite_json, hand_contract, load_prediction, sha256


def scalar_text(value, name: str) -> str:
    a = np.asarray(value)
    if a.size != 1 or a.dtype.kind not in ("U", "S"):
        raise ValueError(f"{name} must be an explicit scalar string")
    v = a.item()
    return v.decode() if isinstance(v, bytes) else str(v)


def average(values, mask=None):
    v = np.asarray(values, dtype=float)
    valid = np.isfinite(v)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    return float(v[valid].mean()) if valid.any() else None


def similarity_align(prediction: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """One proper-rotation Sim(3) for one hand in one frame."""
    if len(prediction) < 3:
        return None
    x, y = prediction.astype(np.float64), target.astype(np.float64)
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    if np.linalg.matrix_rank(xc, tol=1e-8) < 2 or np.linalg.matrix_rank(yc, tol=1e-8) < 2:
        return None
    u, s, vt = np.linalg.svd(xc.T @ yc)
    orientation = np.ones(3)
    orientation[-1] = 1 if np.linalg.det(u @ vt) >= 0 else -1
    rotation = u @ np.diag(orientation) @ vt
    scale = float((s * orientation).sum() / np.square(xc).sum())
    if not np.isfinite(scale) or scale <= 0:
        return None
    return scale * xc @ rotation + my


def evaluate(payload: dict, gt: dict, mapping: dict | None = None, pck_thresholds=(20.0, 50.0)) -> dict:
    mapping = mapping or {}
    hand = hand_contract(payload)
    pred = hand["joints"].astype(np.float64)
    if scalar_text(gt["coordinate_frame"], "coordinate_frame") != "camera":
        raise ValueError("Only camera-space GT is accepted; world GT must be explicitly transformed outside inference")
    units = scalar_text(gt["units"], "units")
    if units not in ("m", "mm"):
        raise ValueError("GT units must explicitly be m or mm")
    truth = np.asarray(gt["joints"], dtype=np.float64) / (1000 if units == "mm" else 1)
    if truth.ndim != 4 or truth.shape[-1] != 3:
        raise ValueError("GT joints must have shape [T,H,J,3]")
    gvalid = np.asarray(gt["valid"])
    if gvalid.shape == truth.shape[:2]:
        gvalid = np.broadcast_to(gvalid[..., None], truth.shape[:3]).copy()
    if gvalid.shape != truth.shape[:3] or not np.isin(gvalid, (0, 1)).all():
        raise ValueError("GT valid must align with hand or joint axes and contain bool values")
    gvalid = gvalid.astype(bool) & np.isfinite(truth).all(-1)
    hmap_value = mapping.get("pred_to_gt_hand", gt.get("pred_to_gt_hand"))
    jmap_value = mapping.get("pred_to_gt_joint", gt.get("pred_to_gt_joint"))
    if hmap_value is None or jmap_value is None:
        raise ValueError("Explicit pred_to_gt_hand and pred_to_gt_joint are required; native hand slots are not guessed")
    hmap, jmap = np.asarray(hmap_value), np.asarray(jmap_value)
    if hmap.dtype.kind not in "iu" or jmap.dtype.kind not in "iu":
        raise ValueError("Hand and joint mapping must be integer arrays")
    if hmap.shape != (2,) or ((hmap < -1) | (hmap >= truth.shape[1])).any():
        raise ValueError("Invalid pred_to_gt_hand mapping")
    if len(set(hmap[hmap >= 0].tolist())) != int((hmap >= 0).sum()):
        raise ValueError("Two prediction slots cannot map to the same GT hand")
    if jmap.shape != (21,) or (jmap < 0).any() or (jmap >= truth.shape[2]).any() or len(set(jmap.tolist())) != 21:
        raise ValueError("pred_to_gt_joint must explicitly map all 21 unique native joints")
    annotated_sides = set(np.flatnonzero(gvalid.any(axis=(0, 2))).tolist())
    if not annotated_sides.issubset(set(hmap[hmap >= 0].tolist())):
        raise ValueError("Mapping omits an annotated GT hand, which would hide failures")
    pids = array(payload.get("source_frame_indices", np.arange(len(pred))), dtype=np.int64)
    gids = np.asarray(gt["source_frame_indices"])
    if pids.shape != (len(pred),) or gids.shape != (len(truth),) or gids.dtype.kind not in "iu":
        raise ValueError("Explicit GT and prediction source frame indices must align")
    if len(np.unique(pids)) != len(pids) or len(np.unique(gids)) != len(gids):
        raise ValueError("Frame indices must be unique; duplicate frames would bias the score")
    p_lookup = {int(fid): i for i, fid in enumerate(pids)}
    # Unpredicted GT frames remain in the denominator; arrays stay unavailable.
    shape = (len(truth), 2, 21)
    target = np.full(shape + (3,), np.nan)
    prediction = np.full_like(target, np.nan)
    eligible = np.zeros(shape, bool)
    evaluated = np.zeros(shape, bool)
    filled_gt = np.zeros((len(truth), 2), bool)
    for g, fid in enumerate(gids):
        p = p_lookup.get(int(fid))
        for slot, gt_slot in enumerate(hmap):
            if gt_slot < 0:
                continue
            target[g, slot] = truth[g, gt_slot, jmap]
            eligible[g, slot] = gvalid[g, gt_slot, jmap]
            if p is not None:
                prediction[g, slot] = pred[p, slot]
                filled_gt[g, slot] = hand["fill_mask"][p, slot]
                evaluated[g, slot] = eligible[g, slot] & hand["valid"][p, slot] & np.isfinite(pred[p, slot]).all(-1)
    raw = np.linalg.norm(prediction - target, axis=-1) * 1000
    root_valid = evaluated[..., 0]
    rr = np.linalg.norm((prediction - prediction[..., :1, :]) - (target - target[..., :1, :]), axis=-1) * 1000
    rr_valid = evaluated & root_valid[..., None]
    pa = np.full(shape, np.nan)
    pa_frames = 0
    for f in range(len(truth)):
        for slot in range(2):
            mask = evaluated[f, slot]
            aligned = similarity_align(prediction[f, slot, mask], target[f, slot, mask])
            if aligned is not None:
                pa[f, slot, mask] = np.linalg.norm(aligned - target[f, slot, mask], axis=-1) * 1000
                pa_frames += 1
    gt_points, valid_points = int(eligible.sum()), int(evaluated.sum())
    result = {"schema_version": "egofound3r.reproduction.camera_metrics.v1",
        "coordinate_frame": "camera", "error_unit": "mm",
        "alignment": {"raw": "none", "root_relative": "subtract own wrist (native joint 0)",
                      "pa": "one proper-rotation Sim(3) per hand per frame"},
        "validity_policy": "official canonical valid AND finite AND NOT official missing-fill",
        "mapping": {"pred_to_gt_hand": hmap.tolist(), "pred_to_gt_joint": jmap.tolist(),
                    "evidence": mapping.get("evidence", "explicitly supplied mapping; no automatic assignment")},
        "coverage": {"gt_frames": len(truth), "prediction_frames": len(pred),
                     "gt_joint_count": gt_points, "evaluated_joint_count": valid_points,
                     "fraction": valid_points / gt_points if gt_points else None,
                     "unavailable_joint_count": gt_points - valid_points,
                     "gt_hand_frames": int(eligible.any(-1).sum()),
                     "evaluated_hand_frames": int(evaluated.any(-1).sum()),
                     "filled_annotated_hand_frames": int((filled_gt & eligible.any(-1)).sum()),
                     "pa_evaluated_hand_frames": pa_frames,
                     "pa_evaluated_joint_count": int(np.isfinite(pa).sum()),
                     "root_relative_evaluated_joint_count": int(rr_valid.sum())},
        "metrics": {"mpjpe_raw_mm": average(raw, evaluated),
                    "mpjpe_root_relative_mm": average(rr, rr_valid),
                    "mpjpe_pa_mm": average(pa),
                    "wrist_position_error_mm": average(raw[..., 0], root_valid)},
        "pck": {}, "per_hand_slot": [],
        "series": [{"id": "mpjpe_raw_mm", "label": "绝对关节误差", "unit": "mm",
                    "frame_indices": gids.tolist(),
                    "values": [average(raw[t], evaluated[t]) for t in range(len(truth))]},
                   {"id": "mpjpe_rr_mm", "label": "减去手腕后的关节误差", "unit": "mm",
                    "frame_indices": gids.tolist(),
                    "values": [average(rr[t], rr_valid[t]) for t in range(len(truth))]}]}
    for threshold in pck_thresholds:
        correct = int(((raw <= threshold) & evaluated).sum())
        result["pck"][f"{threshold:g}mm"] = {
            "all_gt_including_failures": correct / gt_points if gt_points else None,
            "conditional_on_valid_prediction": correct / valid_points if valid_points else None,
            "correct_joint_count": correct}
    for slot in range(2):
        result["per_hand_slot"].append({"native_slot": slot, "gt_slot": int(hmap[slot]),
            "gt_joint_count": int(eligible[:, slot].sum()),
            "evaluated_joint_count": int(evaluated[:, slot].sum()),
            "mpjpe_raw_mm": average(raw[:, slot], evaluated[:, slot]),
            "mpjpe_root_relative_mm": average(rr[:, slot], rr_valid[:, slot]),
            "mpjpe_pa_mm": average(pa[:, slot])})
    return finite_json(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--mapping-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pck-mm", type=float, nargs="+", default=[20., 50.])
    args = parser.parse_args()
    if any(not np.isfinite(x) or x <= 0 for x in args.pck_mm):
        parser.error("PCK thresholds must be positive finite millimeters")
    payload = load_prediction(args.input)
    with np.load(args.gt, allow_pickle=False) as data:
        gt = {key: data[key] for key in data.files}
    mapping = json.loads(args.mapping_json.read_text()) if args.mapping_json else None
    report = evaluate(payload, gt, mapping, args.pck_mm)
    report["source"] = {"inference_sha256": sha256(args.input), "gt_sha256": sha256(args.gt),
                        "mapping_sha256": sha256(args.mapping_json) if args.mapping_json else None,
                        "gt_used_only_for_scoring": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"], "coverage": report["coverage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

