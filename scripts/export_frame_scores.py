#!/usr/bin/env python3
"""Export per-frame/per-anatomical-hand audit rows from existing predictions only.

No model is constructed. Aggregate counts and metrics must reproduce the already
published independent scores before these supplementary files are accepted.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from official_contract import array, hand_contract, load_prediction, sha256
from evaluate_predictions import average, finite_json, scalar_text, similarity_align


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, type=Path)
    root = ap.parse_args().root
    suite = json.loads((root / 'reports/suite_metrics.json').read_text())
    rows = []
    checks = []
    all_runs = []
    for run in suite['runs']:
        rid = run['run_id']
        src = run['source']
        assert sha256(Path(src['prediction_path'])) == src['inference_sha256']
        assert sha256(Path(src['gt_path'])) == src['gt_sha256']
        payload = load_prediction(Path(src['prediction_path']))
        hand = hand_contract(payload)
        pred = hand['joints'].astype(np.float64)
        ids = array(payload['source_frame_indices'], dtype=np.int64)
        anchors = set(array(payload['high_frame_indices']).tolist())
        with np.load(src['gt_path'], allow_pickle=False) as z:
            gt = {k: z[k] for k in z.files}
        assert scalar_text(gt['coordinate_frame'], 'coordinate_frame') == 'camera'
        unit = scalar_text(gt['units'], 'units')
        assert unit in ('m', 'mm')
        truth = gt['joints'].astype(np.float64) / (1000 if unit == 'mm' else 1)
        gv = gt['valid'].astype(bool)
        if gv.ndim == 2:
            gv = np.broadcast_to(gv[..., None], truth.shape[:-1])
        gv = gv & np.isfinite(truth).all(-1)
        gt_index = {int(fid): i for i, fid in enumerate(gt['source_frame_indices'])}
        mapping = run['primary']['mapping']
        hm = mapping['pred_to_gt_hand']
        jm = mapping['pred_to_gt_joint']
        assert set(ids.tolist()) == set(gt_index), 'Export must include all scoring frames'
        per_run = []
        for t, fid in enumerate(ids.tolist()):
            g = gt_index[fid]
            for slot, side in enumerate(('left', 'right')):
                gt_slot = hm[slot]
                eligible = gv[g, gt_slot, jm] if gt_slot >= 0 else np.zeros(21, bool)
                valid = eligible & hand['valid'][t, slot] & np.isfinite(pred[t, slot]).all(-1)
                raw = np.full(21, np.nan)
                rr = raw.copy()
                pa = raw.copy()
                if valid.any():
                    target = truth[g, gt_slot, jm]
                    raw = np.linalg.norm(pred[t, slot] - target, axis=-1) * 1000
                    if valid[0]:
                        rr = np.linalg.norm((pred[t, slot] - pred[t, slot, :1]) -
                                            (target - target[:1]), axis=-1) * 1000
                    aligned = similarity_align(pred[t, slot, valid], target[valid])
                    if aligned is not None:
                        pa[valid] = np.linalg.norm(aligned - target[valid], axis=-1) * 1000
                rrv = valid & bool(valid[0])
                item = {
                    'run_id': rid, 'dataset': run['dataset'], 'input_frame_index': t,
                    'source_frame_index': fid, 'native_slot': slot, 'anatomical_hand': side,
                    'network_H_anchor': bool(fid in anchors),
                    'gt_annotated': bool(eligible.any()), 'gt_joint_count': int(eligible.sum()),
                    'official_valid': bool(hand['official_valid'][t, slot]),
                    'official_fill': bool(hand['fill_mask'][t, slot]),
                    'display_valid': bool(hand['display_valid'][t, slot]),
                    'strict_valid': bool(hand['valid'][t, slot]),
                    'evaluated_joint_count': int(valid.sum()),
                    'root_relative_joint_count': int(rrv.sum()),
                    'pa_joint_count': int(np.isfinite(pa).sum()),
                    'raw_mpjpe_mm': average(raw, valid),
                    'root_relative_mpjpe_mm': average(rr, rrv),
                    'pa_mpjpe_mm': average(pa),
                    'wrist_error_mm': float(raw[0]) if valid[0] else None,
                    'pck50_correct_joints': int(((raw <= 50) & valid).sum()),
                }
                rows.append(item)
                per_run.append(item)
        c = run['primary']['coverage']
        assert sum(x['gt_joint_count'] for x in per_run) == c['gt_joint_count']
        assert sum(x['evaluated_joint_count'] for x in per_run) == c['evaluated_joint_count']
        assert sum(x['gt_annotated'] and x['official_fill'] for x in per_run) == c['filled_annotated_hand_frames']
        assert sum(x['evaluated_joint_count'] > 0 for x in per_run) == c['evaluated_hand_frames']
        reproduced = {}
        for field, report_key, weight in [
            ('raw_mpjpe_mm', 'mpjpe_raw_mm', 'evaluated_joint_count'),
            ('root_relative_mpjpe_mm', 'mpjpe_root_relative_mm', 'root_relative_joint_count'),
            ('pa_mpjpe_mm', 'mpjpe_pa_mm', 'pa_joint_count'),
        ]:
            used = [x for x in per_run if x[field] is not None and x[weight] > 0]
            value = sum(x[field] * x[weight] for x in used) / sum(x[weight] for x in used) if used else None
            expected = run['primary']['metrics'][report_key]
            assert (value is None and expected is None) or np.isclose(value, expected, atol=1e-8, rtol=1e-10), (rid, field, value, expected)
            reproduced[field] = value
        # Also match the published per-frame curves, including undefined frames.
        curves = {s['id']: dict(zip(s['frame_indices'], s['values'])) for s in run['primary']['series']}
        for t, fid in enumerate(ids.tolist()):
            frame_rows = per_run[t * 2:t * 2 + 2]
            for field, curve, weight in [('raw_mpjpe_mm', 'mpjpe_raw_mm', 'evaluated_joint_count'),
                                         ('root_relative_mpjpe_mm', 'mpjpe_rr_mm', 'root_relative_joint_count')]:
                used = [x for x in frame_rows if x[field] is not None and x[weight] > 0]
                value = sum(x[field] * x[weight] for x in used) / sum(x[weight] for x in used) if used else None
                expected = curves[curve][fid]
                assert (value is None and expected is None) or np.isclose(value, expected, atol=1e-8, rtol=1e-10)
        checks.append({'run_id': rid, 'rows': len(per_run), 'aggregate_and_curves_match': True,
                       'reproduced': reproduced})
        all_runs.append({'run_id': rid, 'source': src, 'rows': per_run})
        print(rid, len(per_run), 'validated', flush=True)
    assert len(rows) == 2 * suite['rgb_frames'] == 4160
    with (root / 'reports/per_frame_hand_scores.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {'schema_version': 'egofound3r.per_frame_hand_scores.v1',
              'completed_utc': datetime.now(timezone.utc).isoformat(), 'rows': len(rows),
              'policy': suite['primary_policy'],
              'unannotated_slots': 'Included for prediction/fill auditing; gt_joint_count=0, excluded from accuracy.',
              'undefined_errors': 'JSON null / empty CSV cell, never zero-filled.',
              'runs': all_runs, 'validation': checks}
    (root / 'reports/per_frame_hand_scores.json').write_text(json.dumps(finite_json(result), ensure_ascii=False, indent=2) + '\n')
    receipt = {'status': 'passed', 'rows': len(rows), 'frames': suite['rgb_frames'],
               'runs': len(all_runs), 'aggregate_and_all_frame_curves_match': True,
               'checks': checks, 'gpu_used': False, 'model_inference_run': False}
    (root / 'reports/per_frame_scores_validation.json').write_text(json.dumps(finite_json(receipt), indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'checks'}))


if __name__ == '__main__':
    main()
