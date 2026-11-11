"""Small analytic tests of coordinate, masking and scoring risks.

Fixtures are generated only in memory and tempfile.TemporaryDirectory and are
never written to a website. This does not test or claim model inference.
Run: python test_contracts.py --official-repo /path/to/EgoFound3R
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np

from official_contract import camera_contract, camera_to_world, dense_hand
from export_viewer import build_scene, export_payload
from evaluate_predictions import evaluate


def fixture():
    rng = np.random.default_rng(312)
    joints = rng.normal(0, .02, (2, 2, 21, 3)).astype(np.float32)
    vertices = rng.normal(0, .02, (2, 2, 195, 3)).astype(np.float32)
    joints[..., 2] += 1
    vertices[..., 2] += 1
    c2w = np.repeat(np.eye(4)[None], 2, axis=0)
    c2w[1, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    c2w[1, :3, 3] = [1, 2, 3]
    K = np.array([[2., 0, 0], [0, 2., 0], [0, 0, 1]])
    return {"fps": 30., "h_fps": 10., "g_fps": 2., "image_size_hw": [2, 2],
        "source_frame_indices": np.array([10, 20]), "high_frame_indices": np.array([10]),
        "hand_projection_K_full": np.repeat(K[None], 2, axis=0),
        "hand_projection_K_valid_full": np.ones(2, bool),
        "hand_missing_fill_full": np.array([[False, False], [True, False]]),
        "hand_display_valid_full": np.ones((2, 2), bool),
        "metric_predictions": {"hand": {"unit": "meter", "coordinate_frame": "camera",
            "joint_xyz_metric": joints, "vertex_xyz_metric": vertices, "valid": np.ones((2, 2), bool),
            "vertex_contact_probability": np.full((2, 2, 195), .25, np.float32)},
            "camera": {"translation_unit": "meter", "pose_w2c_metric": np.linalg.inv(c2w),
                "valid": np.ones(2, bool), "frame_indices": np.array([10, 20]), "world_scope": "test_only"},
            "depth": {"unit": "meter", "value_metric": np.full((1, 2, 2, 1), 2., np.float32),
                "confidence": np.ones((1, 2, 2, 1)), "valid": np.ones(1, bool), "frame_indices": np.array([20])},
            "intrinsics": {"K_global": K[None], "K_global_valid": np.ones(1, bool)}}}


def run(repo: Path):
    p = fixture()
    cams = camera_contract(p, 2)
    converted = camera_to_world(np.array([[[0., 0, 2]], [[0., 0, 2]]]), cams["c2w"])
    np.testing.assert_allclose(converted, [[[0., 0, 2]], [[1., 2, 5]]])
    scene = build_scene(p, cams, coordinate="world", stride=1, max_points=4,
                        min_confidence=0, max_depth=20)
    assert scene["frame_indices"] == [1], "G anchor must map to exact R-frame index"
    np.testing.assert_allclose(scene["points"][0, 0], [1., 2, 5])
    np.testing.assert_allclose(scene["points"][0, 1], [1., 3, 5])
    dense = dense_hand(p, repo)
    with np.load(repo / "egohandmetric_prompt/data/mano_upsampling/mano_195_to_778.npz") as a:
        ids = a["source_vertex_ids"]
    np.testing.assert_array_equal(dense["vertices"][..., ids, :], p["metric_predictions"]["hand"]["vertex_xyz_metric"])
    with tempfile.TemporaryDirectory(prefix="egofound_test_only_") as tmp:
        out = Path(tmp)
        report = export_payload(p, out, coordinate="world", official_repo=repo,
                                scene_stride=1, max_scene_points=4)
        spec = report["vertices"]
        vertices = np.fromfile(out / spec["url"], dtype="<f4").reshape(spec["shape"])
        np.testing.assert_allclose(vertices[1][:, ids, :],
            camera_to_world(p["metric_predictions"]["hand"]["vertex_xyz_metric"], cams["c2w"])[1], atol=2e-7)
        valid = np.fromfile(out / report["valid"]["url"], dtype="u1").reshape(2, 2)
        assert not valid[1, 0], "Filled coordinates must not become trusted predictions"
        uv = np.fromfile(out / report["joints2d"]["url"], dtype="<f4").reshape(2, 2, 21, 2)
        original = p["metric_predictions"]["hand"]["joint_xyz_metric"]
        np.testing.assert_allclose(uv, original[..., :2] / original[..., 2:3] * 2, atol=1e-6)
        for s in [report["vertices"], report["joints"], report["valid"], report["scene"]["points"]]:
            assert (out / s["url"]).stat().st_size == s["byte_length"]
    truth = p["metric_predictions"]["hand"]["joint_xyz_metric"][:, :1].astype(np.float64).copy()
    truth[..., 0] -= .05
    gt = {"joints": truth, "valid": np.ones((2, 1, 21), bool), "source_frame_indices": np.array([10, 20]),
          "units": np.array("m"), "coordinate_frame": np.array("camera")}
    try:
        evaluate(p, gt)
    except ValueError as exc:
        assert "Explicit pred_to_gt" in str(exc)
    else:
        raise AssertionError("Anatomical slot order must not be guessed")
    report = evaluate(p, gt, {"pred_to_gt_hand": [0, -1], "pred_to_gt_joint": list(range(21)), "evidence": "analytic test fixture only"}, [20., 60.])
    np.testing.assert_allclose(report["metrics"]["mpjpe_raw_mm"], 50, atol=1e-5)
    np.testing.assert_allclose(report["metrics"]["mpjpe_root_relative_mm"], 0, atol=1e-5)
    np.testing.assert_allclose(report["metrics"]["mpjpe_pa_mm"], 0, atol=1e-5)
    assert report["coverage"]["fraction"] == .5
    assert report["pck"]["60mm"]["all_gt_including_failures"] == .5
    assert report["pck"]["60mm"]["conditional_on_valid_prediction"] == 1
    print("PASS: analytic camera inversion, G/R frame mapping, official upsampling anchors, binary layout, camera-space 2D projection, fill exclusion, GT denominator and explicit mapping checks. NO MODEL INFERENCE PERFORMED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    run(parser.parse_args().official_repo)
