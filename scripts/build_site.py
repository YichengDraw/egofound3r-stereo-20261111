#!/usr/bin/env python3
"""Assemble the static viewer from actual prepared RGB and available outputs.

Absent predictions stay blocked. This script never synthesizes geometry,
copies ground truth into a model panel, or substitutes a different model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--docs", type=Path)
    p.add_argument("--export-predictions", action="store_true")
    p.add_argument("--export-references", action="store_true",
                   help="Export separately labelled ground-truth skeletons for reference only")
    args = p.parse_args()
    root = args.root.resolve()
    docs = (args.docs or root / "docs").resolve()
    preview = json.loads((root / "data/preview_manifest.json").read_text())
    validation = json.loads((root / "reports/data_validation.json").read_text())
    inputs = json.loads((root / "data/input_manifest.json").read_text())
    by_id = {s["id"]: s for s in inputs["sequences"]}
    docs.mkdir(parents=True, exist_ok=True)
    (docs / ".nojekyll").touch()
    for name in ("data_validation.json", "weight_access.json"):
        src = root / "reports" / name
        if src.exists():
            copy_file(src, docs / "downloads" / name)
    for name in ("README.md", "PROTOCOL.md"):
        if (root / name).exists():
            copy_file(root / name, docs / "downloads" / name)
    if (root / "data/preview_contact_sheet.jpg").exists():
        copy_file(root / "data/preview_contact_sheet.jpg", docs / "downloads/input_overview.jpg")
    manifest = {
        "schema_version": "egofound3r.viewer.v1",
        "title": "EgoFound3R · 双 RGB / 已知内参",
        "subtitle": "DexYCB 箱子与香蕉 · 146 个同步时刻 · 292 幅真实 RGB",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "blocked",
        "status_label": "待权重授权 · 未完成推理",
        "notice": "两套官方权重均返回 HTTP 401，需要账户获得访问权限并登录。当前展示真实双 RGB 输入；模型预测、3D 重建和指标尚未生成。",
        "provenance": "官方代码 264f4df · 原始 RGB 与 K 已核对 · 预测状态逐项标明",
        "protocol": [
            {"label": "输入", "value": "两路同步 RGB + 各自已知内参。GT 手、物体位姿、深度与外参只用于独立评价或未使用。"},
            {"label": "双目范围", "value": "官方模型为单视频接口；相机 A、B 分别运行。尚未实现联合双目融合。"},
            {"label": "相机配置", "value": f"固定宽基线：{validation['baseline_m']:.3f} m；相对旋转 {validation['relative_rotation_deg']:.2f}°。与佩戴式窄基线双目存在差异。"},
            {"label": "内参作用", "value": "官方 external K 用于手部定位和投影；场景分支保留自己的预测内参。"},
            {"label": "运行设置", "value": "官方 2 s 窗口 / 0.5 s overlap；H 10 FPS，G stride 5；R 为 30 FPS，包含插值。"},
            {"label": "几何与评价", "value": "米制坐标；各相机独立重建。保留官方有效性与填补标记，确认手槽/关节对应后评分。"},
        ],
        "sources": [
            {"label": "论文", "url": "https://arxiv.org/abs/2610.01210"},
            {"label": "官方代码", "url": "https://github.com/MINT-SJTU/EgoFound3R"},
            {"label": "复现代码", "url": "https://github.com/YichengDraw/egofound3r-stereo-20261111"},
        ],
        "downloads": [
            {"label": "复现协议", "url": "downloads/PROTOCOL.md"},
            {"label": "数据校验", "url": "downloads/data_validation.json"},
            {"label": "权重访问状态", "url": "downloads/weight_access.json"},
            {"label": "输入总览", "url": "downloads/input_overview.jpg"},
        ],
        "sequences": [],
    }
    completed = 0
    scored = 0
    total = 0
    references = 0
    for src in preview["scenes"]:
        sid = src["id"]
        prepared = by_id[sid]
        seq = {
            "id": sid, "label": "DexYCB · " + ("箱子" if sid.endswith("box") else "香蕉"),
            "dataset": "DexYCB", "frames": src["frame_count"], "fps": src["fps"],
            "frame_ids": src["frame_indices"], "timestamps": src["timestamps_s"],
            "input_label": "同步双 RGB · 640 × 480 · 已知 K",
            "coordinate_system": "各自相机坐标 · m",
            "description": f"源序列 {src['sequence_id']}；同步依据为数据集相同源帧号。采用数据集公布的针孔内参约定，未额外拟合标定或输入 GT。",
            "metric_scope": "等待真实预测与独立评价",
            "notes": [
                "两列视频来自同一动作的两台真实 RGB 相机。",
                "已核对帧数、图像尺寸、RGB 通道和标定对应；这些检查属于数据验证。",
                "原版模型不联合读取左右图像。本页保留两路独立结果，便于比较视角影响。",
            ], "rgb": {}, "methods": [],
        }
        reference_methods = []
        for v, input_v in zip(src["views"], prepared["views"]):
            total += 1
            key = v["view"]
            asset = Path("data") / sid / input_v["id"]
            copy_file(Path(v["video_path"]), docs / asset / "input.mp4")
            copy_file(Path(v["poster_path"]), docs / asset / "poster.jpg")
            copy_file(Path(v["intrinsics_path"]), docs / asset / "root_intrinsics.json")
            seq["rgb"][key] = {"video": str(asset / "input.mp4"), "poster": str(asset / "poster.jpg"),
                               "width": v["width"], "height": v["height"], "K": v["K"]}
            run_id = f"{sid}__{input_v['id']}"
            payload = root / "outputs" / run_id / "inference_output.pt"
            geo = docs / asset / "prediction/geometry.json"
            if args.export_predictions and payload.exists() and not geo.exists():
                subprocess.run([sys.executable, str(root / "scripts/export_viewer.py"),
                                "--input", str(payload), "--output", str(geo.parent),
                                "--rgb-video", v["video_path"], "--coordinate", "camera",
                                "--official-repo", str(root / "upstream")], check=True)
            method = {"id": f"official_{key.lower()}", "label": f"EgoFound3R · 相机 {key}",
                      "kind": "official", "view": key, "view_label": f"相机 {key}",
                      "status": "blocked", "coordinate_system": f"相机 {key} 坐标 · m",
                      "summary": f"RGB {v['camera_id']} + 真实 K · 独立官方推理",
                      "missing_reason": "待官方权重授权\n尚未运行 EgoFound3R，没有生成 3D 预测。",
                      "metrics": []}
            if geo.exists() and payload.exists():
                geometry = json.loads(geo.read_text())
                if geometry["frame_count"] != seq["frames"]:
                    raise ValueError("Geometry and source frame count differ")
                method.update(status="complete", geometry=str(geo.relative_to(docs)))
                method.pop("missing_reason")
                completed += 1
                for name in ("overlay.mp4", "overlay_multimode.mp4"):
                    vid = payload.parent / name
                    if vid.exists():
                        copy_file(vid, docs / asset / name)
                        if name == "overlay.mp4":
                            method["overlay_video"] = str(asset / name)
                score = root / "reports" / f"{run_id}_metrics.json"
                if score.exists():
                    metrics = json.loads(score.read_text())
                    for metric_id, title in [("mpjpe_raw_mm", "绝对关节误差"),
                                             ("mpjpe_root_relative_mm", "去腕关节误差"),
                                             ("mpjpe_pa_mm", "PA 关节误差")]:
                        method["metrics"].append({"id": metric_id, "label": title,
                                                  "value": metrics["metrics"].get(metric_id), "unit": "mm"})
                    geometry["series"] = metrics["series"]
                    geo.write_text(json.dumps(geometry, ensure_ascii=False, indent=2) + "\n")
                    copy_file(score, docs / "downloads" / score.name)
                    scored += 1
            seq["methods"].append(method)
            reference = docs / asset / "reference/geometry.json"
            gt = root / "evaluation_ref" / sid / str(v["camera_id"]) / "gt.npz"
            if args.export_references and gt.exists():
                subprocess.run([sys.executable, str(root / "scripts/export_gt_reference.py"),
                                "--gt", str(gt), "--output", str(reference.parent),
                                "--intrinsics", str(v["intrinsics_path"])], check=True)
            if reference.exists():
                ref_geometry = json.loads(reference.read_text())
                if ref_geometry["frame_count"] != seq["frames"] or ref_geometry.get("data_role") != "ground_truth_reference":
                    raise ValueError("Reference must contain matching real GT frames and an explicit reference role")
                reference_methods.append({
                    "id": f"gt_{key.lower()}", "label": f"GT 参考 · 相机 {key} · 仅评价",
                    "kind": "reference", "view": key, "view_label": f"相机 {key}",
                    "status": "complete", "geometry": str(reference.relative_to(docs)),
                    "coordinate_system": f"相机 {key} 坐标 · m",
                    "summary": "真实 DexYCB 标注：21 关节、腕部轨迹及 RGB 投影；不是模型预测。",
                    "metrics": [],
                })
                references += 1
        seq["methods"].extend(reference_methods)
        if reference_methods:
            seq["notes"].append("GT 参考列仅展示真实标注的骨架、腕部轨迹和投影；不输入模型，也不计入预测完成数。")
        manifest["sequences"].append(seq)
    if completed:
        manifest.update(status="complete" if completed == total and scored == total else "partial",
                        status_label=f"推理 {completed}/{total} · 评价 {scored}/{total}",
                        notice=f"已导出 {completed}/{total} 路真实预测，完成 {scored}/{total} 路独立评价。两路仍为官方单视频模型的独立运行。")
    if references:
        manifest["default_columns"] = 3
        manifest["notice"] += f" 可拖动的 GT 参考来自 {references} 路真实标注，不是模型预测。"
    (docs / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    # A new deployment must not reuse a cached script/manifest pair.
    if (docs / "index.html").exists() and (docs / "app.js").exists():
        revision = hashlib.sha256((docs / "app.js").read_bytes() + (docs / "manifest.json").read_bytes()).hexdigest()[:12]
        page, matches = re.subn(r'src="app\.js(?:\?v=[^"]*)?"',
                               f'src="app.js?v={revision}"', (docs / "index.html").read_text())
        if matches != 1:
            raise ValueError("Expected one app.js module script in index.html")
        (docs / "index.html").write_text(page)
    print(json.dumps({"docs": str(docs), "sequences": len(manifest["sequences"]),
                      "camera_streams": total, "predictions": completed, "evaluations": scored,
                      "ground_truth_references": references,
                      "status": manifest["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
