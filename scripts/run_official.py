#!/usr/bin/env python3
"""Run official EgoFound3R on each RGB stream with its measured intrinsics.

Input manifest contains only RGB, K and frame timing. Ground truth is read by
the separate evaluator. Cameras are never interleaved into the time axis.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

OFFICIAL_COMMIT = "264f4df9fd42536aae4dba982d18272cc3334077"


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_result(path: Path, frames: int) -> dict:
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    m = obj["metric_predictions"]
    hand = m["hand"]
    xyz = hand["joint_xyz_metric"]
    if hand.get("unit") != "meter" or hand.get("coordinate_frame") != "camera":
        raise ValueError("Unexpected official coordinate/unit contract")
    if xyz.ndim == 5 and xyz.shape[0] == 1:
        xyz = xyz[0]
    if tuple(xyz.shape) != (frames, 2, 21, 3):
        raise ValueError(f"Output shape {tuple(xyz.shape)} differs from input {frames} frames")
    return {"joint_shape": list(xyz.shape), "unit": hand["unit"],
            "coordinate_frame": hand["coordinate_frame"]}


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    # A single short-window inference shares an available card. Existing jobs
    # are observed only; this script never terminates or suspends them.
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], check=True, text=True, capture_output=True)
    rows = [tuple(int(x.strip()) for x in line.split(","))
            for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("No CUDA device was reported")
    index, free_mb, util = max(rows, key=lambda x: (x[1], -x[2]))
    print(f"Selected GPU {index}: {free_mb} MiB free, utilization {util}%", flush=True)
    return f"cuda:{index}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root = args.root.resolve()
    upstream = root / "upstream"
    sha = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if sha != OFFICIAL_COMMIT:
        raise RuntimeError(f"Expected official commit {OFFICIAL_COMMIT}; found {sha}")
    manifest_path = args.manifest or root / "data/input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "egofound3r.inputs.v1":
        raise ValueError("Unknown input manifest schema")
    for folder in ("logs", "reports", "configs", "outputs"):
        (root / folder).mkdir(exist_ok=True)
    cfg = (upstream / "configs/default.toml").read_text()
    cfg = cfg.replace('vggt_checkpoint_path = ""',
                      "vggt_checkpoint_path = " + json.dumps(str(root / "weights/vggt_omega_1b_512.pt")))
    cfg_path = root / "configs/runtime.toml"
    cfg_path.write_text(cfg)
    if not args.dry_run:
        for name in ("EgoFound3R.pt", "vggt_omega_1b_512.pt"):
            path = root / "weights" / name
            if not path.is_file() or path.stat().st_size < 1024 * 1024:
                raise RuntimeError(f"Official weight unavailable: {name}. Obtain authorized access, then run download_weights.py.")
    report_path = root / "reports/inference_runs.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"official_commit": sha, "runs": {}}
    for seq in manifest["sequences"]:
        for view in seq["views"]:
            run_id = f"{seq['id']}__{view['id']}"
            out = root / "outputs" / run_id
            payload = out / "inference_output.pt"
            if payload.exists() and (out / "summary.json").exists():
                report["runs"][run_id] = {"status": "complete", "validation": validate_result(payload, seq["frames"]), "checked_utc": stamp()}
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                continue
            if out.exists() and any(out.iterdir()):
                raise RuntimeError(f"Partial output exists at {out}; inspect the saved failure before choosing a fresh output directory.")
            device = args.device if args.dry_run else choose_device(args.device)
            command = [args.python, str(upstream / "infer_marker_video.py"),
                       "--config", str(cfg_path), "--checkpoint", str(root / "weights/EgoFound3R.pt"),
                       "--video-path", str(view["video"]), "--output-dir", str(out),
                       "--root-k-source", "external", "--root-k-file", str(view["K_file"]),
                       "--device", device, "--upsample-mano", "--no-overlay-frame-images",
                       "--high-fps", "10", "--global-stride", "5",
                       "--chunk-seconds", "2", "--overlap-seconds", "0.5"]
            entry = {"status": "prepared" if args.dry_run else "running", "started_utc": stamp(),
                     "command": command, "input_frames": seq["frames"], "input_fps": seq["fps"],
                     "camera_id": view["camera_id"], "inputs": ["RGB", "camera intrinsics"],
                     "ground_truth_used": False}
            report["runs"][run_id] = entry
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            if args.dry_run:
                print(json.dumps({"run": run_id, "command": command}))
                continue
            with (root / "logs" / f"{run_id}.log").open("a") as log:
                result = subprocess.run(command, cwd=upstream, stdout=log, stderr=subprocess.STDOUT)
            entry["finished_utc"] = stamp()
            entry["exit_code"] = result.returncode
            if result.returncode != 0:
                entry["status"] = "failed"
                report_path.write_text(json.dumps(report, indent=2) + "\n")
                raise RuntimeError(f"Official inference failed: {run_id}; inspect logs/{run_id}.log")
            entry["validation"] = validate_result(payload, seq["frames"])
            entry["status"] = "complete"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"COMPLETE {run_id}", flush=True)


if __name__ == "__main__":
    main()

