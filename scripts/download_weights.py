#!/usr/bin/env python3
"""Download the two official checkpoints using the user's existing HF authorization.

Never accepts a token argument or records credentials. The Hugging Face client
uses its normal login. Access failures remain failures; no mirror is substituted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MODELS = [
    ("Frank-F2021/EgoFound3R", "EgoFound3R.pt"),
    ("facebook/VGGT-Omega", "vggt_omega_1b_512.pt"),
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    root = args.root.resolve()
    (root / "weights").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(), "checkpoints": []}
    # Resolve the official revision before download, and keep a single immutable
    # revision for each complete checkpoint. hf_hub_download resumes its cache.
    api = HfApi(endpoint="https://huggingface.co")
    for repo, name in MODELS:
        info = api.model_info(repo_id=repo)
        path = Path(hf_hub_download(
            repo_id=repo, filename=name, revision=info.sha,
            endpoint="https://huggingface.co", local_dir=root / "weights",
        ))
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        receipt["checkpoints"].append({
            "repository": repo, "filename": name, "revision": info.sha,
            "sha256": h.hexdigest(), "size_bytes": path.stat().st_size,
            "source": f"https://huggingface.co/{repo}",
        })
        (root / "reports/checkpoint_provenance.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        print(f"READY {repo}/{name}", flush=True)


if __name__ == "__main__":
    main()

