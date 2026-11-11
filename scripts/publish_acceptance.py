#!/usr/bin/env python3
"""Publish the acceptance supplement to the existing repository, once."""
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime, timezone

BASE = Path('/Users/yiche/26fall/Ego WM/egofound3r_reproduction_20261111')
REPO = BASE / 'github_pages'
ROOT = '/nvme-ssd2/zhaoyicheng/egofound3r-reproduction-20261111'
ENV = os.environ.copy()
for key in list(ENV):
    if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy'):
        ENV.pop(key)
ENV['GODEBUG'] = 'http2client=0'
GIT = ['git', '-c', 'http.proxy=', '-c', 'http.https://github.com.proxy=', '-c', 'http.version=HTTP/1.1']
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == 'f2d414bc9e7da7070843cd410512b6f325e8e0d1'
assert not subprocess.check_output(['git', 'diff', '--name-only'], cwd=REPO, text=True).strip()
assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=REPO, text=True).strip()
files = ['README.md', 'status.md', 'docs/manifest.json', 'docs/downloads/README.md', 'docs/downloads/status.md',
         'scripts/export_frame_scores.py', 'scripts/finalize_acceptance.py',
         'reports/github_release_receipt.json', 'reports/github_pages_initial_build.json']
for name in ('browser_acceptance.json', 'publication_receipt.json', 'per_frame_hand_scores.csv',
             'per_frame_hand_scores.json', 'per_frame_scores_validation.json'):
    files += ['reports/' + name, 'docs/downloads/' + name]
for name in ('egofound3r_drag_before.png', 'egofound3r_drag_after.png'):
    files += ['reports/qa/' + name, 'docs/downloads/qa/' + name]
filelist = BASE / 'acceptance_files.txt'
filelist.write_text('\n'.join(files) + '\n')
command = ['rsync', '-a', '--files-from=' + str(filelist), 'l40s:' + ROOT + '/', str(REPO) + '/']
subprocess.run(command, check=True)
(REPO / 'scripts/publish_acceptance.py').write_text(Path(__file__).read_text())
files += ['scripts/publish_acceptance.py']
qa = json.loads((REPO / 'reports/browser_acceptance.json').read_text())
assert len(qa['loaded_exports']) == 39 and all(qa['checks'].values())
receipt = json.loads((REPO / 'reports/publication_receipt.json').read_text())
assert receipt['status'] == 'completed_and_verified_online'
manifest = json.loads((REPO / 'docs/manifest.json').read_text())
for item in manifest['downloads']:
    if not item['url'].startswith('http'):
        assert (REPO / 'docs' / item['url']).exists(), item
import csv
with (REPO / 'reports/per_frame_hand_scores.csv').open() as f:
    assert sum(1 for _ in csv.DictReader(f)) == 4160
subprocess.run(['git', 'add', '--', *files], cwd=REPO, check=True)
staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=REPO, text=True).splitlines()
assert staged and not any(p.startswith(('scripts/data/', 'weights/', 'outputs/', 'evaluation_ref/')) for p in staged)
subprocess.run(['git', '-c', 'user.name=YichengDraw', '-c', 'user.email=202728613+YichengDraw@users.noreply.github.com',
                'commit', '-m', 'Record live acceptance, per-frame hand scores and trajectory failures'], cwd=REPO, env=ENV, check=True)
sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
subprocess.run(GIT + ['push', 'origin', 'main'], cwd=REPO, env=ENV, check=True)
out = {'status': 'acceptance_commit_pushed', 'commit': sha, 'result_payload_commit': receipt['result_payload_commit'],
       'updated_utc': datetime.now(timezone.utc).isoformat(), 'files': staged,
       'predictions_unchanged': True, 'raw_release_archives_unchanged': True}
(BASE / 'acceptance_publication.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out), flush=True)
