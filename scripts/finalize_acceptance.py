#!/usr/bin/env python3
"""Attach observed browser acceptance and diagnostic summaries, without inference."""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    r = ap.parse_args().root
    reports = r / 'reports'
    docs = r / 'docs'
    suite = json.loads((reports / 'suite_metrics.json').read_text())
    qa = json.loads((reports / 'browser_acceptance.json').read_text())
    release = json.loads((reports / 'github_release_receipt.json').read_text())
    pages = json.loads((reports / 'github_pages_initial_build.json').read_text())
    frames = json.loads((reports / 'per_frame_scores_validation.json').read_text())
    assert suite['streams'] == 13 and suite['rgb_frames'] == 2080
    assert len(qa['loaded_exports']) == 39 and all(qa['checks'].values())
    assert frames['status'] == 'passed' and frames['rows'] == 4160
    assert release['archive_count'] == 13 and release['asset_sizes_and_digests_verified']
    assert pages['status'] == 'built' and pages['commit'] == release['commit']
    manifest = json.loads((docs / 'manifest.json').read_text())
    by_run = {x['run_id']: x for x in suite['runs']}
    camera_lines = []
    for seq in manifest['sequences']:
        notes = [n for n in seq['notes'] if not n.startswith('相机轨迹诊断：')]
        runs = [x for x in suite['runs'] if x['run_id'].startswith(seq['id'] + '__')]
        for run in runs:
            cam = run['additional_metrics'].get('camera_first_frame_aligned')
            if cam:
                view = run['run_id'].split('__')[1]
                notes.insert(0, f"相机轨迹诊断：{view} 平移 RMSE {cam['translation_rmse_mm']:.2f} mm，平均旋转误差 {cam['rotation_mean_deg']:.2f}°。只对齐首个有效相机坐标，不拟合尺度。")
                if seq['dataset'] == 'HOT3D-Aria':
                    camera_lines.append(f"| {seq['label']} | {cam['translation_rmse_mm']:.2f} | {cam['rotation_mean_deg']:.2f} |")
        if seq['id'] == 'aria_3302':
            seq['notice'] = '轨迹失败项：Aria 3302 相机平移 RMSE 320.30 mm、平均旋转误差 22.56°。手部相机系误差不代表世界轨迹可靠；仍完整展示该段。默认排除官方填补。'
        seq['notes'] = notes
    additions = [
        {'label': '逐帧左右手 CSV', 'url': 'downloads/per_frame_hand_scores.csv'},
        {'label': '逐帧左右手 JSON', 'url': 'downloads/per_frame_hand_scores.json'},
        {'label': '在线交互验收', 'url': 'downloads/browser_acceptance.json'},
        {'label': '发布验收凭据', 'url': 'downloads/publication_receipt.json'},
    ]
    existing = {x['url'] for x in manifest['downloads']}
    manifest['downloads'] += [x for x in additions if x['url'] not in existing]
    manifest['status'] = 'complete'
    manifest['acceptance_status'] = 'passed'
    save(docs / 'manifest.json', manifest)
    failure_lines = []
    for dataset, group in suite['datasets'].items():
        c = group['coverage']
        missing = c['gt_hand_frames'] - c['evaluated_hand_frames']
        fill = c['filled_annotated_hand_frames']
        failure_lines.append(f"| {dataset} | {c['gt_hand_frames']} | {c['evaluated_hand_frames']} | {missing} | {fill} | {missing-fill} |")
    contact_lines = []
    for run in suite['runs']:
        c = run['contact']
        if c['available']:
            contact_lines.append(f"| {run['run_id']} | {100*c['precision']:.2f}% | {100*c['recall_on_valid']:.2f}% | {c['f1_on_valid']:.4f} | {100*c['recall_all_gt_positives_including_failures']:.2f}% | {c['distance_mae_mm']:.2f} |")
    supplement = '''<!-- acceptance-start -->
## 失败、轨迹与在线验收

下表以已标注 GT 手帧为分母。官方填补不进入主要准确率；未标注的左/右槽位单列在逐帧审计 CSV 中，不混入该分母。

| 数据 | GT 手帧 | 严格有效 | 非有效合计 | 其中官方填补 | 其余无有效输出 |
|---|---:|---:|---:|---:|---:|
''' + '\n'.join(failure_lines) + '''

DexYCB 仅 1 个已标注手帧通过严格有效判定，因此 360.99 mm 来自这一帧，不能视为整个双相机配置的稳定表现。其余三个 DexYCB 路径的主要 MPJPE 不可定义。

### 相机轨迹

首个有效相机位姿只用于固定坐标规范，未进行最佳 Sim(3) 或尺度拟合。下表为实际相机预测与时间戳匹配的 HOT3D 位姿比较；旋转列为均值，平移列为 RMSE。

| Aria 片段 | 平移 RMSE mm | 平均旋转误差 ° |
|---|---:|---:|
''' + '\n'.join(camera_lines) + '''

**Aria 3302 是明确的世界轨迹失败项：320.30 mm、22.56°。** 相机坐标下较小的手部误差不能推导出世界轨迹可靠。该段未删除，网页有专门提示。固定 DexYCB 相机的预测漂移为 2.11–6.54 mm、0.11–0.49°；HO3D 未取得可独立验证的动态相机位姿，不报告轨迹精度。

### HO3D 接触诊断

仅评分原生 195 顶点；概率阈值 0.5。P/R/F1 的条件分母排除无严格有效手部预测的顶点；另外保留全 GT 正例召回率，揭示失败帧影响。接触标签语义差异仍保留，不将此表作为官方基准成绩。

| 路径 | Precision | Recall（有效） | F1（有效） | Recall（全 GT 正例） | 接触距离 MAE mm |
|---|---:|---:|---:|---:|---:|
''' + '\n'.join(contact_lines) + '''

### 可复核逐帧输出

`reports/per_frame_hand_scores.{csv,json}` 包含 2,080 帧 × 2 个原生槽位，共 **4,160 行**：源帧号、解剖左右手、网络 H 锚帧标记、官方 valid、填补、严格有效、GT 标注数、Raw/RR/PA MPJPE、腕误差和 PCK50 命中数。未标注槽位保留审计信息，准确率字段为空。每一路的计数、加权聚合和原逐帧曲线已逐项一致性核验，见 `reports/per_frame_scores_validation.json`。这一步只读取既有 `.pt`，没有再次推理。

GT 3D 评分使用全部有效标注；网页 2D 投影只显示正深度的关节，因此相机后方的已标注手可能不显示在参考投影中。

### 真实浏览器验收

云端 Chrome 已逐一加载 **11 段、26 份相机/世界系预测、13 份 GT 参考**；所有首次载入视频 readyState=4。验证了双/三列切换、帧号跳转、三面板同步播放/暂停、填补开关、接触着色、腕轨迹、3D 拖动和联动视角。实测 Aria 1976 暂停在第 47 帧，三视频 currentTime 均为 1.566666 秒。拖动前后，两个 3D 面板均改变，两个 RGB 区域像素完全一致。

当前云浏览器未提供 WebGL2，实际验收走软件 3D 回退（真实顶点/面片线框）。不声称已验证 WebGL 渲染性能或移动端。未观察到应用控制台错误；浏览器扩展自身的 metadata 通信错误单独记录。

详见 `reports/browser_acceptance.json`；拖动前后原始截图在 `reports/qa/`。GitHub Pages 已构建，13 份原始预测 release 归档均核对了大小及 SHA256；此后验收说明提交不改变预测、评价数值或归档。
<!-- acceptance-end -->
'''
    readme = (r / 'README.md').read_text()
    if '<!-- acceptance-start -->' in readme:
        start = readme.index('<!-- acceptance-start -->')
        end = readme.index('<!-- acceptance-end -->') + len('<!-- acceptance-end -->')
        readme = readme[:start] + supplement + readme[end:]
    else:
        readme += '\n' + supplement
    readme = readme.replace('| 0.3% |', '| 0.34% |').replace('| 94.4% |', '| 94.38% |').replace('| 49.5% |', '| 49.54% |')
    (r / 'README.md').write_text(readme)
    now = datetime.now(timezone.utc).isoformat()
    receipt = {'status': 'completed_and_verified_online', 'updated_utc': now,
               'repository': release['repository'], 'url': qa['tested_site'],
               'result_payload_commit': release['commit'],
               'acceptance_documentation_commit': 'git revision containing this receipt; payload unchanged',
               'pages_result_build': pages,
               'predictions': 13, 'evaluations': 13, 'GTreferences': 13,
               'prediction_geometry_exports': 26, 'frames': 2080, 'sequences': 11,
               'per_frame_anatomical_hand_rows': 4160,
               'official_source_commit': '264f4df9fd42536aae4dba982d18272cc3334077',
               'raw_results': release, 'browser_acceptance': 'reports/browser_acceptance.json',
               'asset_validation': 'reports/site_asset_validation.json',
               'resume_used_existing_predictions': True, 'inference_rerun_during_acceptance': False,
               'H20_used': False, 'other_jobs_preempted_or_terminated': False,
               'remaining_external_blockers': [],
               'limitations': qa['limitations']}
    save(reports / 'publication_receipt.json', receipt)
    (r / 'status.md').write_text('# 当前状态\n\n复现、独立评分、全部预测导出与线上交互验收已完成。13/13 路、2,080 RGB 帧；26 份预测几何、13 份 GT 参考；4,160 行逐帧左右手评分。\n\n页面：https://yichengdraw.github.io/egofound3r-stereo-20261111/\n\n真实失败保留：DexYCB 严格有效 1/292 手帧；Aria 3302 轨迹误差 320.30 mm / 22.56°。页面实际以软件 3D 回退验收，未验证 WebGL 性能。\n\n验收与发布凭据：reports/browser_acceptance.json、reports/publication_receipt.json。权重授权和历史 HTTP 401 已解决；当前无外部阻塞。\n')
    for name in ['README.md', 'status.md']:
        shutil.copy2(r / name, docs / 'downloads' / name)
    for name in ['browser_acceptance.json', 'publication_receipt.json', 'per_frame_hand_scores.csv',
                 'per_frame_hand_scores.json', 'per_frame_scores_validation.json']:
        shutil.copy2(reports / name, docs / 'downloads' / name)
    (docs / 'downloads/qa').mkdir(exist_ok=True)
    for p in (reports / 'qa').glob('*.png'):
        shutil.copy2(p, docs / 'downloads/qa' / p.name)
    print(json.dumps({'status': receipt['status'], 'predictions': 13, 'evaluations': 13,
                      'per_frame_rows': 4160, 'geometry_exports_verified_online': 39}))


if __name__ == '__main__':
    main()
