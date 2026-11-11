#!/usr/bin/env python3
"""Publishable, real-output viewer assembly for the frozen 13-stream suite."""
import argparse,json,hashlib,re,shutil,tarfile
from pathlib import Path
from datetime import datetime,timezone
from official_contract import load_prediction,sha256,finite_json
from export_viewer import export_payload
from export_gt_reference import export_reference

REPO='https://github.com/YichengDraw/egofound3r-stereo-20261111'
PAGE='https://yichengdraw.github.io/egofound3r-stereo-20261111/'
TAG='reproduction-20261111'

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(finite_json(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def cp(a,b):
    b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)
def num(x,d=2):return '—' if x is None else f'{x:.{d}f}'

def write_reports(r,score,inputs):
    table=['| 数据 / 输入 | RGB 帧 | 有效 / GT 手帧 | 覆盖率 | Raw MPJPE mm | 去腕 MPJPE mm | PA MPJPE mm | PCK50 全 GT |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for d,v in score['datasets'].items():
        c=v['coverage'];m=v['metrics'];table.append(f"| {d} | {v['rgb_frames']} | {c['evaluated_hand_frames']} / {c['gt_hand_frames']} | {num(100*c['fraction'],1)}% | {num(m['mpjpe_raw_mm'])} | {num(m['mpjpe_root_relative_mm'])} | {num(m['mpjpe_pa_mm'])} | {num(100*v['pck50_all_gt'],1)}% |")
    text=f'''# EgoFound3R 独立复现与验收

更新：{datetime.now(timezone.utc).isoformat()}。官方推理 **13/13 路完成**，11 段、2,080 帧 RGB，3 个数据来源。权重已合法传入且 SHA256 核验通过；历史 HTTP 401 已解决。

[交互页面]({PAGE}) · [所有原始预测张量]({REPO}/releases/tag/{TAG}) · [逐路 CSV](reports/metrics.csv) · [完整 JSON](reports/suite_metrics.json) · [复现协议](PROTOCOL.md)

## 实测结果

{chr(10).join(table)}

这里的“完成”指实际官方推理、导出和独立评分完成，不意味着模型对所有数据有效。零覆盖率的误差记为不可定义（—），绝不写成 0 mm。缺失预测计入全 GT PCK 分母。两路双 RGB 与单目补充分组报告，不合并冒充双目效果。

## 数据覆盖

- DexYCB：箱子 74、香蕉 72 个同步时刻，每段两个 RGB 相机，共 292 RGB 帧。两路分别推理；固定宽基线约 0.631 m、43.05°，不是佩戴式窄基线双目。
- HOT3D-Aria：1976、2022、3070、3209、3302、3353，共 890 RGB 帧。2022 现有校正缓存为原帧 10–149（140 帧）；其余各 150 帧。仅使用 214-1 RGB，未把灰度相机算作 RGB。
- HO3D-v3：MC1 299、SMu1 299、SM2 300 帧，共 898 RGB 帧。MC1 有 28 帧、SMu1 有 2 帧缺少手部标注；RGB 全部推理，缺标注帧不参与准确率分母。同一固定相机全部已公布 K 完全相同，缺手部标注的帧沿用该真实相机标定。
- ARCTIC 的近期 RGB 缓存存在，但未在给定目录找到可验证的源相机 K，因此不纳入已知 K 组；没有拿 SfM 估计 K 充当已知内参。

选样依据是近期缓存可用性，扩展推理前已冻结，未按结果筛选样本。详见 `reports/expanded_selection.json` 和 `reports/expanded_data_validation.json`。

## 如何阅读页面

选择片段和每列方法，共享时间轴、播放、逐帧和帧号跳转。默认展示真实输入 RGB + 从预测投影的 2D 骨架；官方烘焙 overlay 与四模式视频另有下载。3D 可拖动、缩放、平移；可切换手网格、骨架、场景点云、接触/可见性颜色、有效腕轨迹。默认隐藏官方填补，勾选后以半透明/虚线显示，评价数字不变。

相机空间与预测世界空间均从原始张量导出，单位 m。世界空间按每路预测 w2c 的逆矩阵转换；每次推理的世界坐标独立。联动交互相机只同步观察视角，不代表几何配准或双目融合。场景来自官方 G 锚帧深度及其预测 K；页面显示最近 G 锚帧的实际编号，没有生成不存在的逐帧深度。网格 778 点来自官方原生 195 点的固定插值，不是 MANO 参数回归。GT 列只显示独立标注骨架与投影，不冒充预测或场景。

## 复核材料

- `reports/inference_runs.json`：实际逐路命令、输入、GPU UUID、开始/结束时间、退出码及张量检查。
- `reports/*_metrics.json`：逐路主指标、含填补次级诊断、每帧曲线、K 核验、相机轨迹和接触指标。
- `reports/frozen_anatomical_mapping.json` 与 `reports/anatomical_slot_audit.jpg`：固定左右手槽位和关节顺序的证据。
- `reports/raw_artifacts.json`、`reports/CHECKSUMS.sha256`：原始输出的字节数、SHA256、下载位置。
- `logs/`：原始官方日志、资源快照和预处理问题修复记录。实际 `.pt` 完整保存在服务器 `outputs/<run_id>/inference_output.pt`，也提供逐路 release 归档。
- `scripts/evaluate_predictions.py` 独立实现指标；`test_contracts.py` 用解析坐标和已知偏移检验单位、w2c 逆变换、fill 排除、帧匹配与失败分母。

## 实际限制与失败

这是一组近期样本的复现诊断，不是官方整套测试集分数。原版接口是单视频，未实现联合双目；没有把两台相机交错成时间序列。固定外部视角和手持物体视角与训练/目标分布可能不同，结果中的错检、缺失、填补和尺度异常均保留，不重调阈值挑选最佳结果。

HO3D 的源采集 FPS 在现有缓存 provenance 中无法确认，按有序采样图像明确假设 10 FPS 输入，故不报告物理速度/加速度。Aria 使用已核验针孔校正、顺时针旋转及 768 方图，并按原硬件时间戳匹配 GT。

没有独立核验的场景深度 GT，因此不声称场景深度精度。HO3D 接触评价仅在原生 195 个 MANO 对应顶点上比较现有 binary contact 标签，阈值固定 0.5；其接触定义与模型训练语义未必一致，作为跨数据集诊断。非 HO3D 仅展示真实预测接触概率，不声称有接触真值。

## 来源与许可

- [EgoFound3R 官方实现](https://github.com/MINT-SJTU/EgoFound3R/tree/264f4df9fd42536aae4dba982d18272cc3334077)；代码与权重保持官方许可，本复现未修改上游源码。
- [HO3D 官方可视化及关节转换](https://github.com/shreyashampali/ho3d/blob/master/vis_HO3D.py)。OpenGL → OpenCV 为 diag(1,-1,-1)，固定 MANO → OpenPose21 关节重排。
- [MANO 指尖对应实现](https://github.com/hassony2/manopth/blob/master/manopth/manolayer.py)。仅用于核对内部关节/顶点指尖关系，不用 GT 优化关节排列。
- [HOT3D](https://facebookresearch.github.io/hot3d/) · [DexYCB](https://dex-ycb.github.io/)。数据和第三方模型/拓扑均保留原许可；此仓库提供选定样本的复现材料，不发布模型权重或完整数据集。
'''
    (r/'README.md').write_text(text)
    protocol='''# 复现协议与评价口径

## 固定输入边界

官方提交 `264f4df9fd42536aae4dba982d18272cc3334077`，原仓库保持 clean。合法现成 EgoFound3R 与 VGGT-Omega 权重的 SHA256 见 `reports/weight_transfer_verified_20261111.json`。只向官方 CLI 输入 RGB 视频及真实相机 K；GT 手、物体位姿、深度、相机外参存放在 `evaluation_ref/`，不传给推理。

DexYCB 两相机独立运行，未联合双目、未融合、未将相机交错成时间。HOT3D 单 RGB 与 HO3D 单 RGB 是单独列出的补充，不计入双目统计。外部 K 经官方 `--root-k-source external --root-k-file` 用于手 root solver 和投影；场景仍预测 K。评分前将模型缩放后的 K 恢复到原 RGB 尺寸并与源 K 逐项核对。

## 运行

H=10 FPS，G stride=5，chunk=2s，overlap=0.5s；官方 keep_aspect /512 输入；启用 --upsample-mano，关闭逐帧 overlay 图像文件，其余官方后处理保留。R 为输入视频帧率：Dex/Aria 30，HO3D 假设 10；R 可包含 H 之间的时间插值。官方缺失填补另有 mask，不能与普通时间插值混淆。

全部使用 L40S。逐路重新查询实际 GPU UUID 与可用内存；串行、文件锁避免重复执行；nice=10，CPU 线程2，MPS active thread 20%，PyTorch allocator 上限16GiB，选择至少22GiB空余的 L40S。不终止、不抢占其他健康作业，未用 H20。这不是独占硬件吞吐基准；运行时来自原生 summary。

```bash
R=/nvme-ssd2/zhaoyicheng/egofound3r-reproduction-20261111
PY=/nvme-ssd1/zhaoyicheng/egohsi3r-reproduction-20261104/envs/official/bin/python
$PY $R/scripts/execute_suite.py --root $R --manifest $R/data/input_manifest_expanded.json
$PY $R/scripts/audit_anatomy.py --root $R
$PY $R/scripts/evaluate_suite.py --root $R
$PY $R/scripts/test_contracts.py --official-repo $R/upstream
$PY $R/scripts/build_complete_site.py --root $R
```

原始每路官方参数完整记录于 `reports/inference_runs.json`。已完成输出经验证后复用；非空不完整目录拒绝覆盖，须先检查失败原因。

## 手、单位、坐标

原生手 slot0=left，slot1=right：由 Aria1976 多帧 RGB 解剖投影独立确认并固定，发布 schema 本身未提供语义名称；不按每帧最小 GT 误差交换槽位。21关节为 wrist，再 thumb/index/middle/ring/little 各4关节。以模型自己的指尖 joint 与已知 MANO 顶点对应矩阵核对顺序。

Canonical hand 是每帧 OpenCV 相机坐标，单位米。`pose_w2c_metric` 是 world→camera，取逆得到 c2w。世界系只用于可视化和逐路相机评价。相机轨迹比较仅消除第一有效相机的刚体坐标规范，不拟合尺度、Sim(3) 或轨迹。Dex 相机 GT 固定；Aria 相机由匹配硬件时间戳的官方姿态给出。

## 独立评价

Primary mask = official canonical valid AND finite AND NOT official missing-fill。GT mask 从原标注读取；无预测仍在有效 GT 分母。单手数据只评价已标注解剖侧；另一槽的错检数仍保留在逐槽计数。所有帧通过显式 source_frame_indices 对齐。GT 不进入模型，也不影响模型阈值、尺度和后处理。

- Raw MPJPE：原始相机空间、绝对 3D 关节欧氏距离，m→mm，无配准。
- Root-relative：各自减去 wrist 再算 MPJPE，揭示关节形状误差。
- PA：每只手每帧一个 proper-rotation Sim(3)，允许旋转、平移和尺度；不能用它代表绝对定位精度。
- PCK20/PCK50：提供条件有效预测比例与全有效 GT 比例，后者把缺失预测记为未命中。
- Coverage：有效关节/手帧数除以 GT 关节/手帧数，同时报告 official valid、fill、display、H-anchor 与非 H 的 R 帧数。
- 含填补结果只作 SECONDARY display diagnostic；不改变主覆盖率、主指标或失败分母。
- 每数据组按实际有效关节加权（wrist 按手帧、PA 按可 PA 关节），不得把各视频均值简单平均。
- HO3D contact：native195 IDs 对应 GT778 子集，概率阈值0.5固定，报告 P/R/F1、有效顶点覆盖率、全 GT 正类召回和距离 MAE；缺标签不补标签。

缺失的独立场景深度/接触参考明确标为不可评分，不把输入检查或 GT 展示当模型准确率。

## 失败与展示

HTTP401 是历史准备阶段已解决的问题。预处理的 Aria2022 缓存长度差异、HO3D 缺标注被真实记录并修复，未丢弃 RGB。官方网络成功退出与预测有效性是两回事：页面保留错检、零覆盖与填补。公开几何严格从真实 `.pt` 导出，GT 独立标识；公式测试数据只在临时目录，绝不导入网页。
'''
    (r/'PROTOCOL.md').write_text(protocol)
    (r/'status.md').write_text('# 当前状态\n\n13/13 路官方推理和独立评价已完成；网站由真实预测构建。权重传输已核验，历史401不再阻塞。最终在线交互验收结果见 reports/browser_acceptance.json。\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--skip-archives',action='store_true');a=ap.parse_args();r=a.root;docs=r/'docs'
    inputs=json.loads((r/'data/input_manifest_expanded.json').read_text());score=json.loads((r/'reports/suite_metrics.json').read_text());rows={x['run_id']:x for x in score['runs']};assert len(rows)==13
    evals=json.loads((r/'evaluation_ref/expanded_evaluation_manifest.json').read_text());gpaths={x['run_id']:Path(x['gt_path']) for x in evals['samples']}
    write_reports(r,score,inputs)
    man={'schema_version':'egofound3r.viewer.v1','title':'EgoFound3R · 官方实跑 / 已知 K','subtitle':'11 段 · 13 路 RGB · 2,080 帧 · 双 RGB 独立推理 + 单目补充','status':'complete','generated_at':datetime.now(timezone.utc).isoformat(),'default_columns':2,
      'notice':'已完成 13/13 路官方推理与独立评价。有效覆盖率必须与误差一起读；默认隐藏官方填补，勾选后半透明显示。DexYCB 是两相机独立运行，Aria/HO3D 是单 RGB 补充。',
      'provenance':'官方 clean 264f4df · 真实预测张量 · GT 仅评价 · 每路坐标独立',
      'suite_summary':score['datasets'],
      'protocol':[{'label':'模型输入','value':'仅 RGB + 已知 K；GT 手/物体位姿、深度、外参不进入推理。'},
         {'label':'双目边界','value':'两路单视频分别推理；未联合双目或几何融合。Dex 基线 0.631 m / 43.05°。'},
         {'label':'K 的作用','value':'external K 用于手部 root/projection；场景 K 仍由网络预测。'},
         {'label':'运行设置','value':'官方 2 s / overlap 0.5 s；H 10 FPS；G stride 5；R 含时间插值。'},
         {'label':'有效与填补','value':'严格有效 = 官方 valid ∧ finite ∧ ¬fill；填补单列，失败留在 PCK 分母。'},
         {'label':'3D 与接触','value':'米制相机 / 独立预测世界系；接触色为模型概率。778 顶点由官方195点固定插值。'}],
      'sources':[{'label':'官方代码','url':'https://github.com/MINT-SJTU/EgoFound3R'},{'label':'复现仓库','url':REPO},{'label':'原始预测','url':REPO+'/releases/tag/'+TAG}],
      'downloads':[{'label':'复现报告','url':'downloads/README.md'},{'label':'评价协议','url':'downloads/PROTOCOL.md'},{'label':'全量指标 JSON','url':'downloads/suite_metrics.json'},{'label':'指标 CSV','url':'downloads/metrics.csv'},{'label':'运行命令 / 日志索引','url':'downloads/inference_runs.json'},{'label':'数据校验','url':'downloads/expanded_data_validation.json'},{'label':'左右手核验','url':'downloads/anatomical_slot_audit.jpg'}],
      'sequences':[]}
    artifacts=[]
    for seq in inputs['sequences']:
      sid=seq['id'];dataset=seq['dataset'];dual=len(seq['views'])==2
      out={'id':sid,'label':seq['label'],'dataset':dataset,'frames':seq['frames'],'fps':seq['fps'],'frame_ids':seq.get('source_frame_ids',seq['frame_ids']),'timestamps':seq['timestamps_s'],
        'input_label':'双 RGB · 两路独立 · 已知 K' if dual else '单 RGB 补充 · 已知 K','description':seq.get('pixel_contract',seq['source_timing']),
        'metric_scope':'仅已标注解剖手；有效/GT 手帧与误差同时报告；— 表示无可评分预测。',
        'notes':[seq['source_timing'],'GT 参考是独立真实标注，不是预测。世界坐标列为本次推理的独立世界系；联动视角不做几何配准。','场景点云取最近的真实 G 锚帧；不代表逐帧深度，也未声称通过深度 GT 验证。'],
        'rgb':{},'methods':[],'downloads':[]}
      if dataset=='HO3D-v3':out['notes'].append('源 FPS 未知，10 FPS 为显式输入假设；无物理速度评价。缺手部标注帧保留推理。接触指标采用现有 HO3D binary 标签。')
      refs=[];worlds=[]
      for view in seq['views']:
        rid=sid+'__'+view['id'];metric=rows[rid];c=metric['primary']['coverage'];m=metric['primary']['metrics'];folder=Path('data')/sid/view['id'];dest=docs/folder;payload_path=r/'outputs'/rid/'inference_output.pt';payload=load_prediction(payload_path)
        cp(Path(view['video']),dest/'input.mp4');poster=Path(view.get('poster',str(Path(view['video']).parent/'poster.jpg')))
        if poster.exists():cp(poster,dest/'poster.jpg')
        cp(Path(view['K_file']),dest/'root_intrinsics.json');ih,iw=view['image_size_hw'];v=view['id'];label=('相机 A' if v=='cam_a' else '相机 B') if dual else 'RGB'
        out['rgb'][v]={'video':str(folder/'input.mp4'),'poster':str(folder/'poster.jpg'),'width':iw,'height':ih}
        vals=[{'id':'coverage','label':'有效覆盖率','value':100*c['fraction'],'unit':'%','direction':'higher'},
            {'id':'handframes','label':'有效 / GT 手帧','value':f"{c['evaluated_hand_frames']} / {c['gt_hand_frames']}"},
            {'id':'fill','label':'填补 / 已标注手帧','value':c['filled_annotated_hand_frames']},
            *[{'id':key,'label':title,'value':m[key],'unit':'mm','direction':'lower'} for key,title in [('mpjpe_raw_mm','绝对 MPJPE'),('mpjpe_root_relative_mm','去腕 MPJPE'),('mpjpe_pa_mm','PA MPJPE')]],
            {'id':'pck50','label':'PCK50 全 GT','value':100*metric['primary']['pck']['50mm']['all_gt_including_failures'],'unit':'%','direction':'higher'}]
        if metric['contact']['available']:vals.append({'id':'contact_f1','label':'接触 F1 · 有效顶点','value':metric['contact']['f1_on_valid']})
        for coordinate in ['camera','world']:
          geo_dir=dest/('prediction' if coordinate=='camera' else 'world')
          geometry=export_payload(payload,geo_dir,coordinate=coordinate,official_repo=r/'upstream',rgb_video=Path(view['video']),hand_order='left,right',overwrite=True,max_scene_points=4096)
          geometry['source'].update(inference_sha256=metric['source']['inference_sha256'],inference_filename='inference_output.pt');geometry['series']=metric['primary']['series'];geometry['joint_order_note']='OpenPose21 anatomical order independently audited; frozen_anatomical_mapping.json';geometry['hand_labels']=['左手 · slot 0','右手 · slot 1']
          save(geo_dir/'geometry.json',geometry)
          method={'id':'official_'+v+('' if coordinate=='camera' else '_world'),'label':'EgoFound3R · '+label+('' if coordinate=='camera' else ' · 预测世界系'),'kind':'official','view':v,'view_label':label,'status':'complete',
            'coordinate_system':('RGB 相机坐标' if coordinate=='camera' else '本次预测世界坐标')+' · m','geometry':str((geo_dir/'geometry.json').relative_to(docs)),
            'summary':f"已知 K · 有效 {c['evaluated_hand_frames']}/{c['gt_hand_frames']} GT 手帧 · 官方填补 {sum(metric['validity_counts']['official_fill_by_native_slot'])} 手帧（含未标注槽）",'metrics':vals if coordinate=='camera' else []}
          (out['methods'] if coordinate=='camera' else worlds).append(method)
        gp=gpaths.get(rid,r/'evaluation_ref'/sid/view['camera_id']/'gt.npz');reference=export_reference(gp,dest/'reference',Path(view['K_file']))
        if dataset=='HO3D-v3':reference['series']=[x for x in reference['series'] if 'speed' not in x['id']];save(dest/'reference/geometry.json',reference)
        refs.append({'id':'gt_'+v,'label':'GT 参考 · '+label+' · 仅评价','kind':'reference','view':v,'view_label':label,'status':'complete','geometry':str(folder/'reference/geometry.json'),'coordinate_system':'RGB 相机坐标 · m','summary':'真实标注骨架 / 2D 投影；无预测含义，未进入模型。','metrics':[]})
        for name in ['overlay.mp4','overlay_multimode.mp4']:
          cp(payload_path.parent/name,dest/name);out['downloads'].append({'label':label+(' 官方 overlay' if name=='overlay.mp4' else ' 四模式视频'),'url':str(folder/name)})
        cp(r/'reports'/f'{rid}_metrics.json',docs/'downloads'/f'{rid}_metrics.json');out['downloads'].append({'label':label+' 完整指标','url':'downloads/'+rid+'_metrics.json'})
        cp(payload_path.parent/'summary.json',docs/'downloads'/f'{rid}_summary.json');cp(r/'logs'/f'{rid}.log',docs/'downloads'/f'{rid}.log')
        archive=r/'artifacts/raw'/f'{rid}.tar.gz';archive.parent.mkdir(parents=True,exist_ok=True)
        if not a.skip_archives and not archive.exists():
          with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
            for path in [payload_path,payload_path.parent/'summary.json',r/'reports'/f'{rid}_metrics.json',r/'logs'/f'{rid}.log']:tar.add(path,arcname=rid+'/'+path.name)
        item={'run_id':rid,'prediction_bytes':payload_path.stat().st_size,'prediction_sha256':metric['source']['inference_sha256'],'archive_url':REPO+'/releases/download/'+TAG+'/'+archive.name}
        if archive.exists():item.update(archive_bytes=archive.stat().st_size,archive_sha256=sha256(archive))
        artifacts.append(item);out['downloads'].append({'label':label+' 原始张量归档','url':item['archive_url']})
        print('EXPORTED',rid,flush=True)
      out['methods']+=refs+worlds;man['sequences'].append(out)
    save(r/'reports/raw_artifacts.json',{'tag':TAG,'artifacts':artifacts})
    (r/'reports/CHECKSUMS.sha256').write_text('\n'.join(x['prediction_sha256']+'  outputs/'+x['run_id']+'/inference_output.pt' for x in artifacts)+'\n')
    for name in ['suite_metrics.json','metrics.csv','inference_runs.json','expanded_selection.json','expanded_data_validation.json','data_validation.json','frozen_anatomical_mapping.json','anatomical_slot_audit.jpg','anatomical_slot_audit.json','weight_transfer_verified_20261111.json','raw_artifacts.json','CHECKSUMS.sha256']:
        cp(r/'reports'/name,docs/'downloads'/name)
    for name in ['README.md','PROTOCOL.md']:cp(r/name,docs/'downloads'/name)
    cp(r/'data/input_manifest_expanded.json',docs/'downloads/input_manifest_expanded.json')
    save(docs/'manifest.json',man);(docs/'.nojekyll').touch()
    revision=hashlib.sha256((docs/'app.js').read_bytes()+(docs/'manifest.json').read_bytes()).hexdigest()[:12]
    text=(docs/'index.html').read_text();text,n=re.subn(r'src="app\.js(?:\?v=[^"]*)?"',f'src="app.js?v={revision}"',text);assert n==1;(docs/'index.html').write_text(text)
    save(r/'reports/site_build.json',{'status':'complete','sequences':11,'streams':13,'prediction_geometry_exports':26,'ground_truth_reference_exports':13,'real_rgb_frames':2080,'revision':revision,'generated_utc':datetime.now(timezone.utc).isoformat()})
    print('BUILD_COMPLETE',revision,flush=True)

if __name__=='__main__':main()
