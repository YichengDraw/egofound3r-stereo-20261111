# 复现协议与评价口径

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
