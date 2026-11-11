# EgoFound3R 独立复现与验收

更新：2026-11-11T11:23:33.901761+00:00。官方推理 **13/13 路完成**，11 段、2,080 帧 RGB，3 个数据来源。权重已合法传入且 SHA256 核验通过；历史 HTTP 401 已解决。

[交互页面](https://yichengdraw.github.io/egofound3r-stereo-20261111/) · [所有原始预测张量](https://github.com/YichengDraw/egofound3r-stereo-20261111/releases/tag/reproduction-20261111) · [逐路 CSV](reports/metrics.csv) · [完整 JSON](reports/suite_metrics.json) · [复现协议](PROTOCOL.md)

## 实测结果

| 数据 / 输入 | RGB 帧 | 有效 / GT 手帧 | 覆盖率 | Raw MPJPE mm | 去腕 MPJPE mm | PA MPJPE mm | PCK50 全 GT |
|---|---:|---:|---:|---:|---:|---:|---:|
| DexYCB | 292 | 1 / 292 | 0.3% | 360.99 | 152.67 | 41.10 | 0.0% |
| HOT3D-Aria | 890 | 1680 / 1780 | 94.4% | 27.97 | 19.56 | 9.83 | 84.2% |
| HO3D-v3 | 898 | 430 / 868 | 49.5% | 86.47 | 67.74 | 20.74 | 12.2% |

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
