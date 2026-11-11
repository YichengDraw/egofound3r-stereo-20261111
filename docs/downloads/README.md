# EgoFound3R · 双 RGB / 已知内参

**状态：等待官方模型权重授权，尚无真实推理结果。** 两个官方下载端点均实际返回 HTTP 401。当前交付包含真实同步双 RGB 输入核对页、完整静态 3D viewer、数据准备与推理/导出脚本。页面中的预测和指标保持待运行状态。

[打开输入与准备状态页面](https://yichengdraw.github.io/egofound3r-stereo-20261111/)

## 数据

沿用 DexYCB 箱子（74 帧）和香蕉（72 帧），两台同步 RGB 相机，共 146 个时刻、292 幅图像。每路使用真实内参。相机对属于宽基线；具体配置和数据来源见 [PROTOCOL.md](PROTOCOL.md)。

官方模型采用单相机视频接口。本项目为两路独立的官方已知内参推理，不包含联合双目融合。手部已知内参与场景预测内参分别保留。

## 获得权重后继续

先用自己的 Hugging Face 账户分别取得以下官方模型的访问权限，然后在运行设备通过 Hugging Face 客户端登录。不要把 token 写进命令、仓库或聊天。

- <https://huggingface.co/Frank-F2021/EgoFound3R>
- <https://huggingface.co/facebook/VGGT-Omega>

运行步骤（项目根目录为 `RUN_ROOT`）：

```bash
# 依赖需满足 upstream/requirements.txt，另外安装 huggingface_hub
hf auth login
python scripts/download_weights.py --root RUN_ROOT
python scripts/run_official.py --root RUN_ROOT
```

`download_weights.py` 仅使用官方地址和客户端已有授权，记录实际版本及校验值。`run_official.py` 分别处理四段视频，复用已通过结果契约检查的完成输出，日志与原始预测保留在项目根目录。

## 文件

- `PROTOCOL.md`：输入信息、官方实现范围、窗口与后处理、坐标系和评价规则。
- `reports/weight_access.json`：官方下载状态与当前阻塞证据。
- `scripts/`：数据准备、官方下载、官方推理、网页导出与独立评价工具。
- `docs/`：GitHub Pages 页面，运行时无需第三方 CDN。

**准备脚本和界面通过检查，不等于模型已成功运行。** 完成推理后需实际检查官方输出、手侧/关节对应、评价数值和 3D 内容，再更新页面状态。

## 原始方法

[EgoFound3R 论文](https://arxiv.org/abs/2610.01210) · [官方代码](https://github.com/MINT-SJTU/EgoFound3R)

固定官方提交：`264f4df9fd42536aae4dba982d18272cc3334077`。本项目未修改官方模型实现。
