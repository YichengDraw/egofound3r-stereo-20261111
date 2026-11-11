# 可视化结果格式

静态网站入口为 `index.html`，所有运行依赖都在 `vendor/`。页面通过 HTTP 加载 `manifest.json` 与几何文件；不依赖外部 CDN。以同步 RGB 视频为输入，以实际导出的几何和评估结果填充面板。

## manifest.json

- `schema_version`: `egofound3r.viewer.v1`。
- `title`, `subtitle`, `generated_at`, `provenance`: 展示信息。
- `status`: `complete` / `partial` / `blocked` / `awaiting_results`。非完成状态在标题旁永久显示。
- `status_label`, `notice`: 明确说明当前结果范围。`blocked` 应说明权重授权问题。
- `protocol`: `[{label, value}]` 输入条件与复现方式。
- `sources`, `downloads`: `[{label, url}]`。URL 相对于网站根目录。
- `sequences`: 每个片段包含以下内容。

```json
{
  "id": "dexycb_box",
  "label": "DexYCB · 箱子",
  "dataset": "DexYCB",
  "frames": 74,
  "fps": 30,
  "frame_ids": [0, 1],
  "timestamps": [0.0, 0.0333333333],
  "coordinate_system": "左相机坐标 · m",
  "input_label": "同步双 RGB · 已知 K",
  "description": "数据来源与模型输入说明",
  "notes": ["本片段的实际结论与限制"],
  "metric_scope": "评估样本范围、有效帧与对齐方式",
  "rgb": {
    "left": {"video": "data/box/left.mp4", "width": 640, "height": 480, "K": [[1,0,0],[0,1,0],[0,0,1]]},
    "right": {"video": "data/box/right.mp4", "width": 640, "height": 480}
  },
  "methods": [
    {
      "id": "official_left",
      "label": "EgoFound3R · 左 RGB + 已知 K",
      "kind": "official",
      "view": "left",
      "status": "complete",
      "summary": "该方法实际使用的输入与处理",
      "geometry": "data/box/official_left/geometry.json",
      "metrics": [{"id":"mpjpe", "label":"MPJPE", "value":null, "unit":"mm", "direction":"lower"}]
    }
  ]
}
```

上面的时间数组是格式说明的缩略写法，真实文件需包含 `frames` 个元素。`K` 必须填入真实标定；示例单位矩阵只说明维度，不能用于结果。

方法 `kind` 可为 `official`、`adapted`、`reference`、`baseline`。两路独立官方推理都应标为 `official`；`adapted` 仅用于实际实现且清楚说明的双目适配。

缺少预测时，设置 `status: "blocked"`，提供 `missing_reason`，省略 `geometry` 和虚构指标。真实输入视频仍正常同步播放，三维区域显示原因。`video` 可以覆盖该方法使用的视频；`overlay_video` 可以指定已经渲染投影的视频。视频时间零点不同可设置 `video_offset_s`。视频、几何、评估与索引文件相对路径均以网站根目录解析。

## geometry.json

```json
{
  "frame_count": 74,
  "hand_count": 2,
  "hand_order": ["left", "right"],
  "vertex_count": 778,
  "joint_count": 21,
  "coordinate_convention": "opencv",
  "coordinate_system": "左相机坐标 · m",
  "units": "m",
  "vertices": {"url":"vertices.bin", "dtype":"float32", "shape":[74,2,778,3]},
  "joints": {"url":"joints.bin", "dtype":"float32", "shape":[74,2,21,3]},
  "valid": {"url":"valid.bin", "dtype":"uint8", "shape":[74,2]},
  "faces": {
    "left": {"url":"faces_left.bin", "dtype":"uint32", "shape":[1538,3]},
    "right": {"url":"faces_right.bin", "dtype":"uint32", "shape":[1538,3]}
  },
  "contact": {"url":"contact.bin", "dtype":"float32", "shape":[74,2,778]},
  "visibility": {"url":"visibility.bin", "dtype":"float32", "shape":[74,2,778]},
  "joints2d": {"url":"joints2d.bin", "dtype":"float32", "shape":[74,2,21,2]},
  "fill_mask": {"url":"fill_mask.bin", "dtype":"uint8", "shape":[74,2]},
  "scene": {
    "frame_count": 15,
    "point_count": 4096,
    "frame_indices": [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70],
    "points": {"url":"scene_points.bin", "dtype":"float32", "shape":[15,4096,3]},
    "colors": {"url":"scene_colors.bin", "dtype":"uint8", "shape":[15,4096,3]},
    "valid": {"url":"scene_valid.bin", "dtype":"uint8", "shape":[15,4096]}
  },
  "series": [{"id":"wrist_error", "label":"手腕位置误差", "unit":"mm", "values":[null,null]}]
}
```

这些数字说明数组维度。实际 `faces` 数量由模型输出决定；不在网站内生成、补造任何几何或指标。

- 数组文件为小端、连续 C order、无文件头。支持 `float32`、`float64`、`uint8`、`uint16`、`uint32`、`int32`。数组也可直接内嵌为 JSON 数组或 `{data, dtype}`。
- 所有二进制 URL 相对于对应 `geometry.json` 解析。数组文件长度必须精确匹配 `shape`。
- 顶点、关节、点云必须使用同一坐标系。`opencv` 显示时统一翻转 Y/Z，保留左右手和真实几何关系；`y_up` 保持原方向。`mm` 会统一换成米显示。
- 跨方法联动相机应使用同一坐标系。未对齐的相机坐标须在页面文字说明，用户可关闭联动。
- `valid[T,H]` 决定手部是否显示；缺失结果不要填零坐标假装有效。
- `contact`、`visibility` 为 0–1 概率。缺少这些输出时保持左右手颜色，并明确提示缺失。
- `joints2d` 是当前展示 RGB 的原始像素坐标，默认顺序为常用 21 关节骨架。若为 0–1 坐标，设置 `joints2d_normalized: true`。
- 只有在 `joints` 真正为对应 RGB 相机坐标时，才可省略 `joints2d` 并设置 `projection_space: "camera"`、`projection_K`。世界坐标不能直接用 K 投影。
- `fill_mask` 只表示官方输出的填补状态，不等价于网络推理频率。可选 `observed[T,H]` 必须确实标注直接网络输出，才会显示网络帧标签。
- 点云可只包含部分帧。`scene.frame_indices` 对应整体时间轴中的索引；播放时选最近的已导出点云帧。无 `frame_indices` 时顺序对应手部帧；只有一帧时视为固定点云。
- 动态点云固定每帧 `point_count`，不足的点使用 `valid=0`，不要将零点当作真实重建。
- `series.values` 长度应为几何帧数，可为数组或二进制数组。缺失数值使用 JSON `null` 或 binary NaN，不使用零替代。
- 每方法支持最多前三条逐帧指标，底部曲线展示第一条有数据的指标。

## 页面操作

共用播放、慢放、逐帧和跳帧；每列独立选择方法。鼠标拖动旋转，滚轮缩放，右键平移。勾选“联动视角”时，各列复制相机状态。可以开关点云、手部网格、骨架、腕部轨迹以及接触/可见概率配色。每列“⌖”把当前手部重新居中。

## 依赖与授权

`vendor/three.module.js`、`vendor/OrbitControls.js` 来自 Three.js 0.170.0（MIT，完整许可证见 `vendor/LICENSE.three.txt`）。官方模型、数据和导出几何遵循各自许可；页面的下载链接只指向获得授权且可分发的材料。
