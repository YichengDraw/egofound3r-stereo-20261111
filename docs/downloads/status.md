# 当前状态

复现、独立评分、全部预测导出与线上交互验收已完成。13/13 路、2,080 RGB 帧；26 份预测几何、13 份 GT 参考；4,160 行逐帧左右手评分。

页面：https://yichengdraw.github.io/egofound3r-stereo-20261111/

真实失败保留：DexYCB 严格有效 1/292 手帧；Aria 3302 轨迹误差 320.30 mm / 22.56°。页面实际以软件 3D 回退验收，未验证 WebGL 性能。

验收与发布凭据：reports/browser_acceptance.json、reports/publication_receipt.json。权重授权和历史 HTTP 401 已解决；当前无外部阻塞。
