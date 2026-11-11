# EgoFound3R output tools

These scripts consume **actual official inference output**. They do not run a
model, download weights, produce replacement predictions, or silently locate GT.
The audited upstream revision is
[`264f4df9fd42536aae4dba982d18272cc3334077`](https://github.com/MINT-SJTU/EgoFound3R/tree/264f4df9fd42536aae4dba982d18272cc3334077).

## Export the comparison website assets

Run in the inference environment (PyTorch, NumPy and OpenCV):

```bash
python export_viewer.py \
  --input /path/to/outputs/inference_output.pt \
  --output /path/to/site/assets/sequence/camera/world \
  --rgb-video /path/to/exact-input.mp4 \
  --coordinate world \
  --official-repo /path/to/EgoFound3R
```

For the camera-space panel use another output directory and `--coordinate camera`.
`--official-repo` is only needed when the prediction did not already include
`--upsample-mano` output. It uses the official `mano_195_to_778.npz` asset and
preserves all 195 native predicted vertices exactly.

The script writes `geometry.json` last, after all raw little-endian arrays. Each
array uses `{url, dtype, shape, byte_length, byte_order}`. Filenames are relative
to `geometry.json`. `float32`/`uint32` are little-endian and `uint8` is unsigned.

| Field | Shape / meaning |
|---|---|
| `vertices` | `[T,2,778,3]`, meters; 778 is fixed interpolation of 195 predictions |
| `joints` | `[T,2,21,3]`, native joint order, meters |
| `valid` | `[T,2]`; canonical validity, excluding fill and nonfinite data |
| `display_valid` | `[T,2]`; official display availability, may include fill |
| `fill_mask` | `[T,2]`; official filling marker, not a successful fresh detection |
| `official_valid` | `[T,2]`; original canonical validity before export masking |
| `network_anchor` | `[T]`; frames at which the hand network is sampled, not detection success |
| `faces.left/right` | Same official MANO connectivity; double-sided rendering recommended |
| `joints2d` | `[T,2,21,2]`, pixels in the original RGB video; invalid entries are NaN |
| `contact/visibility` | Optional `[T,2,778]` probabilities |
| `contact_distance` | Optional `[T,2,778]`, meters |
| `scene.points/colors/valid` | `[T_G,N,3]` / `[T_G,N,3]` / `[T_G,N]` |
| `scene.frame_indices` | Exact R-frame positions of each scene anchor |
| `series` | `{id,label,unit,values}` with null for unavailable samples |

### Coordinate and calibration rules

- Official canonical hand tensors are **camera coordinates in meters**, while
  `camera.pose_w2c_metric` maps world to camera. World export applies its inverse
  to **both** hands and scene points.
- Depth uses the official predicted **scene** `K_global`. Known external K is
  used by the official hand root solver/projection only, not substituted into
  scene back-projection.
- The 2D overlay uses `hand_projection_K_full`, scaled from model-image pixels
  to original-video pixels. It is computed from camera-space hands even for a
  world-space 3D export.
- RGB colors come from sequentially decoded exact input frames resized using
  the official whole-image convention. Without `--rgb-video`, colors are
  explicitly marked neutral gray, and 2D image size uses provenance metadata.
- Invalid scene points are NaN plus `scene.valid=0`: the renderer must omit them.
- Camera-invalid frames are excluded from world display. No GT camera is read.
- Source frame IDs, actual G-anchor indices, file hashes and upstream provenance
  are retained. No images are generated for missing predictions.

### Native hand/joint ordering

The upstream source explicitly defines `OPENPOSE_HAND_BONES`; joint 0 is the
wrist. The released inference code does not explicitly name anatomical
left/right slot order. The exporter preserves both slots and all joint indices;
it does not infer anatomy from a color or silently swap slots. Default labels
are `native slot 0/1`. `--hand-order left,right` or `right,left` may be supplied
only after independent verification. Those labels do not reorder coordinates.
Mesh face keys are required by the viewer schema and do not prove hand order.

## Score against explicitly supplied ground truth

```bash
python evaluate_predictions.py \
  --input /path/to/outputs/inference_output.pt \
  --gt /path/to/evaluation_ref/gt.npz \
  --mapping-json /path/to/verified_mapping.json \
  --output /path/to/site/assets/sequence/metrics.json
```

GT NPZ requirements:

- `joints`: `[T,Hgt,Jgt,3]`;
- `valid`: `[T,Hgt,Jgt]` or `[T,Hgt]`;
- `source_frame_indices`: unique integer frame IDs;
- `units`: scalar string `m` or `mm`;
- `coordinate_frame`: scalar string `camera`.

Mapping must provide `pred_to_gt_hand` (length 2, `-1` for an unannotated slot)
and `pred_to_gt_joint` (length 21). It may be embedded in NPZ or supplied through
`--mapping-json`; the JSON can also include an `evidence` string. **No concrete
hand mapping is supplied here because native left/right slots have not been
verified from real inference.** The evaluator refuses to guess mappings or
pick the better-scoring prediction slot. It also refuses a mapping that omits
an annotated GT hand.

Scores are camera-space absolute MPJPE, wrist-relative MPJPE, proper-rotation
per-frame Sim(3) PA-MPJPE, and wrist position error (all mm). No alignment is
applied to the absolute score. Official filled predictions are excluded.
Coverage counts and PCK with failures in the denominator are reported alongside
conditional error/PCK; an unavailable GT frame is not dropped from the denominator.
This scorer does not claim to reproduce all metrics or dataset splits of the
paper. GT contact, depth, world-space metrics and unverified mesh correspondences
are intentionally not inferred from absent labels.

## Completed verification

```bash
python test_contracts.py --official-repo /path/to/EgoFound3R
python -m py_compile official_contract.py export_viewer.py evaluate_predictions.py
```

The analytical tests verify nonidentity camera inversion, exact G/R index
mapping, official upsampling anchor preservation, binary layout, world-panel
camera-space projection, missing-fill exclusion, retained GT denominators and
mandatory explicit joint/hand mapping. Fixtures only exist in memory and a
temporary directory which is removed; they are **not research results** and are
never exported into the website. Model inference, actual `.pt` deserialization
and real RGB video coloring still require real outputs and their runtime.

## Official implementation references

- `egohandmetric_prompt/inference_contracts.py`: canonical units and camera/hand/depth fields.
- `egohandmetric_prompt/inference_intrinsics.py`: external K and projection stream.
- `egohandmetric_prompt/hand_upsampling.py`: fixed interpolation and faces.
- `egohandmetric_prompt/data/marker_vertex_joint_map.py`: native skeleton edges.
- `infer_marker_multiclip.py`: source indices, global anchors and missing-fill behavior.

EgoFound3R code and model weights have their own upstream CC BY-NC-SA 4.0 license;
third-party VGGT-Omega and MeshGraphormer components retain their respective terms.
