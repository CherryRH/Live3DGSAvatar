# data

`src/live3dgsavatar/data/` 负责"数据长什么样"和"怎么取一批"。

## 1. 数据集格式

`datasets/<对象>/` 有两种可驱动格式：

| 格式 | 判据 | 说明 |
|---|---|---|
| manifest | `render_inputs.npz` | 帧内已含网格、驱动参数、相机；训练/评测默认 |
| checkpoint | `checkpoint/*.frame` | 需要 FLAME 前向还原网格 |

共用可选目录：

```text
images/   抠图人像（RGBA），训练/评测目标
input/    原始画面，RDF 实时驱动优先使用
```

从视频构建：

```bash
bash build_dataset.sh --video videos/crh_1 --name crh_new
# 只调试前 300 帧：
bash build_dataset.sh --video videos/crh_1 --name crh_new --limit 300
```

视频入口依次做：

1. `ffmpeg` 按 `--fps`（默认 30）抽帧；
2. 用 U2Net 逐段估人像 bbox，取全片 p90 头宽作为稳定裁剪尺度；
   方形窗口跟随头心移动，默认 `head_factor=2.75`、顶部保留 8% margin；
   窗口超出原图宽度时用背景色补成方形（不做非等比缩放），头顶和脖子
   下方留出空白，不把画幅硬切到脖子；
3. 输出 `input/`（裁好的 RGB，供 RDF 跟踪）与 `images/`（裁好并抠图的 RGBA）；
4. RDF 在 `input/` 上跟踪，最后烘焙成 `datasets/<name>/`；
   `input/` 会复制保留，便于对照。
5. 最终 `images/` 用 BiRefNet-portrait（PyTorch）做人像 matting；裁切阶段仍用
   U2Net 估人像 bbox。BiRefNet 输出的软 alpha 更适合发丝、耳饰和眼镜边缘，
   不再做硬内缩。
6. 再用 SegFormer 衣服解析去掉 Upper-clothes/Dress，保留脸、头发、脖子和
   露出的胸口；肩部衣服变透明，不额外增加训练 mask。
   模型目录：`data/matting/birefnet_portrait/`、`data/matting/clothes_seg/`。
   下载：
   ```
   python -c "from huggingface_hub import snapshot_download; \
     snapshot_download('ZhengPeng7/BiRefNet-portrait', local_dir='data/matting/birefnet_portrait')"
   python -c "from huggingface_hub import snapshot_download; \
     snapshot_download('mattmdjaga/segformer_b2_clothes', local_dir='data/matting/clothes_seg')"
   ```
   说明：衣服解析模型来自 ATR human parsing；使用前确认其许可证适用于项目。

`datasets.py`：

- `inspect(cfg, name)`：读帧数、格式、是否有 images/input；
- `list_datasets(cfg)`：只返回可驱动数据集；
- `resolve_dataset_dir(cfg, name)`：不存在/不可用时给出明确错误；
- `data_root(cfg)`：展开 `paths.data_root`。

## 2. Scene

`scene.py` 把数据集读成渲染所需的统一结构：

```text
Scene.frames        list[dict]：mesh [V,3]、blend_weight [D]
Scene.faces         [F,3]
Scene.uvs           [Vuv,2]
Scene.uv_faces      [F,3]
Scene.K             [3,3]
Scene.R / T         相机外参
Scene.K_list        可选逐帧内参
Scene.dataset_frames 数据集总帧数
```

设备契约：`load_scene(device=...)` 后所有张量都在同一设备；调用方不要再逐个
`.to()`，漏掉一个就会触发 indices/device 不一致。

`load_scene(frames=...)` 可只载入部分帧，`total_frames` 仍保留数据集总数。

## 3. 样本源

`ManifestSource`：

- `split="train" | "test" | "all"`；
- `test_ratio` 默认 0.3，末尾 30% 作为留出集；
- 图像按需读取，带 LRU 缓存；`images.npy` 可选；
- 读取时把 mask 外填成背景色，训练侧只处理 RGB。

样本：

```text
mesh          [V,3] float32
blend_weight  [D] float32
image         [3,H,W] float32（已合成背景）
mask          [1,H,W] bool
```

批次 `Batch`：

```text
mesh / blend_weight / image / mask
weight        [B] 可选，逐样本损失权重
weight_map    [B,1,H,W] 可选，逐像素监督权重
K / w2c       [B,...] 可选，逐样本相机
```

`PrefetchSource` 用后台线程预取，减少 PNG 解码造成的 GPU 空转。

## 4. FLAME

- `flame_model.FlameModel`：加载 FLAME 模板、shape/expression 基，前向输出顶点；
- `flame_topology`：构建/缓存 UV 拓扑；10032 行对应 `face_id` 上界；
- `checkpoint_loader`：读 RDF dump 的 129 维参数：

```text
[0:100]    表情
[100:106]  头姿 rot6d
[106:112]  下颌 rot6d
[112:124]  眼球两组 rot6d
[124:126]  眼睑
[126:129]  平移
```

`mesh_and_weight()` 返回 `(mesh, blend_weight)`；rot6d 归一化后供前向与绑定共用。

## 5. 姿态

`pose.py`：

- `rot6d_to_matrix(d6)`：6D → 3×3；
- `head_yaw(blend_weight)`：`atan2(-R[2,0], R[2,2])`，度；
- `head_pitch(blend_weight)`：`asin(R[2,1])`，度；
- `angle_weights(yaw, scale=20, floor=1)`：大角度样本加权。

这些函数用于选角、统计与角度加权采样，不参与模型前向。

## 6. 模型目录

`models.py`：

- `models_root(cfg)`；
- `list_models(cfg)`：扫描 `models/<对象>/<版本>/model.ply`；
- `promote_model(source_ply, structure_config, report, target)`：把训练产物
  复制成正式模型目录，附带 `config.yaml` 和报告。

模型目录只认 `model.ply`；结构按 PLY 内嵌注释 → `config.yaml` → `configs/render.yaml` 的顺序读取。

## 7. RDF 导出

`tracking/tracked.py`：

- `load_rdf_dump(path)`：读 RDF 二进制/JSON dump；
- `to_frames(dump, flame_dir)`：转成 `mesh + blend_weight`；
- `frames_schema` / `load_frames_dir`：按目录批量读取；
- 数据集构建和实时驱动共用同一套参数解释。
