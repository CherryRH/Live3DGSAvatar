# tracking

`src/live3dgsavatar/tracking/` 把外部 RDF 追踪服务接入为实时驱动。

## 1. 进程边界

追踪器依赖 **realdenseface** 环境，与本项目（`live3dgs`）不兼容，因此作为
独立进程运行：

```text
run_tracking.sh（realdenseface）
  ← 画面帧 ── RdfClient ── 本项目（live3dgs）
  → 参数 + 顶点
```

本项目不加载 RDF 的 FLAME2023；顶点由追踪服务算好再发回。

## 2. 二进制协议

`rdf_wire.py`，无握手，自描述。

请求：

```text
magic     4s   b"RDFQ"
seq       I
width     H
height    H
rgb       w*h*3 uint8
```

回复：

```text
magic     4s   b"RDFF"
seq       I
timestamp d
n_verts   H
n_exp     H
x         n_exp+18 f
verts     n_verts*3 f
```

- `x = [exp][pose 15][trans 3]`；
- `pose` 为 `[global, neck, jaw, eye_l, eye_r]`，每段 3 维轴角；
- 回复自带 `n_verts` / `n_exp`，客户端无需预知配置；
- `timestamp` 仅诊断；新鲜度用本端 `received_at`。

## 3. 客户端

`rdf_client.py`：

- `Observation(seq, timestamp, x, verts, received_at)`；
- `RdfClient.request(rgb, seq)`：一问一答，内部锁串行化；
- 断线抛 `ConnectionError` 并清连接，下次自动重连；
- 超时默认 10 s，追踪单帧约 60–70 ms。

`frame_source.py`：数据集帧源，支持 `input/` / `images/`、循环、fps 控制。

`live_source.py`：

- 独立取帧线程，按 `source_fps` 发送画面；
- 未连接/无观测时回退到数据集首帧；
- 维护当前 `Observation` 与帧号；
- `LiveUnavailable` 表示当前数据源不可用。

## 4. 参数映射

`param_map.py`：

```text
RDF x:        [exp 100][pose 15][trans 3]
blend_weight: [exp 100][head 6][jaw 6][eyes 12][eyelids 2][trans 3]
```

- 轴角 → rot6d（旋转矩阵前两行）；
- 头姿是 `global @ neck` 的复合；`compose_head=False` 可只看 global；
- 眼睑 RDF 不输出，填 0；
- RDF `trans` 在其相机空间，不能直接当世界平移；需要标定。

## 5. 对齐

`align.py`：

- `fit_similarity(src, dst)`：相似变换（尺度 + 旋转 + 平移）；
- `apply_similarity`；
- `fit_scale_about_camera`：绕相机中心缩放；
- `residuals`：对齐误差。

`mesh_build.py`：

- `kabsch(src, dst)`：SVD 求刚体变换；
- `TeethBuilder`：RDF 网格缺少牙齿时补齐模板牙齿；
- 输出可直接作为 `Mesh.verts`。

## 6. 运行

```bash
# 终端 A：追踪服务（realdenseface）
bash run_tracking.sh

# 终端 B：本项目
bash run_gui.sh
```

GUI 右侧选「RDF 实时追踪」。追踪未就绪时停在数据集首帧；播放/暂停/跳帧在
数据集与 live 两种模式下语义一致。
