# app

`src/live3dgsavatar/app/` 是浏览器端后端；`web/` 是原生前端。

## 1. 协议

`protocol.py` 是纯函数，不依赖 FastAPI/torch。

### 二进制帧

```text
<B kind><I seq><f psnr> + payload
```

- 头 9 字节，`struct("<BIf")`；
- `kind`：`MSG_RENDER` / `MSG_TARGET` / `MSG_RENDER_JPEG` / `MSG_TARGET_JPEG`；
- `psnr` 与同一帧号绑定，避免 status 周期推送与画面错位；
- 原始帧 payload 为 RGB8 `H*W*3`；JPEG 帧 payload 为编码后字节。

`pack_frame` / `unpack_frame` / `pack_blob` / `expected_frame_bytes` 是编解码入口。

### 文本消息

- `config`：连接后的第一条，含 `version/width/height/channels/background/num_frames`；
- `status`：周期推送（帧号、FPS、时延、显存等）；
- `params`：客户端修改渲染/驱动参数；
- `frame`/`step`/`stream`/`capture` 等控制消息。

`merge_params(current, msg)` 校验并合并：

```text
render: background / scaling_modifier / scale / target_fps
view:   yaw / pitch
source: kind = checkpoint | live
drive:  frame / playing / amplitude / perturb
```

非法值抛 `ValueError`，由服务端转成前端错误，不崩服务。

### 背压与帧号

- `LatestSlot` / `asyncio.Queue(maxsize=1)`：只保留最新帧，渲染不阻塞；
- `FrameTimeline`：只有 `playing` 决定是否推进帧号；暂停仍继续渲染同一帧；
- `config_message` / `frame_interval` / `output_size` 为纯函数。

## 2. Session

`session.py`：

- 进程内单例；持有 `GaussianAvatar`、`Scene`、相机、光栅化器；
- `load_model(subject, work)`：加载 PLY 与数据集；失败时恢复旧选择；
- `RenderParams`：背景、缩放、目标 FPS、视角、对比开关；
- `set_drive_dataset(name)`：切换驱动数据集，校验模型/数据拓扑；
- `sync_drive_mode()`：数据集帧 / live 模式同步；
- `state()` / `live_brief()`：HTTP/status 数据；
- `apply_settings(patch)`：写回可编辑设置；
- live 未就绪或断线时回退到数据集首帧，保持画面不黑。

渲染路径统一走 `core/render/pipeline.render_frame`，避免各处组合不一致。

## 3. Server

`server.py`：

- FastAPI 应用与路由；
- `GET /api/state`、`/api/settings`、`/api/datasets`、`/api/models`；
- `POST /api/model`、`/api/settings`、`/api/dataset`；
- `GET /healthz`；
- WebSocket：推帧、接收 `params`/控制消息；
- JPEG 编码在进程内完成（`streaming.codec`），失败可回退 raw；
- 双栈 socket（`::` + IPv4）由 `make_dualstack_socket` 处理；
- 静态文件服务 `web/`，禁止缓存。

`app/__main__.py` 提供 `python -m live3dgsavatar.app`。

## 4. 设置

`settings.py`：

- 一张 `Field` 表定义配置面板的元数据：分组、顺序、类型、范围、是否可编辑；
- 模型结构等结构性参数锁定并给出原因；路径/服务参数可编辑，但标注生效方式；
- 可编辑参数分三类：
  - 立即生效：写入运行中的 `Config` / `RenderParams`；
  - `requires="reload"`：写入 `configs/local.yaml`，下次加载生效；
  - `requires="restart"`：写入 `configs/local.yaml`，重启服务生效；
- `validate_patch` 校验，`snapshot` 生成前端数据，`save_overrides` 写本地覆盖；
- 只写 `local.yaml`，不重写带注释的 `system/render/train.yaml`。

## 5. 推流编码

`streaming/codec.py`：

- `jpeg_available()`；
- `encode_jpeg(rgb, quality=85)`：Pillow 编码，失败时上层回退 raw；
- 协议本身不关心编码，只靠 `kind` 区分。
