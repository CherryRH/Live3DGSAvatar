"""配置项的**单一事实来源**：给 GUI 设置面板提供分组、顺序、可编辑性与校验。

## 为什么要一张表

设置面板要同时做到三件事：

1. **全量展示** —— 用户能看到项目所有可配项，包括不能改的（否则会困惑
   "这个参数去哪了"）
2. **只放开一部分** —— 结构性的参数（模型结构、路径、设备）改了会让
   运行中的服务失效，必须锁住并说明原因
3. **按重要性/常用性排序** —— 而不是按 YAML 里的字母序

如果这些信息分散在前端和后端（前端写一遍控件，后端写一遍校验），
两边必然漂移：前端放开了后端拒绝的值、或后端加了参数前端看不见。
因此把**元数据、校验、持久化**都集中在这里，由 `GET /api/settings` 下发给前端。

## 纯逻辑

本模块只依赖 `config.Config`，**不依赖 FastAPI / torch** —— 可以直接单元测试，
也能在没有 GPU 的环境里验证校验规则。

## 可编辑 vs 锁定

| 标记 | 含义 | 应用方式 |
|---|---|---|
| `editable=True` | 改完立刻生效，不需要重载模型或重启 | 写入运行中的 `cfg` / `RenderParams` |
| `editable=False` | 结构性参数 | 面板里只读展示 + 给原因 |

**为什么限制得这么严**：路径与模型结构一旦变化，已加载的张量就与新配置不一致
（例如改 `tex_size` 会让 `face_id` 指向错误的高斯）。这类参数只能改配置文件后重启。

## 持久化写哪里

写 `configs/local.yaml`（已 gitignore，且在 `load_config` 的合并链末尾）。
**不写 `system.yaml` / `render.yaml` / `train.yaml`** —— 那些文件带大量注释，
程序化 `yaml.dump` 会把注释全部丢掉，把手写文档变成机器产物。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..config import Config, repo_root

#: 分组定义：`(键, 标题, 说明, 排序)`。排序值小的在前。
#: 分组 = **配置文件**。设置面板按此分为三个独立栏目（页签）。
#: `(键, 标题, 文件名, 说明, 排序)`
GROUPS: tuple[tuple[str, str, str, str, int], ...] = (
    ("render", "渲染 / 模型", "render.yaml",
     "出图参数、帧率与模型结构。可编辑项改完立即生效", 10),
    ("train", "训练 / 在线更新", "train.yaml",
     "在线更新的门控与超参", 20),
    ("system", "系统 / 服务", "system.yaml",
     "对象、路径、设备与服务端参数（结构性，改动需重启或重载）", 30),
)

#: 取值类型。`color` 是 3 个 [0,1] 分量；`path` 是字符串路径。
KINDS = ("bool", "int", "float", "color", "enum", "text", "path")


@dataclass(frozen=True)
class Field:
    """一个可展示的配置项。

    Attributes:
        key: 点分路径（与 `Config.get` 一致）
        label: 面板上显示的中文名
        group: `GROUPS` 中的键
        kind: `KINDS` 之一
        order: 组内排序（小的在前）；同值按定义顺序
        editable: 是否允许在面板里修改
        reason: `editable=False` 时说明为什么锁住（显示给用户）
        help: 补充说明（可选）
        minimum / maximum / step: 数值范围（前端据此生成输入控件）
        choices: `enum` 的候选值
        unit: 单位后缀（如 "s"、"ms"、"FPS"）
        requires: 生效所需动作，`None` 表示立即生效
    """

    key: str
    label: str
    group: str
    kind: str
    order: int = 100
    editable: bool = False
    reason: str = ""
    help: str = ""
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    unit: str = ""
    requires: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"{self.key}: 未知的 kind={self.kind!r}")
        if self.group not in {g[0] for g in GROUPS}:
            raise ValueError(f"{self.key}: 未知的 group={self.group!r}")
        if not self.editable and not self.reason:
            raise ValueError(
                f"{self.key}: 锁定的字段必须给出 reason —— "
                "否则用户只看到灰掉的输入框，不知道为什么")


def _f(*args, **kwargs) -> Field:      # 简写，让下面的表更紧凑
    return Field(*args, **kwargs)


# --------------------------------------------------------------- 字段表 --
#
# 排序原则：**最常改的排最前**。
# - 渲染：背景色/缩放是调画面时最常动的；批大小等是脚本用的，靠后
# - 训练：门控在最前（决定要不要跑），其次是预算，最后是学习率
# - 系统：先看得到"当前用的是哪个对象"，然后是路径、设备

FIELDS: tuple[Field, ...] = (
    # ---------------------------------------------------------- 渲染 --
    _f("render.background", "背景色", "render", "color", order=10,
       editable=True, step=0.05,
       help="数据集原图为黑底；换成白底可检查组件的轮廓"),
    _f("render.scaling_modifier", "高斯缩放系数", "render", "float", order=20,
       editable=True, minimum=0.05, maximum=5.0, step=0.05,
       help="整体放大/缩小每个高斯的尺度；1.0 表示不额外缩放"),
    _f("render.scale", "分辨率比例", "render", "float", order=30,
       editable=True, minimum=0.05, maximum=1.0, step=0.05,
       help="1.0 = 原生分辨率。显存不足时这是最有效的帧率旋钮"),
    _f("render.target_fps", "帧数上限", "render", "float", order=40,
       editable=True, minimum=0, maximum=480, step=5, unit="FPS",
       help="每秒最多渲染几帧；0 表示不限。实际帧率 = min(该值, 1/渲染耗时)"),
    _f("render.status_interval_s", "状态推送间隔", "render", "float", order=50,
       editable=True, minimum=0.05, maximum=5.0, step=0.05, unit="s",
       help="服务端推 status 的间隔；太小会占满控制通道"),
    _f("render.sh_degree", "球谐阶数", "render", "int", order=60,
       reason="与模型文件绑定的结构参数：模型只存了 DC 项，改了会读错列",
       help="本项目只用 SH 的 DC 项，恒为 0"),
    _f("render.batch_size", "批大小（脚本）", "render", "int", order=70,
       reason="只影响 scripts/ 下的批处理脚本，GUI 逐帧渲染不受它控制"),
    _f("render.frames", "采样帧数（脚本）", "render", "int", order=80,
       reason="只影响 render_test 等脚本的默认采样；GUI 帧数由驱动参数决定"),
    _f("render.max_batch_size", "训练预分配批上限", "render", "int", order=90,
       reason="训练用 BatchRasterizer 的预分配上限，需在构造时确定"),
    _f("render.max_gaussian_size", "训练预分配高斯上限", "render", "int", order=100,
       reason="同上；且必须与模型的高斯数匹配"),
    _f("render.num_streams", "训练并行流数", "render", "int", order=110,
       reason="训练用，构造时确定"),
    _f("model.network.tex_size", "UV 纹理边长", "render", "int", order=120,
       reason="模型结构：必须是 tex_size² ≥ 高斯数；改了会与 .ply 不匹配"),
    _f("model.network.num_basis_in", "驱动参数维度", "render", "int", order=130,
       reason="模型结构：必须与 .ply 的基数量一致"),
    _f("model.network.num_basis_blend", "blendshape 基数量", "render", "int",
       order=140, reason="模型结构：必须与 .ply 的基数量一致"),
    _f("model.network.mlp_hidden", "MLP 隐藏层", "render", "text", order=150,
       reason="模型结构：必须与 .ply 里存的权重形状一致"),
    _f("model.network.use_weight_proj", "使用 MLP 权重投影", "render", "bool",
       order=160, reason="模型结构：与训练时的一致性要求"),
    _f("dataset.use_shape_weight", "使用 shape 权重", "render", "bool", order=170,
       reason="参照格式数据集的读取选项；加载时确定", requires="reload"),
    _f("dataset.use_pose_weight", "使用 pose 权重", "render", "bool", order=180,
       reason="同上", requires="reload"),
    _f("dataset.pin_memory", "锁定内存", "render", "bool", order=190,
       reason="数据加载选项，影响很小"),
    _f("test.dump_diff", "输出并排对比图", "render", "bool", order=200,
       reason="只影响 render_test 脚本，可用 --dump-diff 临时开启"),

    # ------------------------------------------------- 训练 / 在线更新 --
    _f("train.online_update.enabled", "启用在线更新", "train", "bool", order=10,
       editable=True,
       help="总开关。关闭后 GUI 与脚本都不会执行在线更新"),
    _f("train.online_update.update_color", "更新颜色", "train", "bool", order=20,
       editable=True,
       help="颜色（SH DC）是外观偏差的主要载体；与「更新不透明度」不能同时关闭"),
    _f("train.online_update.update_opacity", "更新不透明度", "train", "bool", order=30,
       editable=True,
       help="修正「该透明处不透明」这类误差；关闭更稳"),
    _f("train.online_update.color_l1", "颜色损失用 L1", "train", "bool", order=40,
       editable=True,
       help="L1 对高光/镜面/跟踪抖动的离群像素更稳健，但收敛更慢"),
    _f("train.online_update.time_budget_s", "时间预算", "train", "float", order=50,
       editable=True, minimum=0.1, maximum=600.0, step=0.5, unit="s",
       help="墙钟上限，超过即停。实测约 50 步即收敛，10 s 是宽松上限"),
    _f("train.online_update.steps", "最大迭代步数", "train", "int", order=60,
       editable=True, minimum=1, maximum=100000, step=10,
       help="与时间预算取先到者"),
    _f("train.online_update.frames", "观测帧数", "train", "int", order=70,
       editable=True, minimum=1, maximum=1000, step=1,
       help="取多少帧做观测；帧太多会超预算"),
    _f("train.online_update.frame_chunk", "每步用几帧", "train", "int", order=80,
       editable=True, minimum=1, maximum=64, step=1,
       help="小于观测帧数时只用前若干帧；显存紧张时调小"),
    _f("train.online_update.lr_color", "颜色学习率", "train", "float", order=90,
       editable=True, minimum=0.0, maximum=0.1, step=0.0005,
       help="作用于 SH DC"),
    _f("train.online_update.lr_opacity", "不透明度学习率", "train", "float", order=100,
       editable=True, minimum=0.0, maximum=1.0, step=0.005,
       help="作用于 sigmoid 前的 logit；与颜色学习率差一个数量级"),
    _f("train.online_update.mask_weight", "掩码外权重", "train", "float", order=110,
       editable=True, minimum=0.0, maximum=1.0, step=0.05,
       help="0 = 只用前景。背景通常预测不准，忽略它能改善拟合"),
    _f("train.pipeline.enabled", "启用训练管线", "train", "bool", order=120,
       reason="训练代码由协作者提供，尚未接入（见 docs/ARCHITECTURE.md §6）"),
    _f("train.pipeline.max_frames", "训练采样上限", "train", "int", order=130,
       reason="训练未接入；接入后才会被读取"),

    # ---------------------------------------------------------- 系统 --
    _f("subject", "对象（subject）", "system", "text", order=10,
       reason="换了对象要重载模型；请用左侧「加载模型」按钮，"
              "它会把数据集与模型一起切过去", requires="reload"),
    _f("work_name", "工作名（work_name）", "system", "text", order=20,
       reason="同上：模型目录的一部分，需重载模型", requires="reload"),
    _f("runtime.device", "运行设备", "system", "enum", order=30,
       reason="模型、数据集、相机必须全在同一设备；切换需重启服务",
       choices=("cuda", "cpu"), requires="restart"),
    _f("runtime.split", "数据集切分", "system", "enum", order=40,
       reason="参照格式数据集的切分；加载时确定",
       choices=("all", "train", "test"), requires="reload"),
    _f("paths.data_root", "数据集根目录", "system", "path", order=50,
       reason="路径在加载时解析；改了需重载模型", requires="reload"),
    _f("paths.data_dir", "数据集目录（显式）", "system", "path", order=60,
       reason="同上；留空表示用 <data_root>/<subject>", requires="reload"),
    _f("paths.models_dir", "模型根目录", "system", "path", order=70,
       reason="模型查找的根；改了需重载模型", requires="reload"),
    _f("paths.model_ply", "模型文件（显式）", "system", "path", order=80,
       reason="同上；留空表示按 <models_dir>/<subject>/<work_name>/ 推导",
       requires="reload"),
    _f("paths.models_config", "模型配置（显式）", "system", "path", order=90,
       reason="模型随附 config.yaml 的显式路径", requires="reload"),
    _f("paths.flame_dir", "FLAME 资产目录", "system", "path", order=100,
       reason="只在需要 FLAME 前向（checkpoint 格式）时使用", requires="restart"),
    _f("paths.output_dir", "产物根目录", "system", "path", order=110,
       reason="脚本与缓存写入位置",
       help="拓扑缓存写在 <output_dir>/cache/"),
    _f("paths.image_subdir", "原图子目录", "system", "text", order=120,
       reason="数据集里原图的子目录名；作为 PSNR 的 GT", requires="reload"),
    _f("app.host", "监听地址", "system", "text", order=130,
       reason="绑定地址在启动时确定；改了需重启服务", requires="restart"),
    _f("app.port", "端口", "system", "int", order=140,
       reason="同上：监听端口需重启", requires="restart"),
    _f("app.autostart", "打开页面自动推流", "system", "bool", order=150,
       reason="前端在页面加载时读取；刷新页面后生效"),
    _f("app.preload_frames", "预载帧数", "system", "int", order=160,
       reason="服务启动时预载；-1 表示全部", requires="restart"),
    _f("app.stats_log_interval_s", "后端日志间隔", "system", "float", order=170,
       editable=True, minimum=1.0, maximum=600.0, step=1.0, unit="s",
       help="stdout 统计日志的节奏；与前端 FPS 解耦"),
    _f("smoke.max_frames", "冒烟测试帧数上限", "system", "int", order=270,
       reason="只影响自检脚本"),
)

#: 快速查表
BY_KEY: dict[str, Field] = {f.key: f for f in FIELDS}

#: 可编辑的键（供 API 快速判断）
EDITABLE_KEYS: frozenset[str] = frozenset(f.key for f in FIELDS if f.editable)


# ------------------------------------------------------------------ 值 --


def _coerce(f: Field, value: Any) -> Any:
    """把前端传来的值转成配置里的存储类型，并做**范围/类型**校验。

    Raises:
        ValueError: 类型不符或超出范围（信息面向用户，可直接显示）
    """
    if f.kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str) and value.lower() in ("true", "false", "1", "0"):
            return value.lower() in ("true", "1")
        raise ValueError(f"{f.label}：需要布尔值，实际 {value!r}")

    if f.kind == "int":
        try:
            v = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{f.label}：需要整数，实际 {value!r}") from None
        if isinstance(value, float) and not float(value).is_integer():
            raise ValueError(f"{f.label}：需要整数，实际 {value!r}")
        return _check_range(f, v)

    if f.kind == "float":
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{f.label}：需要数字，实际 {value!r}") from None
        return _check_range(f, v)

    if f.kind == "color":
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"{f.label}：需要 3 个分量，实际 {value!r}")
        out = []
        for c in value:
            try:
                x = float(c)
            except (TypeError, ValueError):
                raise ValueError(f"{f.label}：分量必须是数字，实际 {c!r}") from None
            if not (0.0 <= x <= 1.0):
                raise ValueError(f"{f.label}：各分量应在 [0, 1]，实际 {x}")
            out.append(x)
        return out

    if f.kind == "enum":
        v = str(value)
        if v not in f.choices:
            raise ValueError(
                f"{f.label}：只能是 {' | '.join(f.choices)}，实际 {v!r}")
        return v

    # text / path
    return None if value is None else str(value)


def _check_range(f: Field, v: float) -> float:
    if f.minimum is not None and v < f.minimum:
        raise ValueError(f"{f.label}：不能小于 {f.minimum}，实际 {v}")
    if f.maximum is not None and v > f.maximum:
        raise ValueError(f"{f.label}：不能大于 {f.maximum}，实际 {v}")
    return v


def validate_patch(patch: dict[str, Any]) -> dict[str, Any]:
    """校验一批待修改的键值，返回**已转换类型**的字典。

    Raises:
        ValueError: 含未知键、锁定键或非法值。信息面向用户，可直接显示。
    """
    out: dict[str, Any] = {}
    for key, value in patch.items():
        f = BY_KEY.get(key)
        if f is None:
            raise ValueError(f"未知的配置项：{key}")
        if not f.editable:
            why = f.reason or "该项不可在运行中修改"
            raise ValueError(f"{f.label} 已锁定：{why}")
        out[key] = _coerce(f, value)
    return out


def snapshot(cfg: Config) -> list[dict[str, Any]]:
    """导出全部字段的当前值 + 元数据，供前端渲染。

    返回的列表**已按 分组顺序 → 组内 order** 排好，前端不需要再排。
    """
    group_order = {g[0]: g[4] for g in GROUPS}
    ordered = sorted(FIELDS, key=lambda f: (group_order[f.group], f.order))

    out = []
    for f in ordered:
        value = cfg.get(f.key)
        if f.kind == "color" and value is not None:
            value = [float(x) for x in value]
        out.append({
            "key": f.key,
            "label": f.label,
            "group": f.group,
            "order": f.order,
            "kind": f.kind,
            "editable": f.editable,
            "reason": f.reason,
            "help": f.help,
            "unit": f.unit,
            "minimum": f.minimum,
            "maximum": f.maximum,
            "step": f.step,
            "choices": list(f.choices),
            "requires": f.requires,
            "value": _jsonable(value),
        })
    return out


def groups_payload() -> list[dict[str, Any]]:
    """分组元数据（标题 + 对应配置文件 + 说明），已排序。"""
    return [{"key": k, "title": title, "file": fname, "help": h, "order": o}
            for k, title, fname, h, o in sorted(GROUPS, key=lambda g: g[4])]


def _jsonable(value: Any) -> Any:
    """把 Path / tuple 等转成 JSON 可序列化的形式。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


# ------------------------------------------------------------ 持久化 --


def overrides_path() -> Path:
    """用户覆盖文件：`configs/local.yaml`（已 gitignore，合并链末尾）。"""
    return repo_root() / "configs" / "local.yaml"


def save_overrides(patch: dict[str, Any], path: Path | None = None) -> Path:
    """把一批已校验的值写进 `local.yaml`，与已有内容深合并。

    **只写覆盖项**，不复制整份配置 —— 这样 `local.yaml` 始终是"我改过什么"的
    最小记录，升级默认配置时不会互相打架。

    Raises:
        OSError: 写入失败（权限、只读文件系统）
    """
    import yaml  # noqa: PLC0415

    target = Path(path) if path is not None else overrides_path()
    existing: dict[str, Any] = {}
    if target.exists():
        try:
            loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, yaml.YAMLError):
            # 已有文件坏了：不静默吞掉，交给上层；这里选择覆盖重建
            existing = {}

    merged = _deep_merge(existing, _unflatten(patch))
    target.parent.mkdir(parents=True, exist_ok=True)
    header = ("# 由 GUI 设置面板写入（gitignore）。\n"
              "# 只记录被修改过的项；删除本文件即恢复默认。\n")
    target.write_text(
        header + yaml.safe_dump(merged, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    return target


def load_overrides(path: Path | None = None) -> dict[str, Any]:
    """读回 `local.yaml` 里的覆盖项（扁平化）；不存在返回空字典。"""
    import yaml  # noqa: PLC0415

    target = Path(path) if path is not None else overrides_path()
    if not target.exists():
        return {}
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return _flatten(data) if isinstance(data, dict) else {}


def _unflatten(patch: dict[str, Any]) -> dict[str, Any]:
    """`{"a.b": 1}` → `{"a": {"b": 1}}`。"""
    out: dict[str, Any] = {}
    for key, value in patch.items():
        parts = key.split(".")
        cur = out
        for p in parts[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[parts[-1]] = value
    return out


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """`{"a": {"b": 1}}` → `{"a.b": 1}`。"""
    out: dict[str, Any] = {}
    for key, value in data.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, full + "."))
        else:
            out[full] = value
    return out


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
