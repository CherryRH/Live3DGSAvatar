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
    ("render", "画面", "render.yaml", "画面与帧率", 10),
    ("train", "在线更新", "train.yaml", "在线更新参数", 20),
    ("system", "系统", "system.yaml", "对象、路径与服务", 30),
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
       help="留空区域的颜色"),
    _f("render.scaling_modifier", "高斯缩放系数", "render", "float", order=20,
       editable=True, minimum=0.05, maximum=5.0, step=0.05,
       help="整体放大或缩小高斯"),
    _f("render.scale", "分辨率比例", "render", "float", order=30,
       editable=True, minimum=0.05, maximum=1.0, step=0.05,
       help="1.0 为原始分辨率，越低越流畅"),
    _f("render.target_fps", "帧数上限", "render", "float", order=40,
       editable=True, minimum=0, maximum=480, step=5, unit="FPS",
       help="每秒最多渲染帧数，0 为不限"),
    _f("render.encode", "推流编码", "render", "enum", order=45,
       editable=True, choices=("raw", "jpeg"),
       help="跨网络/SSH 用 jpeg（24 KiB/帧，raw 是 768 KiB）"),
    _f("render.encode_quality", "JPEG 质量", "render", "int", order=46,
       editable=True, minimum=1, maximum=100, step=5,
       help="85 约 24 KiB/帧；调高更清晰也更占带宽"),
    _f("render.status_interval_s", "状态推送间隔", "render", "float", order=50,
       editable=True, minimum=0.05, maximum=5.0, step=0.05, unit="s",
       help="状态更新间隔"),
    _f("render.sh_degree", "球谐阶数", "render", "int", order=60,
       reason="随模型而定",
       help="由模型决定，不可改"),
    _f("render.batch_size", "批大小（脚本）", "render", "int", order=70,
       editable=True, minimum=1, maximum=64, help="仅脚本使用"),
    _f("render.frames", "采样帧数（脚本）", "render", "int", order=80,
       editable=True, minimum=1, maximum=10000, help="仅脚本使用"),
    _f("render.max_batch_size", "训练预分配批上限", "render", "int", order=90,
       reason="训练用，需重启"),
    _f("render.max_gaussian_size", "训练预分配高斯上限", "render", "int", order=100,
       reason="训练用，需重启"),
    _f("render.num_streams", "训练并行流数", "render", "int", order=110,
       reason="训练用，需重启"),
    _f("model.network.tex_size", "UV 纹理边长", "render", "int", order=120,
       reason="模型结构，需重载"),
    _f("model.network.num_basis_in", "驱动参数维度", "render", "int", order=130,
       reason="模型结构，需重载"),
    _f("model.network.num_basis_blend", "blendshape 基数量", "render", "int",
       order=140, reason="模型结构，需重载"),
    _f("model.network.mlp_hidden", "MLP 隐藏层", "render", "text", order=150,
       reason="模型结构，需重载"),
    _f("model.network.use_weight_proj", "使用 MLP 权重投影", "render", "bool",
       order=160, reason="模型结构，需重载"),
    _f("dataset.use_shape_weight", "使用 shape 权重", "render", "bool", order=170,
       editable=True, help="仅 checkpoint 数据集", requires="reload"),
    _f("dataset.use_pose_weight", "使用 pose 权重", "render", "bool", order=180,
       editable=True, help="仅 checkpoint 数据集", requires="reload"),
    _f("dataset.pin_memory", "锁定内存", "render", "bool", order=190,
       editable=True, help="仅 checkpoint 数据集", requires="reload"),
    _f("test.dump_diff", "输出并排对比图", "render", "bool", order=200,
       editable=True, help="仅脚本使用"),

    # ------------------------------------------------- 训练 / 在线更新 --
    _f("train.online_update.enabled", "启用在线更新", "train", "bool", order=10,
       editable=True,
       help="关闭后不执行在线更新"),
    _f("train.online_update.update_color", "更新颜色", "train", "bool", order=20,
       editable=True,
       help="修正整体偏色"),
    _f("train.online_update.update_opacity", "更新不透明度", "train", "bool", order=30,
       editable=True,
       help="可能让画面发虚"),
    _f("train.online_update.color_l1", "颜色损失用 L1", "train", "bool", order=40,
       editable=True,
       help="对高光与抖动更稳健"),
    _f("train.online_update.time_budget_s", "时间预算", "train", "float", order=50,
       editable=True, minimum=0.1, maximum=600.0, step=0.5, unit="s",
       help="超过即停止"),
    _f("train.online_update.steps", "最大迭代步数", "train", "int", order=60,
       editable=True, minimum=1, maximum=100000, step=10,
       help="与时间预算取先到者"),
    _f("train.online_update.frames", "观测帧数", "train", "int", order=70,
       editable=True, minimum=1, maximum=1000, step=1,
       help="参与更新的帧数"),
    _f("train.online_update.frame_chunk", "每步用几帧", "train", "int", order=80,
       editable=True, minimum=1, maximum=64, step=1,
       help="显存紧张时调小"),
    _f("train.online_update.lr_color", "颜色学习率", "train", "float", order=90,
       editable=True, minimum=0.0, maximum=0.1, step=0.0005,
       help="颜色修正步长"),
    _f("train.online_update.lr_opacity", "不透明度学习率", "train", "float", order=100,
       editable=True, minimum=0.0, maximum=1.0, step=0.005,
       help="不透明度修正步长"),
    _f("train.online_update.mask_weight", "掩码外权重", "train", "float", order=110,
       editable=True, minimum=0.0, maximum=1.0, step=0.05,
       help="0 表示只看人物区域"),
    _f("train.pipeline.enabled", "启用训练管线", "train", "bool", order=120,
       reason="训练尚未接入"),
    _f("train.pipeline.iterations", "训练步数", "train", "int", order=110,
       editable=True, minimum=1, maximum=1_000_000,
       help="默认 5 万步；冒烟可设几千", requires="reload"),
    _f("train.pipeline.batch_size", "训练批大小", "train", "int", order=115,
       editable=True, minimum=1, maximum=256,
       help="显存不够就调小", requires="reload"),
    _f("train.pipeline.blend_start", "形变基起点", "train", "int", order=120,
       editable=True, minimum=0, maximum=1_000_000,
       help="前 N 步只用模板网格", requires="reload"),
    _f("train.pipeline.test_frames", "验证集帧数", "train", "int", order=122,
       editable=True, minimum=1, maximum=100_000,
       help="末尾多少帧留作验证", requires="reload"),
    _f("train.pipeline.eval_every", "验证间隔", "train", "int", order=126,
       editable=True, minimum=1, maximum=100_000,
       help="每 N 步验证一次", requires="reload"),
    _f("train.pipeline.patience", "早停耐心", "train", "int", order=127,
       editable=True, minimum=1, maximum=100,
       help="连续 N 次无提升就停", requires="reload"),
    _f("train.pipeline.min_delta", "提升阈值", "train", "float", order=128,
       editable=True, minimum=0.0, maximum=5.0, step=0.01, unit="dB",
       help="小于它不算提升", requires="reload"),
    _f("train.pipeline.save_every", "存档间隔", "train", "int", order=129,
       editable=True, minimum=1, maximum=1_000_000,
       help="每 N 步存一次", requires="reload"),
    _f("train.pipeline.prefetch_workers", "预取线程", "train", "int", order=131,
       editable=True, minimum=0, maximum=16,
       help="0 关闭；取数慢时调大", requires="reload"),
    _f("train.pipeline.mask_weight", "背景损失权重", "train", "float", order=125,
       editable=True, minimum=0.0, maximum=1.0, step=0.1,
       help="1=背景同权；0=只看前景（易出现背景发散）", requires="reload"),
    _f("train.pipeline.max_frames", "训练采样上限", "train", "int", order=130,
       reason="训练尚未接入"),

    # ---------------------------------------------------------- 系统 --
    # ⚠️ 这里**刻意没有** `subject` / `work_name`：
    #    它们不是配置项（system.yaml 里也没有），而是"当前加载了哪个模型"的
    #    运行时状态，由右侧「模型」下拉框 + `POST /api/model` 决定。
    #    放进设置面板会让人去改一个改不动、也不存在对应键的"配置"。
    _f("runtime.device", "运行设备", "system", "enum", order=30,
       editable=True, choices=("cuda", "cpu"),
       help="cuda 或 cpu，改完需重启", requires="restart"),
    _f("runtime.split", "数据集切分", "system", "enum", order=40,
       editable=True, choices=("all", "train", "test"),
       help="数据集切分方式", requires="reload"),
    _f("paths.data_root", "数据集根目录", "system", "path", order=50,
       editable=True, help="数据集都放在这个目录下；改完点「刷新列表」"),
    _f("paths.models_dir", "模型根目录", "system", "path", order=70,
       editable=True, help="模型都放在这个目录下；改完点「刷新列表」"),
    _f("paths.model_ply", "模型文件名", "system", "text", order=80,
       editable=True, help="相对模型目录；也可写绝对路径", requires="reload"),
    _f("paths.models_config", "模型配置文件名", "system", "text", order=90,
       editable=True, help="相对模型目录", requires="reload"),
    _f("paths.flame_dir", "FLAME 资产目录", "system", "path", order=100,
       editable=True, help="含 generic_model.pkl 与 flame_uv.npz",
       requires="restart"),
    _f("paths.output_dir", "产物根目录", "system", "path", order=110,
       editable=True, help="产物与缓存目录", requires="restart"),
    _f("paths.image_subdir", "人像子目录", "system", "text", order=120,
       editable=True, help="抠好人像（黑底）的目录名", requires="reload"),
    _f("app.host", "监听地址", "system", "text", order=130,
       editable=True, help="改完需重启", requires="restart"),
    _f("app.port", "端口", "system", "int", order=140,
       editable=True, minimum=1, maximum=65535, help="改完需重启", requires="restart"),
    _f("app.autostart", "打开页面自动推流", "system", "bool", order=150,
       editable=True, help="刷新页面后生效"),
    _f("app.preload_frames", "预载帧数", "system", "int", order=160,
       editable=True, minimum=-1, maximum=100000, help="启动时预载，-1 为全部", requires="restart"),
    _f("app.stats_log_interval_s", "后端日志间隔", "system", "float", order=170,
       editable=True, minimum=1.0, maximum=600.0, step=1.0, unit="s",
       help="终端日志间隔"),
    _f("smoke.max_frames", "冒烟测试帧数上限", "system", "int", order=270,
       editable=True, minimum=0, maximum=100000, help="仅自检脚本使用"),
    _f("tracking.enabled", "启用实时驱动", "system", "bool", order=30,
       editable=True,
       help="需要追踪服务在运行"),
    _f("tracking.max_age_s", "实时帧过期时间", "system", "float", order=31,
       editable=True, minimum=0.1, maximum=30.0, step=0.1, unit="s",
       help="超过此时长无新帧视为中断"),
    _f("tracking.host", "追踪服务地址", "system", "text", order=32,
       editable=True, help="改完需重启", requires="restart"),
    _f("tracking.port", "追踪服务端口", "system", "int", order=33,
       editable=True, minimum=1, maximum=65535,
       help="改完需重启", requires="restart"),
)

#: 存在于配置、但**不属于设置面板**的键。
#: 它们表示"当前加载了哪个模型"，由「模型」下拉框 + `POST /api/model` 决定，
#: 在设置面板里既改不动、也没有对应的 yaml 键。
RUNTIME_ONLY_KEYS = ("subject", "work_name")

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
