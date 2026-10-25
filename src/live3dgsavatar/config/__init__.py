"""配置加载：把 `configs/*.yaml` 变成可点访问的对象。

**为什么放在 `src/live3dgsavatar/config/` 而不是 core/**
    `core/` 是纯计算层，不允许依赖配置、路径、YAML（见
    `docs/ARCHITECTURE.md` §2.3）。配置属于**应用层**基础设施。

用法::

    from live3dgsavatar.config import load_config

    cfg = load_config()                 # 读仓库根的 configs/
    cfg.paths.data_root                 # 已解析为绝对 Path
    cfg.get("render.batch_size", 4)     # 带默认值的点路径访问
    cfg.render.batch_size = 8           # 可写（就地覆盖默认量）

优先级（高 → 低）::

    命令行参数  >  环境变量  >  configs/*.yaml  >  代码内置兜底值

`configs/local.yaml` 若存在，会在 `system.yaml` / `render.yaml` 之后**深合并**，
用于放本机私有覆盖（该文件已被 .gitignore 忽略）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "Config", "load_config", "repo_root", "CONFIG_DIR",
    "model_dir", "resolve_model_ply",
    "resolve_model_config",
]

# 仓库根：src/live3dgsavatar/config/__init__.py → 上溯 4 层
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"

# 环境变量 → 点路径。便于换机器 / CI 而不改文件。
_ENV_OVERRIDES = {
    "LIVE3DGS_DATA_ROOT": "paths.data_root",
    "LIVE3DGS_OUTPUT_DIR": "paths.output_dir",
    "LIVE3DGS_MODELS_DIR": "paths.models_dir",
    "LIVE3DGS_SUBJECT": "subject",
    "LIVE3DGS_WORK_NAME": "work_name",
    "LIVE3DGS_DEVICE": "runtime.device",
}

# 代码内置兜底：仅当 YAML 缺失该字段时生效。
# **刻意保持最小**——真正的默认量应当在 configs/*.yaml 里，便于用户查看与修改。
_FALLBACK: dict[str, Any] = {
    # 对象与版本**不设默认值**：由界面选择，未选时为空。
    # 写死具体名字会让"默认配置"与仓库状态耦合 —— 一改就产生差异。
    "subject": "",
    "work_name": "",
    "paths.data_root": "datasets",

    "paths.output_dir": "output",
    "paths.models_dir": "models",
    "paths.flame_dir": None,
    # 文件名（相对模型目录）；写绝对路径则原样使用
    "paths.model_ply": "model.ply",
    "paths.models_config": "config.yaml",
    "paths.image_subdir": "images",
    "runtime.device": "cuda",
    "app.host": "localhost",
    "app.port": 8000,
    "app.autostart": True,
    "app.preload_frames": -1,
    "app.stats_log_interval_s": 10.0,
    "runtime.split": "all",
    "smoke.max_frames": 0,
    "tracking.enabled": True,
    "tracking.host": "127.0.0.1",
    "tracking.port": 8980,
    "tracking.max_age_s": 1.0,
    "render.background": [0.0, 0.0, 0.0],
    "render.scaling_modifier": 1.0,
    "render.sh_degree": 0,
    "render.target_fps": 60,
    "render.status_interval_s": 0.25,
    "render.encode": "jpeg",
    "render.encode_quality": 85,
    "render.scale": 1.0,
    "render.batch_size": 1,
    "render.frames": 1,
    "render.max_batch_size": 10,
    "render.max_gaussian_size": 60353,
    "render.num_streams": 3,
    "model.network.tex_size": 256,
    "model.network.num_basis_in": 129,
    "model.network.num_basis_blend": 20,
    "model.network.mlp_hidden": [128, 128],
    "model.network.use_weight_proj": True,
    "dataset.use_shape_weight": True,
    "dataset.use_pose_weight": True,
    "dataset.pin_memory": False,
    "test.dump_diff": False,
    # ---- 训练 / 在线更新（configs/train.yaml）----
    "train.pipeline.enabled": False,
    "train.pipeline.iterations": 50_000,
    "train.pipeline.batch_size": 10,
    "train.pipeline.blend_start": 3_000,
    "train.pipeline.test_frames": 350,
    "train.pipeline.mask_weight": 1.0,
    "train.pipeline.eval_every": 1_000,
    "train.pipeline.patience": 3,
    "train.pipeline.min_delta": 0.05,
    "train.pipeline.save_every": 5_000,
    "train.pipeline.prefetch_workers": 2,
    "train.pipeline.max_frames": None,
    "train.online_update.enabled": True,
    "train.online_update.update_opacity": True,
    "train.online_update.color_l1": False,
    "train.online_update.update_color": True,
    "train.online_update.time_budget_s": 10.0,
    "train.online_update.steps": 300,
    "train.online_update.lr_color": 0.002,
    "train.online_update.lr_opacity": 0.02,
    "train.online_update.mask_weight": 0.0,
    "train.online_update.frames": 8,
    "train.online_update.frame_chunk": 8,
}

# 需要解析为绝对路径的字段（相对路径按**仓库根**解析，而非 cwd）
_PATH_FIELDS = (
    "paths.data_root",
    "paths.output_dir",
    "paths.models_dir",
    # 目录型字段：必须解析成绝对路径，否则脚本从别处运行会按 cwd 解析
    "paths.flame_dir",
)

# 仓库根下不存在的相对路径，按 cwd 解析更符合直觉的字段。
# （例如 `--output-dir out` 通常指用户当前目录）
_CWD_RELATIVE = ("paths.output_dir",)


class Config(dict):
    """支持点路径访问的嵌套配置。

    - `cfg.paths.data_root` 等价于 `cfg["paths"]["data_root"]`
    - 嵌套 dict 会自动包装成 `Config`
    - `.get("a.b.c", default)` 支持点路径
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(
                f"配置项 {name!r} 不存在。可用键：{sorted(self.keys())}") from exc
        return _wrap(value)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        """按**点路径**取值，与 `dict.get` 兼容（无点时行为相同）。"""
        if "." not in key:
            return _wrap(super().get(key, default))
        node: Any = self
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return _wrap(node)

    def set(self, key: str, value: Any) -> None:
        """按点路径赋值，中间层不存在时自动创建。

        路径类字段（`_PATH_FIELDS`）会自动转成绝对 `Path`，
        与 `load_config()` 的处理保持一致 —— 否则 CLI 传入的相对路径
        会在不同 cwd 下解析出不同结果。
        """
        parts = key.split(".")
        node: Any = self
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = Config()
            node = node[part]
        if key in _PATH_FIELDS and value is not None:
            value = _resolve_path(value, key)
        node[parts[-1]] = value

    def as_plain(self) -> dict:
        """转回普通 dict（便于写 JSON / 打印）。"""
        return _unwrap(self)

    def resolved(self) -> dict:
        """返回一份便于阅读的最终生效配置（Path 转 str）。"""
        def enc(v: Any) -> Any:
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, dict):
                return {k: enc(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [enc(x) for x in v]
            return v

        return enc(self.as_plain())

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"Config({self.as_plain()!r})"


def _wrap(value: Any) -> Any:
    return Config(value) if isinstance(value, dict) else value


def _unwrap(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _unwrap(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_unwrap(v) for v in value]
    return value


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并：`override` 里的标量覆盖 `base`，嵌套 dict 逐层合并。"""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml  # 延迟导入：未装 PyYAML 时只有真正读配置才报错

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} 顶层必须是映射（mapping），实际是 {type(data).__name__}")
    return data


def _resolve_path(value: Any, field: str) -> Any:
    """把路径字段解析为绝对 `Path`。

    - `~` 展开
    - 绝对路径原样
    - 相对路径：`output_dir` 按 **cwd**（用户直觉），其余按**仓库根**
    """
    if value is None:
        return None
    p = Path(str(value)).expanduser()
    if p.is_absolute():
        return p
    base = Path.cwd() if field in _CWD_RELATIVE else REPO_ROOT
    return (base / p).resolve()


def load_config(
    config_dir: Path | str | None = None,
    files: Iterable[str] = ("system.yaml", "render.yaml", "train.yaml",
                            "local.yaml"),
    env: bool = True,
) -> Config:
    """加载配置。

    Args:
        config_dir: 配置目录；默认仓库根的 `configs/`
        files: 按顺序加载并深合并的文件名；缺失的自动跳过
        env: 是否应用环境变量覆盖

    Raises:
        ValueError: 必填项缺失或类型不符（例如缺 `paths.data_root`）
    """
    cdir = Path(config_dir).resolve() if config_dir else CONFIG_DIR

    merged: dict[str, Any] = {}
    for name in files:
        merged = deep_merge(merged, _load_yaml(cdir / name))

    cfg = Config(merged)

    # 内置兜底：只填 YAML 里没有的键
    for key, value in _FALLBACK.items():
        if cfg.get(key, _MISSING) is _MISSING:
            cfg.set(key, value)

    if env:
        for var, key in _ENV_OVERRIDES.items():
            if var in os.environ:
                cfg.set(key, os.environ[var])

    # 路径解析
    for field in _PATH_FIELDS:
        if cfg.get(field, None) is not None:
            cfg.set(field, _resolve_path(cfg.get(field), field))

    _validate(cfg)
    return cfg


_MISSING = object()


def _validate(cfg: Config) -> None:
    data_root = cfg.get("paths.data_root")
    if data_root is None:
        raise ValueError(
            "缺少 paths.data_root。请在 configs/system.yaml 中设置，"
            "或设环境变量 LIVE3DGS_DATA_ROOT。")

    for key, expect in (("render.batch_size", int),
                        ("render.frames", int),
                        ("model.network.tex_size", int),
                        ("model.network.num_basis_in", int),
                        ("model.network.num_basis_blend", int),
                        ("runtime.device", str)):
        value = cfg.get(key)
        if not isinstance(value, expect):
            raise ValueError(
                f"配置项 {key} 应为 {expect.__name__}，实际 "
                f"{type(value).__name__}（{value!r}）")

    if not isinstance(cfg.get("app.port"), int):
        raise ValueError(f"app.port 应为 int，实际 {cfg.get('app.port')!r}")
    if not (1 <= cfg.get("app.port") <= 65535):
        raise ValueError(f"app.port 越界：{cfg.get('app.port')}")
    if float(cfg.get("render.target_fps", 0)) < 0:
        raise ValueError(f"render.target_fps 不能为负：{cfg.get('render.target_fps')}")
    sti = float(cfg.get("render.status_interval_s", 0.25))
    if sti <= 0:
        raise ValueError(f"render.status_interval_s 必须 > 0，实际 {sti}")
    si = float(cfg.get("app.stats_log_interval_s", 10.0))
    if si <= 0:
        raise ValueError(f"app.stats_log_interval_s 必须 > 0，实际 {si}")
    pf = int(cfg.get("app.preload_frames", -1))
    if pf == 0 or pf < -1:
        raise ValueError(f"app.preload_frames 应为 -1（全部）或 ≥ 1，实际 {pf}")

    if cfg.get("runtime.device") not in ("cuda", "cpu"):
        raise ValueError(
            f"runtime.device 只能是 'cuda' 或 'cpu'，实际 {cfg.get('runtime.device')!r}")

    hidden = cfg.get("model.network.mlp_hidden")
    if hidden is not None and not isinstance(hidden, list):
        raise ValueError(
            f"model.network.mlp_hidden 应为列表，实际 {type(hidden).__name__}")


def repo_root() -> Path:
    """仓库根目录。"""
    return REPO_ROOT


# --------------------------------------------------------- 模型路径解析 --


def model_selected(cfg: "Config" | None = None) -> bool:
    """配置里是否已经指定了对象与版本。"""
    cfg = cfg or load_config()
    return bool(str(cfg.get("subject") or "").strip()
                and str(cfg.get("work_name") or "").strip())


def model_dir(cfg: "Config" | None = None) -> Path:
    """本项目自己的模型目录：`<models_dir>/<subject>/<work_name>/`。"""
    cfg = cfg or load_config()
    subject = str(cfg.get("subject") or "").strip()
    work = str(cfg.get("work_name") or "").strip()
    if not (subject and work):
        raise ValueError(
            "还没有选择模型。请在界面右侧「模型」里选对象与版本，"
            "或传 --subject/--work-name。")
    return Path(cfg.paths.models_dir) / subject / work


def resolve_model_ply(cfg: "Config" | None = None,
                      must_exist: bool = True) -> Path:
    """解析模型 `.ply` 路径。

    `paths.model_ply` 是**文件名**（默认 `model.ply`），相对模型目录
    `<models_dir>/<subject>/<work_name>/`；写绝对路径则原样使用
    （脚本的 `--ply` 就是这么传的）。

    Args:
        cfg: 配置；``None`` 时读仓库根的 configs/
        must_exist: 为真且文件不存在时报错

    Raises:
        FileNotFoundError: `must_exist` 为真且文件不存在
    """
    cfg = cfg or load_config()

    name = str(cfg.get("paths.model_ply") or "model.ply")
    path = Path(name)
    # 相对名 → 模型目录下；绝对路径 → 原样用（脚本的 --ply 会传绝对路径）
    if not path.is_absolute():
        path = model_dir(cfg) / path

    if must_exist and not path.exists():
        # 注意：未选模型时 `model_dir()` 本身会抛错，别让它盖住真正的错误
        try:
            where = f"\n模型目录：{model_dir(cfg)}"
        except ValueError:
            where = ""
        raise FileNotFoundError(
            f"找不到模型文件：{path}{where}\n"
            "请把模型放到 models/<对象>/<版本>/ 下，或在设置里改「模型文件名」。")
    return path


def resolve_model_config(cfg: "Config" | None = None) -> Path | None:
    """解析模型随附的 `config.yaml`。

    查找顺序：`paths.models_config` → `<models_dir>/<subject>/<work_name>/config.yaml`。
    找不到返回 ``None``。
    """
    cfg = cfg or load_config()
    name = str(cfg.get("paths.models_config") or "config.yaml")
    path = Path(name)
    if not path.is_absolute():
        path = model_dir(cfg) / path
    return path if path.exists() else None


def resolve_data_dir(cfg: Config) -> Path:
    """数据集目录：`<paths.data_root>/<subject>`。

    `subject` 既是模型的一级目录，也是数据集目录名。
    界面上选数据集走的是 `data/datasets.py::resolve_dataset_dir`（按目录名）。
    """
    subject = str(cfg.get("subject") or "").strip()
    if not subject:
        raise ValueError("还没有选择数据集（subject 为空）")
    data_root = cfg.get("paths.data_root")
    if data_root is None:
        raise ValueError("缺少 paths.data_root")
    return (Path(str(data_root)) / subject).resolve()
