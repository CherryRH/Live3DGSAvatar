"""模型发现：列出 `<paths.models_dir>/<subject>/<work_name>/` 下可用的模型。

界面要能直接下拉选模型，而不是让人手打 subject / work_name ——
打错一个字就是一次失败的加载。

## 判定"可用"只查最必要的东西

`model.ply` 必须存在；`config.yaml` 有就记下来（缺了会在加载时报错，
但那是运行期的事）。**刻意不做深度校验**：把"哪些字段对得上"也查一遍，
既慢又容易把能跑的目录误判成不可用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import Config, repo_root

log = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """一个可加载的模型。"""

    subject: str
    work_name: str
    path: Path
    has_config: bool
    #: model.ply 大小（MiB），界面上用来区分同名不同版本的模型
    size_mib: float

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "work_name": self.work_name,
            "label": f"{self.subject} / {self.work_name}",
            "has_config": bool(self.has_config),
            "size_mib": round(float(self.size_mib), 1),
        }


def models_root(cfg: Config) -> Path:
    raw = cfg.get("paths.models_dir", "models")
    p = Path(str(raw)).expanduser()
    return p if p.is_absolute() else (repo_root() / p)


def list_models(cfg: Config) -> list[ModelInfo]:
    """列出所有含 `model.ply` 的模型，按 (subject, work_name) 排序。"""
    root = models_root(cfg)
    if not root.is_dir():
        return []

    out: list[ModelInfo] = []
    for subject_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not subject_dir.is_dir() or subject_dir.name.startswith("."):
            continue
        for work_dir in sorted(subject_dir.iterdir(), key=lambda p: p.name):
            if not work_dir.is_dir() or work_dir.name.startswith("."):
                continue
            ply = work_dir / "model.ply"
            if not ply.is_file():
                # 目录里有别的 .ply 却没有 model.ply：几乎一定是命名不对
                # （旧版本曾把增强训练的模型存成 model_aug.ply）。这类目录
                # 在界面上会"凭空消失"，所以要明确说出来怎么修。
                strays = sorted(p.name for p in work_dir.glob("*.ply"))
                if strays:
                    log.warning(
                        "%s 里没有 model.ply，但有 %s —— 界面不会显示它。"
                        "把主模型重命名为 model.ply 即可："
                        " mv %s %s",
                        work_dir, strays, work_dir / strays[0], ply)
                continue
            try:
                size = ply.stat().st_size / 1024 ** 2
            except OSError:
                size = 0.0
            out.append(ModelInfo(
                subject=subject_dir.name,
                work_name=work_dir.name,
                path=ply,
                has_config=(work_dir / "config.yaml").is_file(),
                size_mib=size))
    return out


def promote_model(*, source_ply: Path, structure_config: Path,
                  report: Path | None, target: Path,
                  force: bool = False) -> Path:
    """把一个训练好的模型装进 `models/<对象>/<版本>/`，供渲染直接使用。

    拷贝三样东西：

    - `model.ply`：高斯参数（PLY 头内嵌 `gaussian_config`，可自描述结构）
    - `config.yaml`：结构配置的可读副本（换机器/人工查看用）
    - `report.json`：训练/微调指标（可选，便于回溯来源）

    Raises:
        FileNotFoundError: 源模型不存在
        FileExistsError: 目标已有 `model.ply` 且没给 `force`
    """
    import shutil

    if not Path(source_ply).is_file():
        raise FileNotFoundError(f"找不到要装的模型：{source_ply}")
    target = Path(target)
    dest = target / "model.ply"
    if dest.exists() and not force:
        raise FileExistsError(f"{dest} 已存在；确认覆盖请加 --force")
    target.mkdir(parents=True, exist_ok=True)

    def _copy(src: Path, dst: Path) -> None:
        # 源与目标可能是同一个文件（例如续训时结构配置就在目标目录里）
        if Path(src).resolve() == Path(dst).resolve():
            return
        shutil.copyfile(src, dst)

    _copy(source_ply, dest)
    _copy(structure_config, target / "config.yaml")
    if report is not None and Path(report).is_file():
        _copy(report, target / "report.json")
    return target
