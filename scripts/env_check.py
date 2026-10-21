#!/usr/bin/env python3
"""环境自检 —— Live3DGSAvatar

逐项校验 docs/ENVIRONMENT.md 中的版本矩阵与关键依赖。
GPU 项非阻塞（无 GPU 权限时记为 warn），其余失败即 exit 1。

用法:
    python scripts/env_check.py            # 有 warn 也返回 0
    python scripts/env_check.py --strict   # 有 warn 即返回 1
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

OK, WARN, FAIL = "ok", "warn", "fail"
_MARK = {OK: "\033[1;32m  ok \033[0m", WARN: "\033[1;33m warn\033[0m", FAIL: "\033[1;31m fail\033[0m"}


@dataclass
class Report:
    rows: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.rows.append((name, status, detail))

    def dump(self) -> tuple[int, int]:
        width = max(len(n) for n, _, _ in self.rows)
        for name, status, detail in self.rows:
            print(f"[{_MARK[status]}] {name.ljust(width)}  {detail}")
        n_fail = sum(1 for _, s, _ in self.rows if s == FAIL)
        n_warn = sum(1 for _, s, _ in self.rows if s == WARN)
        print(f"\n{len(self.rows)} 项检查：{n_fail} 失败，{n_warn} 警告")
        return n_fail, n_warn


def _nvcc_version() -> str | None:
    """取 nvcc 版本：环境内 → CUDA_HOME → PATH。

    `CUDA_HOME` 排在 PATH 之前：用系统 CUDA 时（`LIVE3DGS_CUDA=system`），
    真正参与编译的是它指向的那份，而 PATH 里可能还有别的版本。
    """
    candidates = [
        os.path.join(sys.prefix, "bin", "nvcc"),
        os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "nvcc"),
        shutil.which("nvcc"),
    ]
    for exe in candidates:
        if exe and os.path.exists(exe):
            try:
                out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30).stdout
            except Exception:
                continue
            m = re.search(r"release (\d+\.\d+)", out)
            if m:
                return m.group(1)
    return None


def _candidate_include_dirs() -> list[str]:
    """编译期真正使用的 CUDA include 目录，按优先级排列。

    注意：torch 的 include_paths() 在检测不到 GPU 时**不含** CUDA 目录，
    因此必须显式枚举 CUDA_HOME 与 conda 环境的 targets/<arch>/include。
    """
    dirs: list[str] = []
    try:
        from torch.utils.cpp_extension import CUDA_HOME  # noqa: PLC0415
        if CUDA_HOME:
            dirs.append(os.path.join(CUDA_HOME, "include"))
    except Exception:
        pass
    # conda 版 cuda-toolkit 把头文件放在 targets/<triple>/include
    for triple in ("x86_64-linux", "x86_64-conda-linux-gnu", "x86_64-conda_cos6-linux-gnu"):
        dirs.append(os.path.join(sys.prefix, "targets", triple, "include"))
    dirs.append(os.path.join(sys.prefix, "include"))
    for cuda in ("/usr/local/cuda", "/usr/local/cuda-13.0"):
        dirs.append(os.path.join(cuda, "include"))
    seen, out = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _header_cuda_version() -> tuple[str | None, str | None]:
    """从 CUDA 头文件读版本（编译期真正使用的版本）。

    `CUDART_VERSION` 定义在 `cuda_runtime_api.h`（由 `cuda_runtime.h` include），
    因此以 `cuda_runtime_api.h` 为准。

    返回 (版本, 头文件路径)。
    """
    for inc in _candidate_include_dirs():
        for name in ("cuda_runtime_api.h", "cuda_runtime.h"):
            hdr = os.path.join(inc, name)
            if not os.path.exists(hdr):
                continue
            try:
                text = open(hdr, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            m = re.search(r"#define\s+CUDART_VERSION\s+(\d+)", text)
            if m:
                v = int(m.group(1))
                return f"{v // 1000}.{(v % 1000) // 10}", hdr
    return None, None


def _parse_pybind_defs(ext_cpp: Path) -> list[str]:
    """从 ext.cpp 的 `m.def("<name>", ...)` 解析 pybind 注册的算子名。

    以源码为准，避免手工维护的算子清单与实现漂移。
    """
    if not ext_cpp.exists():
        return []
    pattern = re.compile(r'm\.def\(\s*"([^"]+)"')
    return pattern.findall(ext_cpp.read_text(encoding="utf-8", errors="ignore"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="有警告即视为失败")
    ap.add_argument("--arch", default="8.6", help="期望的 TORCH_CUDA_ARCH_LIST")
    args = ap.parse_args()

    r = Report()
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    r.add("Python 解释器", OK if sys.version_info[:2] == (3, 10) else WARN, f"{pyver} @ {sys.prefix}")
    r.add("conda 环境", OK if sys.prefix != sys.base_prefix else WARN,
          os.environ.get("CONDA_DEFAULT_ENV", "<未激活>"))

    # ---- 第三方兼容补丁（必须在 import chumpy 之前）----
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _compat_shim  # noqa: PLC0415
    _patched = _compat_shim.apply()
    r.add("numpy 兼容补丁", OK,
          f"已补齐 {', '.join(_patched)}" if _patched else "无需应用（numpy 1.x）")

    # ---- torch / torchvision ----
    torch = None
    try:
        import torch  # noqa: PLC0415
        tv = importlib.import_module("torchvision")
        torch = torch
        r.add("torch", OK if torch.version.cuda else FAIL,
              f"{torch.__version__}  (cuda {torch.version.cuda})")
        r.add("torchvision", OK, tv.__version__)
    except Exception as e:  # pragma: no cover
        r.add("torch", FAIL, f"import 失败: {e}")
        r.dump()
        return 1

    # ---- CUDA 工具链一致性 ----
    nvcc_v = _nvcc_version()
    hdr_v, hdr_path = _header_cuda_version()
    r.add("nvcc", OK if nvcc_v else FAIL, nvcc_v or "未找到")
    r.add("CUDA 头文件版本", OK if hdr_v else FAIL,
          f"{hdr_v} @ {hdr_path}" if hdr_v else "未找到 cuda_runtime.h")
    if nvcc_v and hdr_v:
        same = nvcc_v == hdr_v
        r.add("nvcc ↔ torch 头文件一致", OK if same else FAIL,
              f"nvcc {nvcc_v} / header {hdr_v}")
    if torch.version.cuda and hdr_v:
        # CUDA 13.0 头文件应配 cu130 的 torch
        want = torch.version.cuda
        r.add("头文件 ↔ torch.version.cuda", OK if hdr_v == want else WARN,
              f"header {hdr_v} / torch {want}")

    # ---- 编译期架构 ----
    arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    r.add("TORCH_CUDA_ARCH_LIST", OK if arch else WARN, arch or "<未设置>")
    if arch and args.arch not in arch:
        r.add(f"含 sm_{args.arch.replace('.', '')}", WARN, f"当前为 '{arch}'")

    # ---- 第三方 Python 依赖 ----
    for mod, dist, required in [
        ("numpy", "numpy", True),
        ("PIL", "Pillow", True),
        ("yaml", "PyYAML", True),
        ("tqdm", "tqdm", True),
        ("roma", "roma", True),
        ("plyfile", "plyfile", True),
        ("trimesh", "trimesh", True),
        ("smplx", "smplx", False),
        ("scipy", "scipy", False),
        ("tensorboard", "tensorboard", False),
        ("lpips", "lpips", False),
        ("fused_ssim", "fused-ssim", False),
    ]:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            r.add(f"py: {dist}", OK, str(ver))
        except Exception as e:
            r.add(f"py: {dist}", FAIL if required else WARN, f"{type(e).__name__}: {e}"[:70])

    # chumpy 需在补丁之后验证（见上方「numpy 兼容补丁」）
    ok, detail = _compat_shim.verify_chumpy()
    r.add("py: chumpy", OK if ok else FAIL, detail[:70] if ok else detail[:90])

    # ---- nvdiffrast ----
    try:
        import nvdiffrast  # noqa: F401, PLC0415
        from nvdiffrast import torch as _ndf  # noqa: F401, PLC0415
        r.add("nvdiffrast", OK, getattr(importlib.import_module("nvdiffrast"), "__version__", "?"))
    except Exception as e:
        r.add("nvdiffrast", FAIL, f"import 失败: {e}"[:70])

    # ---- 本地光栅化扩展 ----
    # 直接依据 ext.cpp 的 pybind 注册表校验算子，避免手工清单随源码漂移
    try:
        from diff_gaussian_rasterization import _C  # noqa: PLC0415

        ext_cpp = Path(__file__).resolve().parent.parent / "submodules" / \
            "diff-gaussian-rasterization" / "ext.cpp"
        registered = _parse_pybind_defs(ext_cpp)
        missing = [s for s in registered if not hasattr(_C, s)]
        detail = f"{len(registered) - len(missing)}/{len(registered)} 算子（依据 ext.cpp）"
        if missing:
            detail += f"；缺 {missing}"
        r.add("ext: diff_gaussian_rasterization",
              OK if registered and not missing else FAIL, detail)
    except Exception as e:
        r.add("ext: diff_gaussian_rasterization", FAIL, f"import 失败: {e}"[:70])

    # ---- GPU（非阻塞）----
    try:
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            r.add("GPU", OK, f"{p.name}, {p.total_memory / 2**30:.1f} GiB, sm_{p.major}{p.minor}")
            cap = f"{p.major}.{p.minor}"
            r.add("GPU ↔ 编译架构", OK if cap in arch else WARN,
                  f"device sm_{p.major}{p.minor} / built for '{arch or '?'}'")
        else:
            r.add("GPU", WARN,
                  "torch.cuda.is_available() == False：确认已激活正确环境，且当前会话可访问 GPU 设备"
                  "（容器/远程会话常见）")
    except Exception as e:
        r.add("GPU", WARN, f"探测失败: {e}"[:70])

    n_fail, n_warn = r.dump()
    if n_fail:
        return 1
    if args.strict and n_warn:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
