#!/usr/bin/env bash
# 一键启动 GUI 服务 —— 浏览器实时预览。
#
# 用法：
#   bash run_gui.sh                     # 默认读 configs/system.yaml 的 app.host / app.port
#   bash run_gui.sh --port 8080         # 覆盖端口
#   bash run_gui.sh --subject <对象> --work-name <版本>   # 启动时直接加载
#   bash run_gui.sh --help              # 透传给后端，见全部选项
#
# 前置：先跑过一次 `bash setup_env.sh`（建环境）。
#
# ⚠️ 与其他脚本不同，本脚本**必须**在装好 CUDA 扩展的 conda 环境里跑 ——
#    后端要真正调用光栅化器，不像 `render_test.py --dry-run` 那样能降级到 CPU。

set -eo pipefail
# 注意：**不要加 -u**。conda 的 deactivate 钩子会引用未定义变量，
# 一旦开启 -u 就会以 "unbound variable" 致命退出（见 setup_env.sh 顶部说明）。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="${ENV_NAME:-live3dgs}"

die() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[1;34m[info]\033[0m %s\n' "$*"; }

# ------------------------------------------------------------ 前置检查 ----

command -v conda >/dev/null 2>&1 || die "未找到 conda。请先安装 miniconda。"
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    die "conda 环境 '$ENV_NAME' 不存在。请先运行：bash setup_env.sh"
fi

conda activate "$ENV_NAME"

# 依赖自检：给出可执行的修复建议，而不是让用户看 ImportError 堆栈
python - <<'PY' || die "环境依赖不完整。请先运行：bash setup_env.sh"
import importlib, sys
missing = []
for mod in ("fastapi", "uvicorn", "torch", "plyfile", "yaml"):
    try:
        importlib.import_module(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f"  缺少依赖：{', '.join(missing)}", file=sys.stderr)
    sys.exit(1)
try:
    import torch
    import diff_gaussian_rasterization  # noqa: F401
except Exception as e:
    print(f"  CUDA 扩展不可用：{type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
if not torch.cuda.is_available():
    print("  CUDA 不可用（本服务无法降级到 CPU）", file=sys.stderr)
    sys.exit(1)
PY

# ---------------------------------------------------------------- 启动 ----

info "环境：$ENV_NAME"
info "工作目录：$SCRIPT_DIR"
info "启动 GUI 服务（Ctrl+C 停止）"
echo

exec python -m live3dgsavatar.app "$@"
