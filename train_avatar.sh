#!/usr/bin/env bash
# 单阶段头像重建（可挂加密）。
#
# 流程：
#   1) 初始化：UV 绑定高斯（各向同性球/盘状由 configs/model.yaml 决定）
#      → 合并成 tex 512 的统一集合（形变基 K=20 一开始就可用）
#   2) 训练：每批 8 张 = 6 张真实帧 + 2 张修复图；L1 + SSIM + （可选）LPIPS；
#      可选挂载加密（残差驱动 split）
#
# 修复图来自 **base** 模型（先 bash train_base.sh --subject <对象> --work-name base）。
# 缺修复图时本脚本会用 base 自动生成；缺 surface_masks.npz 时自动缓存。
#
# 用法：
#   bash train_avatar.sh --subject bala                     # 默认不加密
#   bash train_avatar.sh --subject bala --densify           # 挂上加密
#   bash train_avatar.sh --subject bala --lpips             # 额外开 LPIPS 损失
#   bash train_avatar.sh --subject bala --no-repairs        # 等同 base 的对照
#   bash train_avatar.sh --subject bala --plan              # 只打印命令
#   bash train_avatar.sh --subject bala --eval              # 训练后自动评测
#
# 环境变量：AVATAR_* 覆盖 train.pipeline.avatar，DENS_* 覆盖加密子项；
# 完整默认值见 configs/train.yaml。
# 产物：models/<对象>/<名字>（默认 avatar）。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MAIN_ENV="${MAIN_ENV:-live3dgs}"
PY_RUN=(conda run -n "$MAIN_ENV" --no-capture-output python)

die() { echo "[error] $*" >&2; exit 2; }
info() { echo "[avatar] $*"; }
usage() { sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

SUBJECT=""
NAME="avatar"
DENSIFY=0
USE_LPIPS=0
NO_REPAIRS=0
PLAN=0
RUN_EVAL=0
FORCE=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject)     SUBJECT="${2:-}"; shift 2 ;;
    --work-name)   NAME="${2:-}"; shift 2 ;;
    --densify)     DENSIFY=1; shift ;;
    --lpips)       USE_LPIPS=1; shift ;;
    --no-repairs)  NO_REPAIRS=1; shift ;;
    --plan|--show-plan) PLAN=1; shift ;;
    --eval)        RUN_EVAL=1; shift ;;
    --force)       FORCE=1; shift ;;
    --)            shift; EXTRA=("$@"); break ;;
    -h|--help)     usage ;;
    *)             die "未知参数：$1" ;;
  esac
done

[[ -n "$SUBJECT" ]] || die "需要 --subject（例如 --subject bala）"
DATA="datasets/$SUBJECT"
[[ -d "$DATA" ]] || die "数据集不存在：$DATA"
MODELS="models/$SUBJECT"

cfg_shell="$("${PY_RUN[@]}" scripts/show_config.py --shell-avatar)" || \
  die "读取 configs/train.yaml 的 train.pipeline.avatar 失败"
eval "$cfg_shell"

STEPS="${AVATAR_STEPS:-${CFG_AVATAR_STEPS:-7000}}"
BATCH="${AVATAR_BATCH:-${CFG_AVATAR_BATCH:-8}}"
REPAIR="${AVATAR_REPAIR:-${CFG_AVATAR_REPAIR:-2}}"
BLEND="${AVATAR_BLEND_START:-${CFG_AVATAR_BLEND_START:-0}}"
SSIM="${AVATAR_SSIM_WEIGHT:-${CFG_AVATAR_SSIM_WEIGHT:-0.1}}"
XYZ_LR="${AVATAR_XYZ_LR:-${CFG_AVATAR_XYZ_LR:-0.0002}}"
# 默认不把 LPIPS 放进损失（保持与其他对照单一变量）；--lpips 才启用
LPIPS=0
(( USE_LPIPS )) && LPIPS="${AVATAR_LPIPS_WEIGHT:-${CFG_AVATAR_LPIPS_WEIGHT:-0.01}}"
LPIPS_FROM="${AVATAR_LPIPS_FROM:-${CFG_AVATAR_LPIPS_FROM:-3000}}"
TEX="${AVATAR_TEX_SIZE:-${CFG_AVATAR_TEX_SIZE:-512}}"
PATIENCE="${AVATAR_PATIENCE:-${CFG_AVATAR_PATIENCE:-4}}"
SCHEDULE="${AVATAR_REPAIR_SCHEDULE:-${CFG_AVATAR_REPAIR_SCHEDULE:-none}}"
FINAL_W="${AVATAR_REPAIR_FINAL_WEIGHT:-${CFG_AVATAR_REPAIR_FINAL_WEIGHT:-0.2}}"

EVERY="${DENS_EVERY:-${CFG_AVATAR_DENSIFY_EVERY:-300}}"
COUNT="${DENS_COUNT:-${CFG_AVATAR_DENSIFY_COUNT:-900}}"
MAX_GAUSS="${DENS_MAX:-${CFG_AVATAR_DENSIFY_MAX_GAUSSIANS:-200000}}"
PRUNE_OP="${DENS_PRUNE_OPACITY:-${CFG_AVATAR_DENSIFY_PRUNE_OPACITY:-0.01}}"
JITTER="${DENS_SPLIT_JITTER:-${CFG_AVATAR_DENSIFY_SPLIT_JITTER:-0.8}}"

FORCE_FLAG=()
(( FORCE )) && FORCE_FLAG=(--force)
run() {
  echo
  echo "\$ $*"
  (( PLAN )) || "$@"
}

echo "=============================================================="
echo "单阶段头像重建：$SUBJECT / $NAME"
echo "  训练 : $STEPS 步  batch $BATCH  repair $REPAIR  ssim $SSIM  lpips $LPIPS  退火 $SCHEDULE"
echo "  结构 : tex $TEX  xyz_lr $XYZ_LR（盘状初始化配套）  加密 $([[ "$DENSIFY" == 1 ]] && echo "开（every $EVERY count $COUNT）" || echo "关")"
echo "  产物 : $MODELS/$NAME"
(( PLAN )) && echo "  --plan : 只打印命令，不执行"
echo "=============================================================="

# ---- 依赖：修复图（缺失时用 base 自动生成）----
REPAIRS="$DATA/repair/repairs"
if (( NO_REPAIRS )); then
  info "不使用修复图（对照 base）：跳过生成与 surface_masks"
elif [[ ! -d "$REPAIRS" ]] || [[ -z "$(ls -A "$REPAIRS" 2>/dev/null)" ]]; then
  if [[ -f "$MODELS/base/model.ply" ]]; then
    info "缺少修复图，用 $MODELS/base 自动生成（p90 起点、只做 yaw 扇）"
    run "${PY_RUN[@]}" scripts/make_repairs.py --subject "$SUBJECT" \
      --work-name base --auto \
      --out "output/repair/$SUBJECT" --out-dir "$REPAIRS" \
      --repair-root "$DATA/repair"
  elif (( PLAN )); then
    info "缺少修复图，将用 base 自动生成（plan 模式只打印）"
  else
    die "缺少修复图且没有 $MODELS/base：先跑 bash train_base.sh --subject $SUBJECT --work-name base"
  fi
fi
(( NO_REPAIRS )) || [[ -d "$REPAIRS" ]] || die "修复图生成失败：$REPAIRS"

# ---- 依赖：surface_masks.npz（缺失时自动缓存）----
if (( ! NO_REPAIRS )) && [[ ! -f "$DATA/surface_masks.npz" ]]; then
  info "缺少 surface_masks.npz，自动生成（训练混合修复帧需要）"
  run "${PY_RUN[@]}" scripts/cache_surface_masks.py --data-dir "$DATA"
fi

# ---- 初始化 + 合并成统一集合 ----
INIT_DIR="tmp/${SUBJECT}_${NAME}_init"
UNIFIED="tmp/${SUBJECT}_${NAME}_unified.ply"
UNIFIED_CFG="tmp/${SUBJECT}_${NAME}_unified.yaml"
info "初始化：UV 绑定高斯（未训练）"
run "${PY_RUN[@]}" scripts/init_uv.py --subject "$SUBJECT" --out-dir "$INIT_DIR"
run "${PY_RUN[@]}" - "$INIT_DIR" "$UNIFIED_CFG" "$TEX" <<'PY'
import re
import sys
from pathlib import Path

init_dir, out_cfg, tex = sys.argv[1:]
src = Path(init_dir) / "config.yaml"
text = src.read_text(encoding="utf-8")
text, n = re.subn(r"tex_size:\s*\d+", f"tex_size: {tex}", text, count=1)
if n != 1:
    raise SystemExit(f"[error] 无法在 {src} 里替换 tex_size")
Path(out_cfg).parent.mkdir(parents=True, exist_ok=True)
Path(out_cfg).write_text(text, encoding="utf-8")
print(f"结构配置：{src} -> {out_cfg}（tex {tex}）")
PY
run "${PY_RUN[@]}" scripts/merge_layers.py \
  --hard "$INIT_DIR/model.ply" --out "$UNIFIED" --tex-size "$TEX"

# ---- 训练 ----
DENSIFY_ARGS=()
if (( DENSIFY )); then
  DENSIFY_ARGS=(--densify --densify-every "$EVERY" --densify-count "$COUNT"
                --densify-max "$MAX_GAUSS" --densify-prune-opacity "$PRUNE_OP"
                --densify-split-jitter "$JITTER")
fi
REPAIR_ARGS=()
(( NO_REPAIRS )) || REPAIR_ARGS=(--repair-dir "$REPAIRS" --repair-per-batch "$REPAIR")
run "${PY_RUN[@]}" scripts/train.py \
  --subject "$SUBJECT" --work-name "$NAME" --stage base \
  --init-ply "$UNIFIED" --model-config "$UNIFIED_CFG" \
  "${REPAIR_ARGS[@]}" \
  --batch-size "$BATCH" --iterations "$STEPS" \
  --blend-start "$BLEND" --hard-xyz-lr "$XYZ_LR" --patience "$PATIENCE" \
  --ssim-weight "$SSIM" --lpips-weight "$LPIPS" --lpips-from "$LPIPS_FROM" \
  --repair-schedule "$SCHEDULE" --repair-final-weight "$FINAL_W" \
  "${DENSIFY_ARGS[@]}" "${FORCE_FLAG[@]}" "${EXTRA[@]}"

info "完成：$MODELS/$NAME"
(( RUN_EVAL )) && bash eval.sh --subject "$SUBJECT" --work-name "$NAME"
info "评测：bash eval.sh --subject $SUBJECT --work-name $NAME"
