"""`python -m live3dgsavatar.app` 入口。

刻意**不直接** `from .server import main` —— 那样在缺依赖时会抛出
一长串 ModuleNotFoundError 堆栈，用户看不出该做什么。这里转成一句可执行的提示。
"""

import sys


def _run() -> int:
    try:
        from .server import main
    except ImportError as e:
        # ⚠️ 只在**缺少第三方包**时才提示安装依赖。
        #    本项目自身的导入错误（如某个名字被删了）也抛 ImportError，
        #    一律当成"依赖未安装"会把真正的 bug 掩盖成环境问题。
        name = getattr(e, "name", "") or ""
        third_party = ("fastapi", "uvicorn", "starlette", "torch",
                       "diff_gaussian_rasterization")
        if name.startswith(third_party) or not name:
            print(f"[error] GUI 后端依赖未安装：{e}", file=sys.stderr)
            print("        请先运行：bash setup_env.sh", file=sys.stderr)
        else:
            print(f"[error] GUI 后端导入失败（本项目内部错误，不是依赖问题）：{e}",
                  file=sys.stderr)
        return 1
    return main()


raise SystemExit(_run())
