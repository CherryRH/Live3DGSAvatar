

def test_warns_when_work_dir_has_ply_but_no_model_ply(tmp_path) -> None:
    """目录里有 `model_aug.ply` 却没有 `model.ply` 时，要明确告诉用户怎么修。

    真实故障：旧版本把增强模型存成 `model_aug.ply`，`list_models` 只认
    `model.ply`，于是这个 work 在前端下拉框里"凭空消失"，而日志一声不响。
    """
    import logging

    from live3dgsavatar.config import load_config
    from live3dgsavatar.data.models import list_models

    work = tmp_path / "crh_1" / "run_aug"
    work.mkdir(parents=True)
    (work / "model_aug.ply").write_bytes(b"x")

    cfg = load_config()
    cfg.set("paths.models_dir", str(tmp_path))
    import io

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)
    logger = logging.getLogger("live3dgsavatar.data.models")
    logger.addHandler(handler)
    try:
        infos = list_models(cfg)
    finally:
        logger.removeHandler(handler)

    assert infos == [], "没有 model.ply 的不该出现在列表里"
    text = stream.getvalue()
    assert "model_aug.ply" in text
    assert "mv" in text, "提示里要给出可执行的修复命令"
