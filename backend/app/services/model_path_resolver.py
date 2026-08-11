from pathlib import Path

from backend.app.core import config


class ModelPathResolutionError(RuntimeError):
    pass


TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "sentencepiece.bpe.model",
    "vocab.txt",
    "vocab.json",
}


def check_model_directory(path: str | Path) -> tuple[bool, list[str]]:
    model_dir = Path(path)
    missing: list[str] = []
    if not model_dir.exists() or not model_dir.is_dir():
        return False, ["模型目录不存在"]

    if not (model_dir / "config.json").is_file():
        missing.append("config.json")

    if not any((model_dir / name).is_file() for name in TOKENIZER_FILES):
        missing.append("tokenizer 相关文件")

    has_weight_file = any(model_dir.glob("*.safetensors")) or (model_dir / "pytorch_model.bin").is_file()
    has_weight_file = has_weight_file or any(model_dir.glob("*.bin"))
    if not has_weight_file:
        missing.append("模型权重文件（*.safetensors 或 pytorch_model.bin）")

    return not missing, missing


def resolve_embedding_model_path() -> str:
    return _resolve_model_path(
        display_name="BGE-Small-ZH",
        env_var_name="EMBEDDING_MODEL_PATH",
        explicit_path=config.EMBEDDING_MODEL_PATH,
        default_dir=config.DEFAULT_EMBEDDING_MODEL_DIR,
        hub_cache_dir=config.HF_HUB_CACHE,
        hub_repo_cache_name="models--BAAI--bge-small-zh-v1.5",
        online_model_name=config.EMBEDDING_MODEL_NAME,
        offline_mode=config.RESUME_OFFLINE_MODE,
    )


def resolve_embedding_model_local_path() -> str:
    return _resolve_model_path(
        display_name="BGE-Small-ZH",
        env_var_name="EMBEDDING_MODEL_PATH",
        explicit_path=config.EMBEDDING_MODEL_PATH,
        default_dir=config.DEFAULT_EMBEDDING_MODEL_DIR,
        hub_cache_dir=config.HF_HUB_CACHE,
        hub_repo_cache_name="models--BAAI--bge-small-zh-v1.5",
        online_model_name=config.EMBEDDING_MODEL_NAME,
        offline_mode=True,
    )


def resolve_reranker_model_path() -> str:
    return _resolve_model_path(
        display_name="BGE reranker",
        env_var_name="RERANKER_MODEL_PATH",
        explicit_path=config.RERANKER_MODEL_PATH,
        default_dir=config.DEFAULT_RERANKER_MODEL_DIR,
        hub_cache_dir=config.HF_HUB_CACHE,
        hub_repo_cache_name="models--BAAI--bge-reranker-base",
        online_model_name=config.RERANKER_MODEL_NAME,
        offline_mode=config.RESUME_OFFLINE_MODE,
    )


def resolve_reranker_model_local_path() -> str:
    return _resolve_model_path(
        display_name="BGE reranker",
        env_var_name="RERANKER_MODEL_PATH",
        explicit_path=config.RERANKER_MODEL_PATH,
        default_dir=config.DEFAULT_RERANKER_MODEL_DIR,
        hub_cache_dir=config.HF_HUB_CACHE,
        hub_repo_cache_name="models--BAAI--bge-reranker-base",
        online_model_name=config.RERANKER_MODEL_NAME,
        offline_mode=True,
    )


def resolve_reranker_onnx_model_local_path() -> str:
    """Resolve the single-file ONNX reranker artifact used by the CPU runtime."""

    model_path = config.RERANKER_ONNX_MODEL_PATH
    model_dir = model_path.parent
    missing: list[str] = []
    if not model_path.is_file():
        missing.append("model.onnx")
    if not any((model_dir / name).is_file() for name in TOKENIZER_FILES):
        missing.append("tokenizer related files")
    if missing:
        detail = ", ".join(missing)
        raise ModelPathResolutionError(
            "ONNX reranker artifact is incomplete. "
            f"Expected {detail} under {model_dir}. "
            "Run scripts/export_reranker_onnx.py before setting MODEL_BACKEND=onnx."
        )
    return str(model_path)


def _resolve_model_path(
    *,
    display_name: str,
    env_var_name: str,
    explicit_path: str | None,
    default_dir: Path,
    hub_cache_dir: Path,
    hub_repo_cache_name: str,
    online_model_name: str,
    offline_mode: bool,
) -> str:
    expected_paths = [default_dir]

    if explicit_path:
        explicit_dir = Path(explicit_path)
        if _is_valid_model_dir(explicit_dir):
            return str(explicit_dir)
        if offline_mode:
            raise ModelPathResolutionError(
                _missing_model_message(
                    display_name=display_name,
                    env_var_name=env_var_name,
                    default_dir=default_dir,
                    checked_paths=[explicit_dir, *expected_paths],
                    explicit_path=explicit_dir,
                )
            )

    if _is_valid_model_dir(default_dir):
        return str(default_dir)

    hub_snapshot = _find_hub_snapshot(hub_cache_dir / hub_repo_cache_name)
    if hub_snapshot is not None:
        return str(hub_snapshot)

    if not offline_mode:
        return online_model_name

    raise ModelPathResolutionError(
        _missing_model_message(
            display_name=display_name,
            env_var_name=env_var_name,
            default_dir=default_dir,
            checked_paths=expected_paths + [hub_cache_dir / hub_repo_cache_name / "snapshots"],
        )
    )


def _is_valid_model_dir(path: Path) -> bool:
    ready, _missing = check_model_directory(path)
    return ready


def _find_hub_snapshot(repo_cache_dir: Path) -> Path | None:
    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for snapshot in snapshots:
        if _is_valid_model_dir(snapshot):
            return snapshot
    return None


def _missing_model_message(
    *,
    display_name: str,
    env_var_name: str,
    default_dir: Path,
    checked_paths: list[Path],
    explicit_path: Path | None = None,
) -> str:
    lines = [
        f"离线模式已开启，但未找到 {display_name} 模型目录。",
        f"请将模型放置到：{_host_relative_path(default_dir)}",
        f"容器内路径应为：{_container_path(default_dir)}",
        f"或通过环境变量 {env_var_name} 指定本地模型目录。",
    ]
    if explicit_path is not None:
        lines.append(f"当前 {env_var_name} 指向的目录不可用：{explicit_path}")
    lines.append("已检查路径：")
    lines.extend(f"- {path}" for path in checked_paths)
    return "\n".join(lines)


def _host_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def _container_path(path: Path) -> str:
    try:
        relative_path = path.relative_to(config.DATA_DIR).as_posix()
    except ValueError:
        return str(path)
    return f"/app/data/{relative_path}"
