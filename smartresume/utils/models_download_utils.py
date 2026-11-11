"""
Model download utilities for SmartResume
"""
import os
from typing import Optional

try:
    from huggingface_hub import snapshot_download as hf_snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


from .model_paths import ModelPath, ModelType, ModelSource


def auto_download_and_get_model_path(relative_path: str, model_type: ModelType = ModelType.LLM, save_path: Optional[str] = None) -> str:

    model_source = os.getenv('SMARTRESUME_MODEL_SOURCE', "huggingface")

    if model_source == 'local':
        from .config import config
        local_path = config.model_download.get('models_dir', {}).get(model_type.value, '')
        if not local_path:
            raise ValueError(f"Local path for model_type '{model_type.value}' is not configured.")
        return local_path

    repo_mapping = {
        ModelType.LLM: ModelPath.SMART_RESUME_ROOT_HF.value,
        ModelType.LAYOUT: ModelPath.YOLOV10_ROOT_HF.value
    }

    if model_type not in repo_mapping:
        raise ValueError(f"Unsupported model_type: {model_type}, must be 'llm' or 'layout'")

    repo = repo_mapping[model_type]

    if not HF_HUB_AVAILABLE:
        raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub")
    snapshot_download = hf_snapshot_download

    cache_dir = None

    if model_type == ModelType.LLM:
        relative_path = relative_path.strip('/')
        if save_path:
            # Create the save directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            cache_dir = snapshot_download(
                repo,
                allow_patterns=[relative_path, relative_path + "/*"],
                local_dir=save_path
            )
        else:
            cache_dir = snapshot_download(repo, allow_patterns=[relative_path, relative_path + "/*"])
    elif model_type == ModelType.LAYOUT:
        relative_path = relative_path.strip('/')
        if save_path:
            # Create the save directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            cache_dir = snapshot_download(
                repo,
                allow_patterns=[relative_path, relative_path + "/*"],
                local_dir=save_path
            )
        else:
            cache_dir = snapshot_download(repo, allow_patterns=[relative_path, relative_path + "/*"])

    if not cache_dir:
        raise FileNotFoundError(f"Failed to download model: {relative_path} from {repo}")

    return cache_dir


def llm_repo_id() -> str:
    """
    The Hugging Face repo of the configured LLM: config.direct_model_name when it is
    a repo id ("google/gemma-4-E4B-it"), otherwise the default model.
    """
    from .config import config

    name = str(getattr(config, 'direct_model_name', '') or '').strip().replace('\\', '/').rstrip('/')
    if name.count('/') == 1 and not os.path.exists(name) and not name.startswith(('.', 'models/')):
        return name
    return ModelPath.GEMMA_4_E4B_IT_HF.value


def download_llm(save_path: Optional[str] = None) -> str:
    """
    Download the configured LLM repo into <save_path>/<repo name>,
    e.g. models/gemma-4-E4B-it. Files already present are reused.
    """
    from .config import config

    if not HF_HUB_AVAILABLE:
        raise ImportError("huggingface_hub not available. Install with: pip install huggingface_hub")

    repo = llm_repo_id()
    target = os.path.join(save_path or 'models', repo.split('/')[-1])
    os.makedirs(target, exist_ok=True)
    path = hf_snapshot_download(repo, local_dir=target)

    ## Gemma "E" models: prepare the text-only export now rather than on the first run
    from smartresume.model import text_export
    if (getattr(config, 'direct_model', {}) or {}).get('offload_per_layer_embeddings', True) \
            and text_export.needs_text_export(target):
        text_export.ensure_text_export(target)
    return path


def get_model_path(model_type: ModelType) -> Optional[str]:
    from .config import config
    return config.model_download.get('models_dir', {}).get(model_type.value)


def download_model(model_type: ModelType, model_source: Optional[ModelSource] = None, save_path: Optional[str] = None) -> str:
    if model_source:
        os.environ['SMARTRESUME_MODEL_SOURCE'] = model_source.value

    if model_type == ModelType.LLM:
        return download_llm(save_path)
    elif model_type == ModelType.LAYOUT:
        return auto_download_and_get_model_path(ModelPath.YOLOV10_MODEL.value, ModelType.LAYOUT, save_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == '__main__':
    try:
        llm_path = download_llm()
        print(f"LLM model path: {llm_path}")

        layout_path = auto_download_and_get_model_path(ModelPath.YOLOV10_MODEL.value, ModelType.LAYOUT)
        print(f"Layout model path: {layout_path}")

    except Exception as e:
        print(f"Download failed: {e}")
