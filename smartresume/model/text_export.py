"""
Text-only export of multimodal Gemma "E" checkpoints (Gemma 4 E2B / E4B) for 6 GB GPUs.

The published checkpoint (google/gemma-4-E4B-it) holds:
  * the language model (~4.6B params incl. its token embedding),
  * vision + audio encoders (~0.47B params), never used to parse resume text,
  * per-layer embedding tables ("PLE", 262144 x 10752 = 2.8B params, 5.6 GB in 16-bit).

PLE tables are only looked up (one row per token), never multiplied, so they can live
in system RAM. transformers cannot do that by itself: a "cpu" entry in a device_map
means *offload* (weights parked in RAM, copied to the GPU for every forward pass), which
needs the full 5.6 GB on the GPU anyway and keeps a float32 copy in RAM while loading.

So the checkpoint is exported once, tensor by tensor (low RAM), into
<model>-text/:
    model-0000x-of-0000n.safetensors   language-model weights, keys renamed for
                                       Gemma4ForCausalLM ("model.language_model.X" -> "model.X")
    per_layer_embeddings.safetensors   the PLE table
    config.json                        the text config (model_type gemma4_text)
    tokenizer / chat template / generation config, copied
The model is then loaded with a 1-row placeholder PLE (see load_text_model in
llm_client) and the real table is attached as a CPU lookup.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

EXPORT_SUFFIX = "-text"
PLE_FILE = "per_layer_embeddings.safetensors"
PLE_KEY = "embed_tokens_per_layer.weight"
## Written last: a folder without it is an interrupted export and is redone
MARKER_FILE = "smartresume_text_export.json"
EXPORT_VERSION = 1

_SOURCE_PREFIX = "model.language_model."
_TARGET_PREFIX = "model."
_PLE_SOURCE_KEY = _SOURCE_PREFIX + PLE_KEY
_SHARD_BYTES = 2 * 1024 ** 3
## Model files that are rewritten, not copied
_SKIP_COPY = {"config.json", "model.safetensors", "model.safetensors.index.json", "README.md", ".gitattributes"}


def _read_config(model_path: str) -> Dict:
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def needs_text_export(model_path: str) -> bool:
    """True for a multimodal Gemma checkpoint whose text model has per-layer embeddings."""
    try:
        config = _read_config(model_path)
    except (OSError, json.JSONDecodeError):
        return False
    text_config = config.get("text_config") or {}
    return (
        str(config.get("model_type", "")).startswith("gemma")
        and bool(text_config.get("hidden_size_per_layer_input"))
    )


def has_cpu_ple(model_path: str) -> bool:
    """True when model_path is an export made by this module."""
    return (os.path.isfile(os.path.join(model_path, MARKER_FILE))
            and os.path.isfile(os.path.join(model_path, PLE_FILE)))


def export_dir_for(model_path: str) -> str:
    return os.path.normpath(model_path).rstrip("\\/") + EXPORT_SUFFIX


def _source_files(model_path: str) -> List[str]:
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index, encoding="utf-8") as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(model_path, name) for name in files]
    return [os.path.join(model_path, "model.safetensors")]


def ensure_text_export(model_path: str, export_path: Optional[str] = None) -> str:
    """Return the text-only export of model_path, creating it first if needed."""
    export_path = export_path or export_dir_for(model_path)
    if has_cpu_ple(export_path):
        return export_path
    logger.info("One-time text-only export: %s -> %s (a few minutes)", model_path, export_path)
    export_text_only(model_path, export_path)
    return export_path


def export_text_only(model_path: str, export_path: str) -> None:
    """Write the text-only export (see module docstring). Peak RAM is about one shard."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = _read_config(model_path)
    text_config = dict(config["text_config"])
    text_config["architectures"] = ["Gemma4ForCausalLM"]
    text_config.setdefault("tie_word_embeddings", config.get("tie_word_embeddings", True))
    if "transformers_version" in config:
        text_config["transformers_version"] = config["transformers_version"]

    os.makedirs(export_path, exist_ok=True)
    marker = os.path.join(export_path, MARKER_FILE)
    if os.path.exists(marker):
        os.remove(marker)

    shard: Dict = {}
    shard_bytes = 0
    shard_names: List[str] = []
    key_to_shard: Dict[str, int] = {}

    def flush() -> None:
        nonlocal shard, shard_bytes
        if not shard:
            return
        name = f"part-{len(shard_names):05d}.safetensors"
        save_file(shard, os.path.join(export_path, name), metadata={"format": "pt"})
        shard_names.append(name)
        shard, shard_bytes = {}, 0

    for source in _source_files(model_path):
        with safe_open(source, framework="pt") as reader:
            for key in reader.keys():
                if not key.startswith(_SOURCE_PREFIX):
                    continue  # vision / audio encoders and their projectors
                tensor = reader.get_tensor(key)
                if key == _PLE_SOURCE_KEY:
                    save_file({PLE_KEY: tensor}, os.path.join(export_path, PLE_FILE), metadata={"format": "pt"})
                    del tensor
                    continue
                target = _TARGET_PREFIX + key[len(_SOURCE_PREFIX):]
                shard[target] = tensor.contiguous()
                key_to_shard[target] = len(shard_names)
                shard_bytes += tensor.numel() * tensor.element_size()
                if shard_bytes >= _SHARD_BYTES:
                    flush()
    flush()

    if not os.path.isfile(os.path.join(export_path, PLE_FILE)):
        raise RuntimeError(f"{_PLE_SOURCE_KEY} not found in {model_path}")

    ## Final shard names follow the usual model-0000x-of-0000n pattern
    total = len(shard_names)
    final_names = [f"model-{i + 1:05d}-of-{total:05d}.safetensors" for i in range(total)]
    for old, new in zip(shard_names, final_names):
        os.replace(os.path.join(export_path, old), os.path.join(export_path, new))
    weight_map = {key: final_names[idx] for key, idx in key_to_shard.items()}
    with open(os.path.join(export_path, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
    with open(os.path.join(export_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(text_config, f, indent=2)

    for name in os.listdir(model_path):
        source = os.path.join(model_path, name)
        if name in _SKIP_COPY or name.endswith(".safetensors") or not os.path.isfile(source):
            continue
        shutil.copy2(source, os.path.join(export_path, name))

    with open(marker, "w", encoding="utf-8") as f:
        json.dump({"version": EXPORT_VERSION, "source": os.path.abspath(model_path),
                   "tensors": len(weight_map)}, f, indent=2)
    logger.info("Text-only export written: %d tensors in %d shard(s) + %s", len(weight_map), total, PLE_FILE)
