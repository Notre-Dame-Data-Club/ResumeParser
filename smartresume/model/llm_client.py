"""
LLM client module
"""
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from smartresume.utils.config import config
from smartresume.utils.prompts import get_prompts
from smartresume.utils.schema import EXTRACT_TYPE_SECTIONS, LIST_SECTIONS, SECTION_ENTRY_KEYS
from smartresume.data.sections import scoped_inputs
from smartresume.model import text_export
from smartresume.model.ollama_backend import OllamaBackend, schema_from_template

import random
import json_repair

# Direct model imports
try:
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    StoppingCriteria = object

logger = logging.getLogger(__name__)

DEFAULT_MAX_NEW_TOKENS = 1024
## Output caps below max_new_tokens for extraction types whose answer is always short.
## A row that loops or rambles then stops early instead of holding its whole batch.
DEFAULT_MAX_NEW_TOKENS_BY_TYPE = {"basic_info": 256}
## Batches are filled in this order, so rows with answers of similar length share a
## call (a call runs until its longest answer ends; shorter rows just wait)
_OUTPUT_LENGTH_ORDER = {"basic_info": 0, "education": 1, "projects": 2, "work_experience": 3}
## quantization "auto": GPUs with at least this much VRAM load the model unquantized
_AUTO_FULL_PRECISION_BYTES = 14 * 1024 ** 3
## generate() budget on ZeroGPU, where VRAM cannot be measured at load time
## (~10 GB of activations / KV cache at DEFAULT_BYTES_PER_BATCH_TOKEN)
ZERO_GPU_TOKEN_BUDGET = 60000
## Longest prompt (template + resume text) sent to the model; longer resume text is cut
MAX_PROMPT_TOKENS = 8192
## GPU memory one batched token needs during generate() (KV cache, activations,
## per-layer inputs). Measured on Gemma 4 E4B NF4: ~125 KB; rounded up for margin.
DEFAULT_BYTES_PER_BATCH_TOKEN = 160 * 1024
## VRAM kept free under the cap for the CUDA context / allocator fragmentation
_GPU_SAFETY_BYTES = 300 * 1024 ** 2



def on_zero_gpu() -> bool:
    """
    True inside a Hugging Face ZeroGPU Space.

    There, CUDA is emulated while the app starts and a real GPU only exists inside
    @spaces.GPU calls, in a forked worker. Loading therefore has to place weights
    with the string device "cuda" (never an index) and must not ask the GPU for its
    memory.
    """
    return os.environ.get("SPACES_ZERO_GPU", "").strip().lower() in ("1", "true")


_DTYPES = {"float16": "float16", "fp16": "float16", "half": "float16",
           "bfloat16": "bfloat16", "bf16": "bfloat16", "float32": "float32", "fp32": "float32"}


# -- Loop detection -- #
def _find_loop(tokens: List[int], min_span: int = 30, max_period: int = 120) -> bool:
    """True when the end of tokens is one block repeated 3+ times, covering >= min_span tokens."""
    n = len(tokens)
    for period in range(2, min(max_period, n // 3) + 1):
        span = period * max(3, -(-min_span // period))
        if span > n:
            continue
        tail = tokens[-span:]
        if tail[period:] == tail[:-period]:
            return True
    return False


class _LoopGuard(StoppingCriteria):
    """
    Stop a row once its output starts repeating itself.

    Small models sometimes fall into a loop ("Diving", "Musical Theater", ... over and
    over) and would otherwise burn the whole token budget and never close their JSON.
    Digit tokens are compared as equal, so a loop that only counts upward
    ("2040", "2041", "2042", ...) is caught too. Rows are checked every few steps on
    the last `window` tokens only, so it is cheap.
    """

    def __init__(self, prompt_width: int, stop_ids: Set[int], digit_ids: Set[int],
                 check_every: int = 8, window: int = 400,
                 caps: Optional[List[int]] = None) -> None:
        self.prompt_width = prompt_width
        self.stop_ids = stop_ids
        self.digit_ids = digit_ids
        self.check_every = check_every
        self.window = window
        self.looped: List[bool] = []
        ## Per-row output caps (rows of one call can have different limits)
        self.caps = caps
        self.capped: List[bool] = []

    def __call__(self, input_ids, scores, **kwargs):
        if not self.looped:
            self.looped = [False] * input_ids.shape[0]
            self.capped = [False] * input_ids.shape[0]
        generated = input_ids.shape[1] - self.prompt_width
        if self.caps and generated in self.caps:
            last = input_ids[:, -1].tolist()
            for row, cap in enumerate(self.caps):
                ## A row that already ended is padded with EOS/PAD; it was not cut off
                if generated == cap and not self.looped[row] and last[row] not in self.stop_ids:
                    self.capped[row] = True
        if generated >= 30 and generated % self.check_every == 0:
            tails = input_ids[:, -min(generated, self.window):].tolist()
            for row, tail in enumerate(tails):
                ## A finished row is padded with EOS/PAD; that run is not a loop
                if self.looped[row] or tail[-1] in self.stop_ids:
                    continue
                masked = [-1 if token in self.digit_ids else token for token in tail]
                if _find_loop(masked):
                    self.looped[row] = True
        done = [looped or capped for looped, capped in zip(self.looped, self.capped)]
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


class _CpuEmbedding(torch.nn.Module if TRANSFORMERS_AVAILABLE else object):
    """
    Embedding lookup kept in system RAM (Gemma per-layer embeddings, 5.6 GB in E4B).

    Ids are moved to the CPU, rows are gathered there and only the gathered rows
    (a few MB) go to the GPU. Same result as Gemma4TextScaledWordEmbedding: rows
    times embed_scale.
    """

    def __init__(self, weight, scale: float, out_dtype) -> None:
        super().__init__()
        ## A plain attribute, not a buffer: model.to("cuda") must leave the table in RAM
        self.weight = weight
        self.scale = scale
        self.out_dtype = out_dtype
        self.num_embeddings, self.embedding_dim = weight.shape

    def forward(self, input_ids):
        rows = torch.nn.functional.embedding(input_ids.to(self.weight.device), self.weight)
        ## Scale on the GPU, in the compute dtype (the table itself stays bf16 in RAM).
        ## The ids say where the model runs; no device index is stored (ZeroGPU hands
        ## each worker its own).
        return rows.to(input_ids.device).to(self.out_dtype) * self.scale


# -- Response parsing -- #
_REPEATED_TEXT = re.compile(r"(.{10,}?)\1{2,}", re.DOTALL)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
## One "key": value pair, for salvaging entries out of broken JSON
_JSON_STRING = r'"(?:[^"\\]|\\.)*"'
_PAIR = re.compile(
    r'"(?P<key>[A-Za-z_][\w]*)"\s*:\s*(?P<value>'
    + _JSON_STRING
    + r'|\[[^\[\]]*\]|-?\d+(?:\.\d+)?|true|false|null)'
)


def _dedupe(value: Any) -> Any:
    """Drop repeats left over from a loop: repeated list items and repeated text runs."""
    if isinstance(value, str):
        return _REPEATED_TEXT.sub(r"\1", value)
    if isinstance(value, dict):
        return {key: _dedupe(item) for key, item in value.items()}
    if isinstance(value, list):
        seen, kept = set(), []
        for item in value:
            item = _dedupe(item)
            marker = json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
            if marker not in seen:
                seen.add(marker)
                kept.append(item)
        return kept
    return value


def _keep_entry_keys(entry: Dict[str, Any], allowed: Tuple[str, ...]) -> Dict[str, Any]:
    """Only the template's keys survive ("Do not add keys" is not always obeyed)."""
    if not allowed:
        return dict(entry)
    return {key: entry[key] for key in allowed if key in entry}


def _normalize_section(parsed: Any, section: str) -> Dict[str, Any]:
    """
    Reduce a parsed answer to {section: value} with only template keys per entry.

    Keys outside the section (left over when broken JSON is repaired, e.g. a stray
    "name" at the top level) are dropped instead of being merged into the result.
    An answer that forgot the {"section": ...} wrapper is wrapped.
    """
    if not isinstance(parsed, dict):
        parsed = {section: parsed} if isinstance(parsed, list) else {}
    allowed = SECTION_ENTRY_KEYS.get(section, ())

    if section in parsed:
        body = parsed[section]
    elif allowed and any(key in parsed for key in allowed):
        body = parsed  # bare entry without the wrapper
    else:
        return {}

    if section in LIST_SECTIONS:
        if isinstance(body, dict):
            body = [body]
        if not isinstance(body, list):
            return {section: []}
        return {section: [_keep_entry_keys(e, allowed) for e in body if isinstance(e, dict)]}

    return {section: _keep_entry_keys(body, allowed) if isinstance(body, dict) else {}}


## Keys that name an entry; an entry without any of them is a fragment, not an entry
_IDENTITY_KEYS: Dict[str, Tuple[str, ...]] = {
    "workExperience": ("companyName", "position"),
    "education": ("school", "major"),
    "projects": ("name",),
}


def _has_identity(entry: Dict[str, Any], section: str) -> bool:
    return any(str(entry.get(key) or "").strip() for key in _IDENTITY_KEYS.get(section, ()))


def _scan_entries(text: str, section: str) -> List[Dict[str, Any]]:
    """
    Salvage list entries from JSON too broken to repair structurally.

    A small model sometimes forgets the "}, {" between entries and writes every
    entry's keys into one object. A JSON parser then keeps only the last value of
    each key (5 projects become 1). Instead, every "key": value pair is read in
    order and a new entry starts when the entry's anchor key (its first template
    key, e.g. "name") repeats. Any other repeated key keeps its first value: the
    model sometimes appends further "technologies" lists copied from other entries.
    """
    allowed = SECTION_ENTRY_KEYS.get(section, ())
    if not allowed:
        return []
    anchor = allowed[0]
    entries: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    for match in _PAIR.finditer(text):
        key = match.group("key")
        if key not in allowed:
            continue
        raw_value = match.group("value")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = json_repair.loads(raw_value)
        starts_entry = key == anchor and anchor in current
        ## An entry whose anchor was skipped: a repeated key is the only boundary signal
        starts_entry = starts_entry or (key in current and anchor not in current)
        if starts_entry:
            entries.append(current)
            current = {}
        current.setdefault(key, value)
    if current:
        entries.append(current)
    return [e for e in entries if _has_identity(e, section)]


class _DuplicateKeys(Exception):
    """An object in the answer repeats a key (entries were merged into one object)."""


def _reject_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _DuplicateKeys(keys)
    return dict(pairs)


def parse_llm_response(response: str, prompt_key: str, truncated: bool = False) -> Dict[str, Any]:
    """
    Pull the JSON answer for one extraction type out of a model response.

    Returns {section: value} for the prompt's own section only ({} if nothing usable).
    A truncated response (stopped by the loop guard or the token cap) never closes
    its JSON; its open brackets are closed and every entry written before the cut is
    kept, with the looped repeats removed. Quotes inside the text are left alone (an
    apostrophe in "O'Brien" is valid JSON and must not be turned into a quote).
    """
    section = EXTRACT_TYPE_SECTIONS.get(prompt_key, prompt_key)
    text = _THINK_BLOCK.sub("", response or "").strip()

    start = text.find("{")
    if start == -1:
        return {}
    body = text[start:]

    parsed: Any = None
    strict = False
    duplicate_keys = False
    end = body.rfind("}")
    if not truncated and end != -1:
        try:
            parsed = json.loads(body[:end + 1], object_pairs_hook=_reject_duplicate_keys)
            strict = True
        except _DuplicateKeys:
            ## A plain parse would silently keep the last value of each repeated key
            duplicate_keys = True
            try:
                parsed = json.loads(body[:end + 1])
            except json.JSONDecodeError:
                parsed = None
        except json.JSONDecodeError:
            parsed = None
    if parsed is None:
        try:
            parsed = json_repair.loads(body)
        except Exception:
            parsed = {}

    result = _normalize_section(parsed, section)

    if not strict and section in LIST_SECTIONS:
        parsed_entries = [e for e in result.get(section, []) if _has_identity(e, section)]
        scanned = _scan_entries(body, section)
        ## The scan keeps first values of repeated keys, so it wins ties on duplicate keys
        if len(scanned) > len(parsed_entries) or (duplicate_keys and len(scanned) == len(parsed_entries)):
            result = {section: scanned}

    return result if strict else _dedupe(result)


def _merge_result(combined: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Merge one prompt's answer: lists from several sections are joined, other keys replaced."""
    for key, value in result.items():
        if isinstance(value, list) and isinstance(combined.get(key), list):
            combined[key].extend(value)
        else:
            combined[key] = list(value) if isinstance(value, list) else value


def _count_entries(result: Dict[str, Any], prompt_key: str) -> int:
    """Entries in this prompt's own section (a dict section counts as 1 when non-empty)."""
    value = result.get(EXTRACT_TYPE_SECTIONS.get(prompt_key, prompt_key))
    if isinstance(value, list):
        return len(value)
    return 1 if value else 0


class LLMClient:
    """LLM client responsible for interacting with language models"""

    def __init__(self) -> None:
        self.prompts = get_prompts()

        # Initialize direct model support first, so remote clients are only
        # created if they're actually needed as a fallback.
        self.use_direct_models = getattr(config, 'use_direct_models', False)
        self.direct_model = None
        self.direct_tokenizer = None
        self.stop_ids: Set[int] = set()
        self.eos_ids: List[int] = []
        self.pad_id: int = 0
        self.digit_ids: Set[int] = set()
        options = getattr(config, 'direct_model', {}) or {}
        self.max_new_tokens = int(options.get('max_new_tokens', DEFAULT_MAX_NEW_TOKENS))
        self.max_new_tokens_by_type: Dict[str, int] = {
            **DEFAULT_MAX_NEW_TOKENS_BY_TYPE, **(options.get('max_new_tokens_by_type') or {})}
        ## How the model ended up loaded ("nf4" / "none"), for logs and callers
        self.quantization = ""
        ## Max rows x (prompt + new tokens) per generate() call; 0 = no limit (CPU)
        self.token_budget = 0
        ## Which runtime serves the local model: "ollama" (llama.cpp server, fast on
        ## small GPUs) or "transformers" (in-process; the only one that runs on ZeroGPU)
        self.backend = str(getattr(config, 'llm_backend', 'transformers') or 'transformers').lower()
        self.ollama: Optional[OllamaBackend] = None
        self._schemas: Dict[str, Any] = {}
        if self.use_direct_models and self.backend == "ollama":
            self._init_ollama()
            if self.ollama is None:
                logger.warning("Ollama is not usable; loading the model with transformers instead "
                               "(several times slower on a small GPU).")
        if self.ollama is None:
            self.backend = "transformers"
            self._init_direct_model()

        self.default_client = None
        self.channel_clients: Dict[str, OpenAI] = {}
        if not (self.use_direct_models):
            self.default_client = OpenAI(
                base_url=config.model.api_url,
                api_key=config.model.api_key
            )
            self._init_channel_clients()

    def _init_channel_clients(self) -> None:
        """Initialize multi-channel clients"""
        if hasattr(config, 'channels') and isinstance(config.channels, dict):
            channel_names = list(config.channels.keys())

            for channel_name in channel_names:
                try:
                    channel_config = config.channels[channel_name]
                    if hasattr(channel_config, 'api_url') and hasattr(channel_config, 'api_key'):
                        self.channel_clients[channel_name] = OpenAI(
                            base_url=channel_config.api_url,
                            api_key=channel_config.api_key
                        )

                    else:
                        pass
                except Exception:
                    pass

    @property
    def ready(self) -> bool:
        """True when the local model can take prompts (either backend)."""
        if self.backend == "ollama":
            return self.ollama is not None
        return self.direct_model is not None and self.direct_tokenizer is not None

    @property
    def model_label(self) -> str:
        """Model name and runtime, for logs."""
        if self.backend == "ollama" and self.ollama is not None:
            return f"{self.ollama.model} via Ollama"
        return f"{getattr(config, 'direct_model_name', '')} ({self.quantization or 'not loaded'})"

    def _init_ollama(self) -> None:
        """Connect to (or start) the Ollama server and load the model into it."""
        from smartresume.utils import prompts as prompt_texts
        try:
            backend = OllamaBackend(getattr(config, 'ollama', {}) or {})
            backend.start()
        except Exception as e:
            logger.error("Failed to start the Ollama backend: %s", e)
            return
        self.ollama = backend
        self.quantization = "gguf"
        ## The answer shape of each extraction type, taken from its prompt TEMPLATE
        for prompt_key in self.prompts:
            template = getattr(prompt_texts, f"{prompt_key.upper()}_TEMPLATE", None)
            if template:
                self._schemas[prompt_key] = schema_from_template(template)

    # -- Direct model loading -- #
    @staticmethod
    def _resolve_local_model_path(direct_model_name: str) -> Optional[str]:
        """The local folder of the configured model, or None if it is not downloaded."""
        models_dir = getattr(config, 'model_download', {}).get('models_dir', {}).get('llm', 'models') or 'models'
        if os.path.isdir(direct_model_name):
            return direct_model_name
        for path in (
            os.path.join(models_dir, direct_model_name),
            os.path.join(models_dir, os.path.basename(direct_model_name)),
            os.path.join('models', direct_model_name),
            os.path.join('models', os.path.basename(direct_model_name)),
        ):
            if os.path.isdir(path):
                return path
        return None

    def _init_direct_model(self) -> None:
        """Load the local model (NF4-quantized on GPU when configured)."""
        if not self.use_direct_models or not TRANSFORMERS_AVAILABLE:
            return

        try:
            direct_model_name = getattr(config, 'direct_model_name', None)
            if not direct_model_name:
                logger.warning("use_direct_models is True but direct_model_name is not configured")
                return

            local_model_path = self._resolve_local_model_path(direct_model_name)
            if not local_model_path:
                from ..utils.models_download_utils import download_llm, llm_repo_id
                logger.info("Local model not found, downloading: %s", llm_repo_id())
                try:
                    models_dir = getattr(config, 'model_download', {}).get('models_dir', {}).get('llm', 'models')
                    download_llm(models_dir or 'models')
                    local_model_path = self._resolve_local_model_path(direct_model_name)
                except Exception as download_error:
                    logger.error("Failed to download model: %s", download_error)
                if not local_model_path:
                    ## Last resort: let transformers resolve a Hub repo id
                    local_model_path = llm_repo_id()

            options = getattr(config, 'direct_model', {}) or {}
            dtype = getattr(torch, _DTYPES.get(str(options.get('compute_dtype', 'float16')).lower(), 'float16'))
            cpu_ple = bool(options.get('offload_per_layer_embeddings', True))
            on_gpu = torch.cuda.is_available()
            zero_gpu = on_gpu and on_zero_gpu()

            ## Gemma "E" models: load a text-only export whose per-layer embeddings stay in RAM
            if on_gpu and cpu_ple and text_export.needs_text_export(local_model_path):
                local_model_path = text_export.ensure_text_export(local_model_path)
            use_cpu_ple = on_gpu and text_export.has_cpu_ple(local_model_path)

            self.direct_tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)

            ## "auto" tries full precision first and falls back to NF4 when it does not fit
            candidates = self._quantization_candidates(options, on_gpu, zero_gpu)
            for attempt, quantization in enumerate(candidates):
                logger.info("Loading direct model from: %s (%s, %s, %s%s)", local_model_path,
                            "zerogpu" if zero_gpu else "cuda" if on_gpu else "cpu",
                            quantization if on_gpu else "no quantization", str(dtype).replace("torch.", ""),
                            ", per-layer embeddings in RAM" if use_cpu_ple else "")
                try:
                    self._load_direct_model(local_model_path, dtype, quantization, use_cpu_ple, on_gpu, zero_gpu)
                    self.quantization = quantization if on_gpu else "none"
                    break
                except Exception as load_error:
                    self.direct_model = None
                    if attempt == len(candidates) - 1:
                        raise
                    logger.warning("Loading with quantization=%s failed (%s); trying %s.",
                                   quantization, load_error, candidates[attempt + 1])
                    if on_gpu and not zero_gpu:
                        torch.cuda.empty_cache()

            self._init_token_ids()

            if zero_gpu:
                ## No real GPU exists yet: nothing to measure and nothing to cap
                self.token_budget = int(options.get('token_budget') or ZERO_GPU_TOKEN_BUDGET)
                logger.info("Direct model placed for ZeroGPU; generate() budget %d batch tokens", self.token_budget)
            elif on_gpu:
                self._set_gpu_budget(options)
                logger.info("Direct model loaded: %.2f GB on GPU", torch.cuda.memory_allocated() / 1024 ** 3)
            else:
                logger.info("Direct model loaded on CPU")

        except Exception as e:
            logger.error("Failed to load direct model: %s", e, exc_info=True)
            self.direct_model = None
            self.direct_tokenizer = None

    @staticmethod
    def _quantization_candidates(options: Dict[str, Any], on_gpu: bool, zero_gpu: bool) -> List[str]:
        """
        Quantization modes to try, in order.

        nf4   4-bit weights (bitsandbytes): ~3x less VRAM, but every forward pass
              first unpacks all weights, so each generated token is slower.
        none  weights in compute_dtype: more VRAM, fastest tokens, no quantization loss.
        auto  none where it fits (ZeroGPU, or a GPU with >= 14 GB), else nf4; falls
              back to nf4 when the full-precision load fails.
        """
        quantization = str(options.get('quantization', 'nf4')).lower()
        if not on_gpu:
            return ["none"]
        if quantization != "auto":
            return [quantization if quantization == "nf4" else "none"]
        if zero_gpu:
            return ["none"]
        total = torch.cuda.get_device_properties(0).total_memory
        return ["none", "nf4"] if total >= _AUTO_FULL_PRECISION_BYTES else ["nf4"]

    def _load_direct_model(self, model_path: str, dtype, quantization: str,
                           use_cpu_ple: bool, on_gpu: bool, zero_gpu: bool) -> None:
        """Load the weights for one quantization mode and put them on the device."""
        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        ple_rows = None
        if use_cpu_ple:
            ## A 1-row placeholder table: the real one is attached on the CPU after loading
            ple_rows = model_config.vocab_size_per_layer_input
            model_config.vocab_size_per_layer_input = 1

        load_kwargs: Dict[str, Any] = {
            "config": model_config,
            "trust_remote_code": True,
            "dtype": dtype if on_gpu else torch.float32,
        }
        ## ZeroGPU: the device is the string "cuda". bitsandbytes' loader places the
        ## weights itself; a full-precision model is loaded in RAM and moved with
        ## .to("cuda"), the path ZeroGPU intercepts and packs at startup.
        move_after_load = zero_gpu and quantization != "nf4"
        if on_gpu and not move_after_load:
            load_kwargs["device_map"] = "cuda" if zero_gpu else {"": 0}
        if on_gpu and quantization == "nf4":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=["lm_head"],
            )
        if move_after_load:
            load_kwargs["low_cpu_mem_usage"] = True

        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        if move_after_load:
            model = model.to("cuda")
        elif not on_gpu:
            model = model.to("cpu")
        model.eval()
        self.direct_model = model

        if use_cpu_ple:
            ## After the move, so the table is never sent to the GPU
            self._attach_cpu_ple(model_path, dtype)
            model.config.vocab_size_per_layer_input = ple_rows

    def _set_gpu_budget(self, options: Dict[str, Any]) -> None:
        """
        Cap PyTorch's share of VRAM and size generate() calls to fit under the cap.

        Without a cap, Windows lets CUDA spill into shared system memory once VRAM is
        full instead of raising out-of-memory, and generation slows to a crawl. With
        it, an oversized call fails fast and is retried one prompt at a time.
        """
        total = torch.cuda.get_device_properties(0).total_memory
        gb = options.get('gpu_memory_gb')
        fraction = (float(gb) * 1024 ** 3 / total) if gb else float(options.get('gpu_memory_fraction', 0.88))
        fraction = max(0.1, min(fraction, 1.0))
        torch.cuda.set_per_process_memory_fraction(fraction, 0)

        headroom = fraction * total - torch.cuda.memory_allocated() - _GPU_SAFETY_BYTES
        bytes_per_token = int(options.get('bytes_per_batch_token', DEFAULT_BYTES_PER_BATCH_TOKEN))
        self.token_budget = int(options.get('token_budget') or max(0, int(headroom // bytes_per_token)))
        logger.info("GPU cap %.2f GB; generate() budget %d batch tokens (rows x (prompt + %d new))",
                    fraction * total / 1024 ** 3, self.token_budget, self.max_new_tokens)

    def _attach_cpu_ple(self, model_path: str, dtype) -> None:
        """Swap the placeholder per-layer embedding for the real table, held in RAM."""
        from safetensors import safe_open

        with safe_open(os.path.join(model_path, text_export.PLE_FILE), framework="pt") as reader:
            weight = reader.get_tensor(text_export.PLE_KEY)

        for name, module in list(self.direct_model.named_modules()):
            if not name.endswith("embed_tokens_per_layer"):
                continue
            parent_name, _, attr = name.rpartition(".")
            parent = self.direct_model.get_submodule(parent_name)
            scale = float(getattr(module, "scalar_embed_scale", 1.0))
            setattr(parent, attr, _CpuEmbedding(weight, scale, dtype))
            logger.info("Per-layer embeddings in RAM: %s (%.1f GB)", tuple(weight.shape),
                        weight.numel() * weight.element_size() / 1024 ** 3)
            return
        raise RuntimeError("Model has no embed_tokens_per_layer module to attach the table to")

    def _init_token_ids(self) -> None:
        """EOS ids (Gemma has several), the pad id, and which tokens are digits."""
        tokenizer = self.direct_tokenizer
        eos: List[int] = []
        gen_eos = getattr(getattr(self.direct_model, "generation_config", None), "eos_token_id", None)
        for value in (gen_eos, tokenizer.eos_token_id):
            if isinstance(value, int):
                eos.append(value)
            elif isinstance(value, (list, tuple)):
                eos.extend(v for v in value if isinstance(v, int))
        self.eos_ids = list(dict.fromkeys(eos))
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_ids[0]
        self.stop_ids = set(self.eos_ids) | {self.pad_id}
        self.digit_ids = {
            token_id for token, token_id in tokenizer.get_vocab().items()
            if token.lstrip("▁Ġ ").isdigit()
        }

    def _get_client(self, extract_type: str, use_backup_channel: bool = False) -> OpenAI:
        """Get the client for a given extraction type"""
        if use_backup_channel and hasattr(config, 'extract_channels_backup') and config.extract_channels_backup:
            channel_name = getattr(config.extract_channels_backup, extract_type, "")
        elif hasattr(config, 'extract_channels_main') and config.extract_channels_main:
            channel_name = getattr(config.extract_channels_main, extract_type, "")
        elif hasattr(config, 'extract_channels') and config.extract_channels:
            channel_name = getattr(config.extract_channels, extract_type, "")
        else:
            channel_name = ""

        if channel_name and channel_name in self.channel_clients:
            return self.channel_clients[channel_name]
        else:
            return self.default_client

    def _get_channel_config(self, extract_type: str, use_backup_channel: bool = False) -> Any:
        """Get the channel configuration for a given extraction type"""
        if use_backup_channel and hasattr(config, 'extract_channels_backup') and config.extract_channels_backup:
            channel_name = getattr(config.extract_channels_backup, extract_type, "")
        elif hasattr(config, 'extract_channels_main') and config.extract_channels_main:
            channel_name = getattr(config.extract_channels_main, extract_type, "")
        elif hasattr(config, 'extract_channels') and config.extract_channels:
            channel_name = getattr(config.extract_channels, extract_type, "")
        else:
            channel_name = ""

        if channel_name and hasattr(config, 'channels') and channel_name in config.channels:
            return config.channels[channel_name]
        else:
            return config.model

    def _extract_info_remote(self, text_content: str, extract_types: List[str],
                             resume_id: str, use_backup_channel: bool = False) -> Dict[str, Any]:
        """
        Extract structured information using remote LLM API.

        Args:
            text_content: The input text content.
            extract_types: List of extraction types to run.
            resume_id: Resume identifier.
            use_backup_channel: Whether to use backup channel mapping.

        Returns:
            A dictionary with extracted fields.
        """
        def call_llm(prompt_key: str) -> Dict[str, Any]:
            """Call the LLM for a single extraction type"""
            client = self._get_client(prompt_key, use_backup_channel)
            channel_config = self._get_channel_config(prompt_key, use_backup_channel)

            messages = [
                {
                    "role": "system",
                    "content": self.prompts[prompt_key]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text_content
                        }
                    ]
                }
            ]

            params = {
                'model': channel_config.name,
                'messages': messages,
                'max_tokens': channel_config.max_tokens,
                'temperature': channel_config.temperature,
                'top_p': channel_config.top_p,
                'seed': channel_config.seed,
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                    "repetition_penalty": 1.01
                },
            }

            if config.processing.use_force_json:
                params['response_format'] = {"type": "json_object"}

            max_retries = 2
            for attempt in range(max_retries + 1):
                if attempt > 0:
                    params['temperature'] = 1.0
                    params['seed'] = random.randint(0, 1000000)

                try:
                    completion = client.chat.completions.create(**params)
                    content = completion.choices[0].message.content
                    truncated = completion.choices[0].finish_reason == "length"
                    result = parse_llm_response(content, prompt_key, truncated=truncated)
                    if not result:
                        raise ValueError("No valid JSON content found")
                    return result

                except Exception as e:
                    if attempt < max_retries:
                        continue
                    os.makedirs("contents", exist_ok=True)
                    error_info = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "prompt_key": prompt_key,
                        "channel_config": {
                            "name": channel_config.name,
                            "api_url": channel_config.api_url,
                            "max_tokens": channel_config.max_tokens
                        },
                        "params": params
                    }
                    with open(
                        f"contents/{resume_id}_{prompt_key}_error.json",
                        "w",
                        encoding='utf-8',
                    ) as f:
                        json.dump(error_info, f, ensure_ascii=False, indent=2)
                    return {}

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(call_llm, extract_types))

        combined_result: Dict[str, Any] = {}
        for result in results:
            _merge_result(combined_result, result)

        return combined_result

    def extract_info_direct(self, text_content: str, extract_types: List[str],
                            resume_id: str, use_backup_channel: bool = False) -> Dict[str, Any]:
        """
        Extract structured information using directly loaded model.

        Args:
            text_content: The input text content
            extract_types: List of extraction types to run
            resume_id: Resume identifier
            use_backup_channel: Whether to use backup channel mapping (not used in direct mode)

        Returns:
            A dictionary with extracted fields
        """
        self._require_direct_model()
        # One prompt per generate() call (to avoid memory issues)
        results, _ = self._extract_direct_jobs([(resume_id, text_content)], extract_types, batch_size=1)
        return results[0]

    def extract_info_batch(self, jobs: List[Tuple[str, str]], extract_types: List[str],
                           batch_size: int = 4, return_calls: bool = False):
        """
        Extract several resumes, running up to batch_size prompts per generate() call.

        Each extraction type reads only its own resume section (see data/sections.py);
        work experience gets one prompt per section. A batch that fails (e.g. out of
        memory under --gpu-memory) is retried one prompt at a time.

        Args:
            jobs: (resume_id, indexed text_content) pairs
            extract_types: List of extraction types to run for every resume
            batch_size: Prompts per generate() call; 1 means one prompt at a time
            return_calls: Also return, per job, a log of every prompt (section, stop
                reason, token count, raw response)

        Returns:
            One combined result dict per job, in the same order as jobs
            (and the per-job call logs when return_calls is True)
        """
        if not self.use_direct_models:
            results = [self.extract_info(text, extract_types, resume_id) for resume_id, text in jobs]
            calls: List[List[Dict[str, Any]]] = [[] for _ in jobs]
        else:
            self._require_direct_model()
            results, calls = self._extract_direct_jobs(jobs, extract_types, max(1, batch_size))
        return (results, calls) if return_calls else results

    def _extract_direct_jobs(self, jobs: List[Tuple[str, str]], extract_types: List[str],
                             batch_size: int) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """Run every (resume, extraction type, section) prompt; merge the answers per resume."""
        prompts: List[Dict[str, Any]] = []
        for job_idx, (resume_id, text) in enumerate(jobs):
            for prompt_key in extract_types:
                for scoped in scoped_inputs(text, prompt_key):
                    entry: Dict[str, Any] = {
                        "job": job_idx,
                        "extract_type": prompt_key,
                        "section": scoped["section"],
                        "line_range": scoped["line_range"],
                    }
                    prompts.append(entry)
                    entry["cap"] = self._output_cap(prompt_key)
                    if self.backend == "ollama":
                        entry["text"] = scoped["text"]
                        continue
                    try:
                        entry["ids"] = self._encode_direct_prompt(prompt_key, scoped["text"])
                    except Exception as e:
                        entry["result"] = self._record_direct_error(resume_id, prompt_key, e)
                        entry.update(stop_reason="error", error=str(e))

        if self.backend == "ollama":
            self._run_ollama_prompts(prompts, jobs)

        # Prompts with answers of similar length, then of similar prompt length, share a
        # batch: less waiting on one long answer and less padding
        pending = sorted(
            (p for p in prompts if "ids" in p),
            key=lambda p: (_OUTPUT_LENGTH_ORDER.get(p["extract_type"], len(_OUTPUT_LENGTH_ORDER)), len(p["ids"])),
        )

        for chunk in self._plan_batches(pending, batch_size):
            try:
                outputs = self._generate_direct([p["ids"] for p in chunk], [p["cap"] for p in chunk])
            except Exception as e:
                if len(chunk) > 1:
                    logger.warning("Batch of %d prompts failed (%s); retrying one at a time.", len(chunk), e)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                outputs = [None] * len(chunk)

            for entry, output in zip(chunk, outputs):
                resume_id = jobs[entry["job"]][0]
                prompt_key = entry["extract_type"]
                try:
                    if output is None:  # batch failed: retry this prompt alone
                        output = self._generate_direct([entry["ids"]], [entry["cap"]])[0]
                    raw, stop_reason, new_tokens = output
                    entry.update(stop_reason=stop_reason, new_tokens=new_tokens, raw_response=raw)
                    entry["result"] = parse_llm_response(raw, prompt_key, truncated=stop_reason != "eos")
                except Exception as e:
                    entry["result"] = self._record_direct_error(resume_id, prompt_key, e)
                    entry.update(stop_reason="error", error=str(e))

        # Merge in extract_types order, sections top to bottom, like the sequential path
        type_order = {key: i for i, key in enumerate(extract_types)}
        prompts.sort(key=lambda p: (p["job"], type_order[p["extract_type"]], (p["line_range"] or [0])[0]))

        results: List[Dict[str, Any]] = [{} for _ in jobs]
        calls: List[List[Dict[str, Any]]] = [[] for _ in jobs]
        for entry in prompts:
            result = entry.pop("result", {})
            _merge_result(results[entry["job"]], result)
            entry.pop("ids", None)
            entry.pop("text", None)
            entry.pop("cap", None)
            entry["entries"] = _count_entries(result, entry["extract_type"])
            calls[entry.pop("job")].append(entry)
        return results, calls

    def _run_ollama_prompts(self, prompts: List[Dict[str, Any]], jobs: List[Tuple[str, str]]) -> None:
        """
        Send every prompt to the Ollama server, `parallel` at a time, and store the
        answers on the prompt entries (same fields as the transformers path).

        Longest answers go first: the call ends when its longest answer does, so
        that one must not start behind a queue of short ones. Prompts of one
        extraction type follow each other, so a slot that just served that type
        still holds its system prompt and only the resume text is read again.
        """
        started = time.perf_counter()
        order = sorted(
            range(len(prompts)),
            key=lambda i: (-_OUTPUT_LENGTH_ORDER.get(prompts[i]["extract_type"], 0), -len(prompts[i]["text"])),
        )

        def looping(pieces: List[str]) -> bool:
            ## Digits are compared as equal, as in _LoopGuard
            return _find_loop([-1 if piece.strip().isdigit() else piece for piece in pieces[-400:]])

        def call(entry: Dict[str, Any]) -> Tuple[str, str, int]:
            prompt_key = entry["extract_type"]
            system_prompt = self.prompts[prompt_key]
            text = self.ollama.fit_text(system_prompt, entry["text"], entry["cap"])
            return self.ollama.chat(system_prompt, text, entry["cap"],
                                    schema=self._schemas.get(prompt_key), should_stop=looping)

        sent_before, cached_before = self.ollama.prompt_tokens, self.ollama.cached_prompt_tokens
        outputs = self.ollama.run([lambda entry=prompts[i]: call(entry) for i in order])
        tokens = 0
        for i, output in zip(order, outputs):
            entry = prompts[i]
            prompt_key = entry["extract_type"]
            try:
                if isinstance(output, Exception):
                    raise output
                raw, stop_reason, new_tokens = output
                tokens += new_tokens
                entry.update(stop_reason=stop_reason, new_tokens=new_tokens, raw_response=raw)
                entry["result"] = parse_llm_response(raw, prompt_key, truncated=stop_reason != "eos")
            except Exception as e:
                entry["result"] = self._record_direct_error(jobs[entry["job"]][0], prompt_key, e)
                entry.update(stop_reason="error", error=str(e))
        elapsed = time.perf_counter() - started
        logger.info("ollama: %d prompt(s), %d new tokens in %.1fs (%.0f tokens/s overall); "
                    "%d prompt tokens, %d of them cached",
                    len(prompts), tokens, elapsed, tokens / max(elapsed, 1e-6),
                    self.ollama.prompt_tokens - sent_before, self.ollama.cached_prompt_tokens - cached_before)

    def _plan_batches(self, pending: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
        """
        Split length-sorted prompts into generate() calls that fit in GPU memory.

        A call holds every row until its longest answer ends, so its peak memory grows
        with rows x (longest prompt + max_new_tokens). Rows are added while that stays
        within self.token_budget (measured at load time) and below batch_size.
        """
        batches: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        width = new_tokens = 0  # widest prompt / largest output cap of the open batch
        for entry in pending:
            rows = len(current) + 1
            entry_width = max(width, len(entry["ids"]))
            entry_new = max(new_tokens, entry["cap"])
            cost = rows * (entry_width + entry_new)
            if current and (rows > batch_size or (self.token_budget and cost > self.token_budget)):
                batches.append(current)
                current = []
                entry_width, entry_new = len(entry["ids"]), entry["cap"]
            current.append(entry)
            width, new_tokens = entry_width, entry_new
        if current:
            batches.append(current)
        return batches

    def _output_cap(self, prompt_key: str) -> int:
        """Most tokens one answer of this extraction type may take."""
        return max(1, min(self.max_new_tokens, int(self.max_new_tokens_by_type.get(prompt_key, self.max_new_tokens))))

    # -- Direct model helpers (shared by the sequential and batched paths) -- #
    def _require_direct_model(self) -> None:
        if not self.ready:
            raise RuntimeError(
                "Local model not loaded and use_direct_models=true — no remote "
                "fallback is attempted. Check llm_backend, the model name and the model files."
            )

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        tokenizer = self.direct_tokenizer
        if getattr(tokenizer, 'chat_template', None):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            ## enable_thinking=False: no reasoning block, the token budget goes to the JSON
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

    def _encode_direct_prompt(self, prompt_key: str, text_content: str) -> List[int]:
        """
        Format one extraction prompt and tokenize it.

        The chat template already writes the BOS token, so the tokenizer must not add
        a second one. Over-long resume text is shortened, never the template around it
        (cutting the end would drop the assistant turn the model answers in).
        """
        system_prompt = self.prompts[prompt_key]
        ids = self._tokenize(self._format_prompt(system_prompt, text_content))
        if len(ids) > MAX_PROMPT_TOKENS:
            keep = max(200, int(len(text_content) * MAX_PROMPT_TOKENS / len(ids) * 0.95))
            logger.warning("%s prompt is %d tokens; resume text cut to %d chars.", prompt_key, len(ids), keep)
            ids = self._tokenize(self._format_prompt(system_prompt, text_content[:keep]))
        return ids

    def _tokenize(self, prompt: str) -> List[int]:
        return self.direct_tokenizer(prompt, add_special_tokens=False)["input_ids"]

    def _generate_direct(self, id_lists: List[List[int]],
                         caps: Optional[List[int]] = None) -> List[Tuple[str, str, int]]:
        """
        Generate a response for each tokenized prompt in one generate() call (greedy).

        caps holds each row's own output limit (default: max_new_tokens for all); the
        call runs until every row has ended or reached its limit.

        Returns (text, stop_reason, new_tokens) per prompt; stop_reason is "eos" (finished
        normally), "loop" (stopped because it repeated itself) or "max_tokens" (cut off).
        """
        tokenizer = self.direct_tokenizer
        pad_id = self.pad_id

        # Left-pad so every prompt ends at the same position and generation starts together
        width = max(len(ids) for ids in id_lists)
        input_ids = torch.full((len(id_lists), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(id_lists), width), dtype=torch.long)
        for row, ids in enumerate(id_lists):
            input_ids[row, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, width - len(ids):] = 1

        caps = list(caps) if caps else [self.max_new_tokens] * len(id_lists)
        max_new_tokens = max(caps)
        device = self.direct_model.get_input_embeddings().weight.device
        guard = _LoopGuard(prompt_width=width, stop_ids=self.stop_ids, digit_ids=self.digit_ids, caps=caps)
        started = time.perf_counter()
        with torch.no_grad():
            outputs = self.direct_model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                max_new_tokens=max_new_tokens,
                ## Greedy: the same resume gives the same answer on every run
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=pad_id,
                eos_token_id=self.eos_ids,
                stopping_criteria=StoppingCriteriaList([guard]),
            )

        steps = outputs.shape[1] - width
        elapsed = time.perf_counter() - started
        ## The numbers to watch when tuning: a call costs steps x ms/step, whatever the row count
        logger.info("generate(): %d row(s) x %d prompt tokens, %d steps in %.1fs (%.0f ms/step)",
                    len(id_lists), width, steps, elapsed, 1000 * elapsed / max(steps, 1))

        results = []
        for row, seq in enumerate(outputs):
            new = seq[width:].tolist()
            ## Rows that stop early are padded, so the loop / cap flags are checked first
            if guard.looped and guard.looped[row]:
                stop_reason = "loop"
            elif guard.capped and guard.capped[row]:
                stop_reason = "max_tokens"
                new = new[:caps[row]]
            else:
                stop_reason = "eos"
            end = next((i for i, token in enumerate(new) if token in self.stop_ids), None)
            if stop_reason == "eos" and end is None and len(new) >= max_new_tokens:
                stop_reason = "max_tokens"
            new_tokens = end if end is not None else len(new)
            results.append((tokenizer.decode(new[:new_tokens], skip_special_tokens=True), stop_reason, new_tokens))
        del outputs
        ## Calls differ in shape; returning cached blocks keeps the pool from fragmenting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    @staticmethod
    def _parse_direct_response(response: str, prompt_key: str, truncated: bool = False) -> Dict[str, Any]:
        """Kept for callers of the old name; see parse_llm_response."""
        return parse_llm_response(response, prompt_key, truncated)

    @staticmethod
    def _record_direct_error(resume_id: str, prompt_key: str, error: Exception) -> Dict[str, Any]:
        """Save error info for debugging and return an empty result."""
        logger.error("Error in direct model call for %s: %s", prompt_key, error)
        os.makedirs("contents", exist_ok=True)
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "prompt_key": prompt_key,
            "model_name": getattr(config, 'direct_model_name', 'unknown')
        }
        with open(
            f"contents/{resume_id}_{prompt_key}_direct_error.json",
            "w",
            encoding='utf-8',
        ) as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2)
        return {}

    def extract_info(self, text_content: str, extract_types: List[str],
                     resume_id: str, use_backup_channel: bool = False) -> Dict[str, Any]:
        """
        Extract structured information using LLM (direct or remote).

        Args:
            text_content: The input text content.
            extract_types: List of extraction types to run.
            resume_id: Resume identifier.
            use_backup_channel: Whether to use backup channel mapping.

        Returns:
            A dictionary with extracted fields.
        """
        if self.use_direct_models:
            return self.extract_info_direct(
                text_content=text_content,
                extract_types=extract_types,
                resume_id=resume_id,
                use_backup_channel=use_backup_channel
            )

        return self._extract_info_remote(
            text_content=text_content,
            extract_types=extract_types,
            resume_id=resume_id,
            use_backup_channel=use_backup_channel
        )
