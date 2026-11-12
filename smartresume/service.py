"""
Programmatic entry point: parse resumes from Python, e.g. inside a Gradio app or
a Hugging Face ZeroGPU Space (see app.py for a complete Space).

    from smartresume.service import ResumeParser

    parser = ResumeParser()                  # loads the model once, at import time
    results = parser.parse(["resume.pdf"])   # [{"filename", "status", "data", ...}]

The work is split in two so a ZeroGPU Space only holds a GPU for the part that
needs one:

    documents = [parser.read(path, allow_ocr=False) for path in paths]   # CPU
    results = parser.extract(documents)                                  # GPU

read() does the PDF text extraction and layout detection. A file that needs OCR
(a scan, a page with pictures) is only marked when allow_ocr=False; extract()
then reads it, where the GPU is. Documents and results are plain dicts, lists
and strings, so they survive the pickling between a Space and its GPU worker.

Settings come from the arguments, then these environment variables, then
configs/config.yaml:

    RESUME_MODEL           Hugging Face repo id or local folder of the LLM
    RESUME_QUANTIZATION    none | nf4 | auto
    RESUME_DTYPE           bfloat16 | float16
    RESUME_MODELS_DIR      where models are downloaded (default: models)
    RESUME_BATCH_SIZE      prompts per generate() call (default: 16)
    RESUME_MAX_NEW_TOKENS  output token cap per prompt

On ZeroGPU the defaults are quantization "none" with bfloat16: the card has the
memory for it, tokens come out faster than with 4-bit weights, and nothing is
lost to quantization. Set RESUME_QUANTIZATION=nf4 to trade speed for ~6 GB less VRAM.
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from smartresume.utils.config import config

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/gemma-4-E4B-it"
DEFAULT_BATCH_SIZE = 16

## stop_reason of an LLM call -> why the row is only "partial"
_CUT_OFF = {
    "loop": "repeated itself and was stopped",
    "max_tokens": "hit the token limit",
}


def _on_zero_gpu() -> bool:
    return os.environ.get("SPACES_ZERO_GPU", "").strip().lower() in ("1", "true")


def configure(model_name: Optional[str] = None,
              quantization: Optional[str] = None,
              compute_dtype: Optional[str] = None,
              models_dir: Optional[str] = None,
              max_new_tokens: Optional[int] = None,
              layout_detection: Optional[bool] = None,
              page_limit: Optional[int] = None) -> None:
    """
    Point the shared config at a local (in-process) model. Call before the first
    ResumeParser is built; values not given keep their environment / config.yaml value.
    """
    config.use_direct_models = True
    config.direct_model_name = (
        model_name or os.environ.get("RESUME_MODEL")
        or getattr(config, "direct_model_name", None) or DEFAULT_MODEL
    )

    options = config.direct_model
    zero_gpu = _on_zero_gpu()
    quantization = quantization or os.environ.get("RESUME_QUANTIZATION") or ("none" if zero_gpu else None)
    compute_dtype = compute_dtype or os.environ.get("RESUME_DTYPE") or ("bfloat16" if zero_gpu else None)
    max_new_tokens = max_new_tokens or os.environ.get("RESUME_MAX_NEW_TOKENS")
    if quantization:
        options["quantization"] = str(quantization).lower()
    if compute_dtype:
        options["compute_dtype"] = str(compute_dtype).lower()
    if max_new_tokens:
        options["max_new_tokens"] = int(max_new_tokens)

    models_dir = models_dir or os.environ.get("RESUME_MODELS_DIR")
    directories = config.model_download.setdefault("models_dir", {})
    for kind in ("llm", "layout"):
        directories[kind] = models_dir or directories.get(kind) or "models"

    if layout_detection is not None:
        config.layout_detection.enabled = bool(layout_detection)
    if page_limit:
        config.processing.page_limit = int(page_limit)


class ResumeParser:
    """The resume pipeline as two calls: read() on the CPU, extract() on the GPU."""

    def __init__(self, batch_size: Optional[int] = None, **settings: Any) -> None:
        """
        Load the layout model and the LLM. settings are passed to configure().

        On ZeroGPU build this at module level (not inside the @spaces.GPU function):
        that is where ZeroGPU expects weights to be placed on "cuda".
        """
        configure(**settings)
        self.batch_size = max(1, int(batch_size or os.environ.get("RESUME_BATCH_SIZE") or DEFAULT_BATCH_SIZE))

        from smartresume.backend.resume_analyzer import ResumeAnalyzer
        from smartresume.utils.prompts import EXTRACT_TYPES

        self.extract_types = list(EXTRACT_TYPES)
        started = time.perf_counter()
        self.analyzer = ResumeAnalyzer()
        llm = self.analyzer.llm_client
        if llm is None or llm.direct_model is None:
            raise RuntimeError(
                f"The LLM '{config.direct_model_name}' failed to load (see the log above). "
                "Check the model name, the Hugging Face token for gated models, and free disk space."
            )
        logger.info("ResumeParser ready in %.1fs (model %s, quantization %s)",
                    time.perf_counter() - started, config.direct_model_name, llm.quantization)

    # -- Stage 1: file -> text (CPU; OCR only when allowed) -- #
    def read(self, path: str, allow_ocr: bool = True, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Read one file into a document for extract().

        allow_ocr=False never starts OCR: a file that needs it comes back with
        document["needs_ocr"] True and is read by extract() instead.
        """
        started = time.perf_counter()
        resume_id = f"resume_{uuid.uuid4().hex[:12]}"
        trace, processed_data = self.analyzer.prepare_with_trace(
            cv_path=str(path), resume_id=resume_id, extract_types=self.extract_types, allow_ocr=allow_ocr)
        return {
            "path": str(path),
            "filename": filename or os.path.basename(str(path)),
            "resume_id": resume_id,
            "trace": trace,
            "processed_data": processed_data,
            "needs_ocr": bool(trace.get("needs_ocr")),
            "read_seconds": round(time.perf_counter() - started, 3),
        }

    # -- Stages 2 + 3: text -> fields (GPU) -- #
    def extract(self, documents: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run the LLM over read() documents and post-process the answers.

        All prompts of all documents are batched together, so several resumes cost
        little more GPU time than the slowest one. Returns one result per document:

            {"filename", "status", "warnings", "error", "data", "seconds"}

        status is "success", "partial" (some output was cut off; see warnings),
        "empty" or "error". data is the parsed resume (basicInfo, workExperience,
        education, projects).
        """
        documents = [dict(document) for document in documents]
        for document in documents:
            if document.get("needs_ocr"):
                ## Deferred by read(allow_ocr=False): OCR runs here, next to the GPU
                redone = self.read(document["path"], allow_ocr=True, filename=document["filename"])
                redone["read_seconds"] += document.get("read_seconds", 0)
                document.update(redone)

        ready = [d for d in documents if d.get("processed_data") is not None]
        llm_seconds = 0.0
        if ready:
            started = time.perf_counter()
            try:
                outputs, call_logs = self.analyzer.llm_client.extract_info_batch(
                    [(d["resume_id"], d["trace"]["indexed_text_content"]) for d in ready],
                    self.extract_types,
                    batch_size=self.batch_size,
                    return_calls=True,
                )
                for document, structure_output, calls in zip(ready, outputs, call_logs):
                    document["trace"]["llm_calls"] = calls
                    self.analyzer.finish_with_trace(document["trace"], document["processed_data"], structure_output)
            except Exception as exc:
                logger.error("LLM batch failed: %s", exc, exc_info=True)
                for document in ready:
                    document["trace"]["final_result"] = {"error": str(exc)}
            llm_seconds = time.perf_counter() - started

        return [self._result(document, llm_seconds, len(ready)) for document in documents]

    def parse(self, paths: Sequence[str]) -> List[Dict[str, Any]]:
        """read() + extract() for a list of files, in one process."""
        return self.extract([self.read(path) for path in paths])

    @staticmethod
    def _result(document: Dict[str, Any], llm_seconds: float, batch_files: int) -> Dict[str, Any]:
        trace = document["trace"]
        final = trace.get("final_result") or {}
        warnings: List[str] = []
        incomplete = False
        for call in trace.get("llm_calls") or []:
            where = f"{call['extract_type']} [{call['section']}]"
            stop_reason = call.get("stop_reason")
            if stop_reason in _CUT_OFF:
                incomplete = True
                warnings.append(f"{where} {_CUT_OFF[stop_reason]}; kept {call['entries']} entries")
            elif stop_reason == "error":
                incomplete = True
                warnings.append(f"{where} failed: {call.get('error')}")

        error = str(final.get("error") or "")
        if error:
            status, data = "error", {}
        elif not final:
            status, data = "empty", {}
        else:
            status, data = ("partial" if incomplete else "success"), final
            warnings += list((final.get("metadata") or {}).get("issues") or [])
        return {
            "filename": document["filename"],
            "status": status,
            "warnings": warnings,
            "error": error,
            "data": data,
            "seconds": {
                "read": document.get("read_seconds", 0),
                ## The LLM call is shared by every file of the batch
                "llm_batch": round(llm_seconds, 3),
                "batch_files": batch_files,
            },
        }


def to_csv_row(result: Dict[str, Any]) -> Dict[str, str]:
    """One extract() result as a CSV row with the columns of process_resumes.py."""
    from smartresume.utils.schema import flatten_result

    row = flatten_result(result.get("data") or {})
    row["filename"] = result.get("filename", "")
    row["extraction_status"] = result.get("status", "")
    row["warnings"] = "; ".join(result.get("warnings") or [])
    row["error_message"] = result.get("error", "")
    return row
