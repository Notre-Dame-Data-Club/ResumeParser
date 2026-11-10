#!/usr/bin/env python3
"""
Resume run script for Windows
- Invoked by run_processor.bat
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

#-- CONSTANTS and PATHS --#
ROOT_DIR = Path(__file__).resolve().parent # Anchor everything to the scripts parent's directory (ROOT) in case setup is changed.

## Find folders / files in root directory
LOGS_DIR = ROOT_DIR / "logs"
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"
CONFIG_EXAMPLE_PATH = ROOT_DIR / "configs" / "config.yaml.example"

# -- Checks -- #
def ensure_config_exists() -> None:
    """Create configs/config.yaml from .example file if missing."""
    if CONFIG_PATH.exists(): return
    if CONFIG_EXAMPLE_PATH.exists():
        shutil.copy(CONFIG_EXAMPLE_PATH, CONFIG_PATH)
        logger.info("Created %s from example.", CONFIG_PATH)
    else:
        logger.warning("No config.yaml or .example file exist. Please create compatible config file: %s", CONFIG_EXAMPLE_PATH)


def apply_gpu_memory_limit(gb: float) -> None:
    '''
    GPU limit to make sure your computer doesn't die in the name of Data Club.
    You are welcome :)
    '''
    try:
        import torch
        if _cuda_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            fraction = min(gb / total_gb, 1.0)
            torch.cuda.set_per_process_memory_fraction(fraction)
            logger.info("GPU memory capped at %.1f GB", gb)
    except Exception as exc:
        logger.warning("Could not apply GPU memory limit: %s", exc)

@functools.lru_cache(maxsize=1)
def _cuda_available() -> bool:
    '''Global CUDA availability check.'''
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# -- Imports -- #
def _load_analyzer_class():
    '''
    Load ResumeAnalyzer class from smartresume package.
    Brings the main resume parsing logic into script.
    '''
    try:
        from smartresume.backend.resume_analyzer import ResumeAnalyzer
        return ResumeAnalyzer
    except ImportError as exc:
        logger.error(
            "SmartResume package not found. "
            "Install it from the repo root with:  pip install -e ."
        )
        raise SystemExit(1) from exc

def _load_extract_types() -> tuple[str, ...]:
    '''
    Load the categories to extract from resumes.
    '''
    try:
        from smartresume.utils.prompts import EXTRACT_TYPES
        return EXTRACT_TYPES
    except ImportError as exc:
        logger.error(
            "SmartResume package not found. "
            "Install it from the repo root with:  pip install -e ."
        )
        raise SystemExit(1) from exc

def _load_config():
    '''
    Load the live SmartResume config singleton (shared across imports).
    '''
    try:
        from smartresume.utils.config import config
        return config
    except ImportError as exc:
        logger.error(
            "SmartResume package not found. "
            "Install it from the repo root with:  pip install -e ."
        )
        raise SystemExit(1) from exc

@functools.lru_cache(maxsize=1)
def _load_schema_module():
    '''
    Base CSV schema import for exporting stage.
    '''
    try:
        from smartresume.utils import schema
    except ImportError as exc:
        logger.error(
            "SmartResume package not found. "
            "Install it from the repo root with:  pip install -e ."
        )
        raise SystemExit(1) from exc

    ## A missing definition means an incomplete schema, not a missing package
    missing = [
        name for name in (
            "CSV_SCHEMA",
            "CSV_COLUMNS",
            "LIST_SECTIONS",
            "KIND_LIST_JSON",
            "validate_csv_schema",
        )
        if not hasattr(schema, name)
    ]
    if missing:
        logger.error(
            "smartresume/utils/schema.py is incomplete. Missing: %s. "
            "The CSV schema lives there; restore the missing definitions.",
            ", ".join(missing),
        )
        raise SystemExit(1)

    ## Warn if a prompt key was renamed without updating the schema
    for problem in schema.validate_csv_schema():
        logger.warning("CSV schema no longer matches the prompts — %s", problem)

    return schema



# -- Logging -- #
def setup_logging(log_file: str | Path = "processing.log") -> None:
    '''Setup logging handlers: console logging & backup log files.'''
    logging.root.handlers.clear() # Clear any existing handlers

    log_path = Path(log_file)
    if not log_path.is_absolute():
        log_path = ROOT_DIR / log_path

    ## File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    ## Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    ## Formatting log messages
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler],
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ProgressTracker:
    """Tracks batch progress within console, including ETA and speed."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.processed = 0
        self.start_time = time.time()
        self.file_times: list[float] = []

    def update(self, elapsed: float) -> None:
        """Record completion of one file and print progress."""
        self.processed += 1
        self.file_times.append(elapsed)
        self._print_progress()

    def _print_progress(self) -> None:
        elapsed_total = time.time() - self.start_time
        avg_file_time = sum(self.file_times) / len(self.file_times)
        remaining_files = self.total - self.processed
        eta_seconds = avg_file_time * remaining_files

        ## ETA format
        eta_mins = int(eta_seconds // 60)
        eta_secs = int(eta_seconds % 60)
        eta_str = f"{eta_mins}m{eta_secs}s" if eta_mins > 0 else f"{eta_secs}s"

        speed = self.processed / elapsed_total if elapsed_total > 0 else 0
        progress_pct = (self.processed / self.total) * 100

        ## Print inline progress
        status = (
            f"[{self.processed}/{self.total}] "
            f"{progress_pct:5.1f}% | "
            f"{speed:.2f} files/sec | "
            f"Last: {self.file_times[-1]:.1f}s | "
            f"ETA: {eta_str}"
        )
        logger.info(status)

        # Flush both stdout and file handlers for immediate display
        sys.stdout.flush()
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()

def log_device_info() -> None:
    '''Log GPU/CPU info if PyTorch is available.'''
    try:
        import torch
        if _cuda_available():
            props = torch.cuda.get_device_properties(0)
            logger.info(
                "GPU detected: %s  (%.1f GB VRAM)",
                props.name,
                props.total_memory / 1024 ** 3,
            )
        else:
            logger.warning("No CUDA GPU detected. running on CPU (slower).")
    except ImportError:
        logger.warning("PyTorch not importable. cannot detect GPU.")

# -- Data Processing -- #
def csv_columns() -> list[str]:
    '''CSV header order, taken from the schema.'''
    return list(_load_schema_module().CSV_COLUMNS)

def _walk_path(entry: dict, path: tuple) -> object:
    '''Follow nested keys through an extracted entry; "" if any step is missing.'''
    value: object = entry
    for key in path:
        if not isinstance(value, dict):
            return ""
        value = value.get(key, "")
    return "" if value is None else value


def _clean_value(value: object) -> object:
    '''Strip strings; keep lists/numbers as-is so json.dumps preserves types.'''
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return [_clean_value(v) for v in value]
    return value


def _scalar_cell(value: object) -> str:
    '''Scalar cell: plain text. A list (e.g. URLs) becomes a ", " joined string.'''
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v).strip() for v in value if str(v).strip())
    return str(value or "").strip()


def flatten_extracted_data(raw: dict) -> dict:
    """
    Flatten pipeline output into a single row. Transforms data to fit 1 resume per row.

    Scalar columns are plain strings. list_json columns are a JSON array with one
    object per resume entry (see schema.py); written with json.dumps so cells parse.
    """
    schema = _load_schema_module()
    row: dict[str, str] = {}

    for col in schema.CSV_SCHEMA:
        ## Columns the caller fills in (filename, extraction_status, ...)
        if not col.section:
            row[col.name] = ""
            continue

        section = raw.get(col.section)

        if col.kind == schema.KIND_LIST_JSON:
            entries = section if isinstance(section, list) else []
            cell = [
                {
                    spec.out_key: _clean_value(_walk_path(entry, spec.path))
                    for spec in col.fields
                }
                for entry in entries
                if isinstance(entry, dict)
            ]
            row[col.name] = json.dumps(cell, ensure_ascii=False) if cell else ""
        else:
            entry = section if isinstance(section, dict) else {}
            row[col.name] = _scalar_cell(entry.get(col.field, ""))

    return row

# --Terminal Processing/Connection -- #

class WindowsResumeProcessor:
    """
    Main connection with .bat file
    """

    def __init__(
        self,
        pdf_raw_text: bool,
        force_ocr: bool,
        no_layout_detection: bool,
        page_limit: int,
    ) -> None:
        ResumeAnalyzer = _load_analyzer_class()
        self.EXTRACT_TYPES = _load_extract_types()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_dir = LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_device_info()

        # Apply performance tweaks before initialization 
        self._config_snapshot: dict | None = None
        self._apply_config_overrides(
            pdf_raw_text, force_ocr, no_layout_detection, page_limit
        )

        try:
            logger.info("Initialising ResumeAnalyzer (loading model + OCR)…")
            self.analyzer = ResumeAnalyzer()
            logger.info("Analyzer ready.")
        except Exception:
            self._restore_config_overrides()
            raise

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(k): WindowsResumeProcessor._json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [WindowsResumeProcessor._json_safe(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    def _write_parse_history(
        self,
        pdf_path: Path,
        resume_id: str,
        extraction_status: str,
        trace: dict,
        flat_result: dict,
    ) -> None:
        timestamp = datetime.now().isoformat(timespec="seconds")
        history_record = {
            "timestamp": timestamp,
            "run_id": self.run_id,
            "resume_id": resume_id,
            "file_path": str(pdf_path.resolve()),
            "filename": pdf_path.name,
            "extract_types": self.EXTRACT_TYPES,
            "extraction_status": extraction_status,
            "flat_result": flat_result,
            "trace": trace,
        }
        safe_record = self._json_safe(history_record)

        detail_path = self.logs_dir / f"{self.run_id}_{resume_id}_{pdf_path.stem}.json"
        detail_path.write_text(
            json.dumps(safe_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        history_path = self.logs_dir / "parsing_history.jsonl"
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(safe_record, ensure_ascii=False) + "\n")

        logger.info("Parse history saved -> %s", detail_path)

    def _apply_config_overrides(
        self,
        pdf_raw_text: bool,
        force_ocr: bool,
        no_layout_detection: bool,
        page_limit: int,
    ) -> None:
        """
        Apply the resolved CLI values onto the live config singleton, for this run only.
        These values already default to whatever configs/config.yaml has on disk (see
        build_parser), so this is a no-op unless a flag was explicitly passed with a
        value different from what's already configured. Never touches config.yaml itself.
        """
        config = _load_config()
        self._config_snapshot = {
            "use_pdf_raw_text": config.processing.use_pdf_raw_text,
            "use_force_ocr": config.processing.use_force_ocr,
            "page_limit": config.processing.page_limit,
            "layout_detection_enabled": config.layout_detection.enabled,
        }
        layout_detection_enabled = not no_layout_detection

        if pdf_raw_text != self._config_snapshot["use_pdf_raw_text"]:
            logger.info(
                "Config: PDF raw text extraction %s [this run only]",
                "enabled" if pdf_raw_text else "disabled",
            )
        if force_ocr != self._config_snapshot["use_force_ocr"]:
            logger.info(
                "Config: Force OCR %s [this run only]",
                "enabled" if force_ocr else "disabled",
            )
        if layout_detection_enabled != self._config_snapshot["layout_detection_enabled"]:
            logger.info(
                "Config: Layout detection %s [this run only]",
                "enabled" if layout_detection_enabled else "disabled",
            )
        if page_limit != self._config_snapshot["page_limit"]:
            logger.info("Config: Page limit set to %d [this run only]", page_limit)

        config.processing.use_pdf_raw_text = pdf_raw_text
        config.processing.use_force_ocr = force_ocr
        config.layout_detection.enabled = layout_detection_enabled
        config.processing.page_limit = page_limit

    def _restore_config_overrides(self) -> None:
        """Restore the config singleton to its pre-run values."""
        snapshot = self._config_snapshot
        if not snapshot:
            return
        config = _load_config()
        config.processing.use_pdf_raw_text = snapshot["use_pdf_raw_text"]
        config.processing.use_force_ocr = snapshot["use_force_ocr"]
        config.processing.page_limit = snapshot["page_limit"]
        config.layout_detection.enabled = snapshot["layout_detection_enabled"]
        self._config_snapshot = None

    def __enter__(self) -> "WindowsResumeProcessor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._restore_config_overrides()


    def process_directory(
        self,
        directory: Path,
        output_file: Path,
        save_json: bool = False,
    ) -> int:
        """
        Process every *.pdf in *directory*, write a consolidated CSV,
        and optionally a JSON file.  Returns the count of successes.
        Displays real-time progress with ETA.
        """

        ## Get .pdf files in directory (case-insensitive for cross-platform use)
        pdf_files = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() == ".pdf"
        )
        if not pdf_files:
            logger.warning("No PDF files found in: %s", directory)
            return 0

        total = len(pdf_files)
        logger.info("Found %d PDF file(s) in '%s'.", total, directory)
        logger.info("=" * 70)

        results: list[dict] = []
        progress = ProgressTracker(total)

        for idx, pdf_path in enumerate(pdf_files, start=1):
            file_start = time.time()
            logger.info("[%d/%d] Processing: %s", idx, total, pdf_path.name)

            result = self._process_one(pdf_path, resume_id=f"resume_{idx:04d}")
            results.append(result)

            file_elapsed = time.time() - file_start
            progress.update(file_elapsed)

        logger.info("=" * 70)
        self._write_csv(results, output_file)

        if save_json:
            json_path = output_file.with_suffix(".json")
            json_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("JSON saved -> %s", json_path)

        successes = sum(1 for r in results if r.get("extraction_status") == "success")
        total_time = time.time() - progress.start_time
        avg_per_file = total_time / total if total > 0 else 0
        logger.info(
            "Finished — %d/%d succeeded.  Total time: %.1fs (avg %.1fs/file).  Results: %s",
            successes, total, total_time, avg_per_file, output_file,
        )

        # Ensure all log output is flushed before returning
        sys.stdout.flush()
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        return successes

    # Process a single PDF (Option 2)
    def process_single(
        self,
        pdf_path: Path,
        output_file: Path,
        save_json: bool = False,
    ) -> int:
        """
        Process a single PDF, write a one-row CSV, and optionally a JSON file.
        Returns 1 on success, 0 otherwise.
        """
        logger.info("Processing: %s", pdf_path.name)
        file_start = time.time()

        result = self._process_one(pdf_path, resume_id="resume_0001")
        file_elapsed = time.time() - file_start

        self._write_csv([result], output_file)

        if save_json:
            json_path = output_file.with_suffix(".json")
            json_path.write_text(
                json.dumps([result], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("JSON saved -> %s", json_path)

        status = result.get("extraction_status")
        logger.info(
            "Finished — %s (%.1fs).  Results: %s",
            status, file_elapsed, output_file,
        )

        sys.stdout.flush()
        for handler in logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        return 1 if status == "success" else 0

    def _process_one(self, pdf_path: Path, resume_id: str) -> dict:
        """Run the SmartResume pipeline on a single file; return a flat row dict."""
        base: dict = {
            "filename": pdf_path.name,
            "extraction_status": "error",
            "error_message": "",
        }

        ## Handles pipeline call, error handling and flattening
        try:
            trace = self.analyzer.pipeline_with_trace(
                cv_path=str(pdf_path),
                resume_id=resume_id,
                extract_types=self.EXTRACT_TYPES,
            )
            raw = trace.get("final_result") or {}

            if raw.get("error"):
                error_message = raw.get("error")
                error_details = raw.get("error_details") or {}
                if not error_message:
                    error_message = error_details.get("error_message", "Unknown parsing error")
                base["error_message"] = error_message
                self._write_parse_history(
                    pdf_path=pdf_path,
                    resume_id=resume_id,
                    extraction_status=base["extraction_status"],
                    trace=trace,
                    flat_result=base,
                )
                return base

            if not raw:
                base["extraction_status"] = "empty"
                logger.warning("Empty result for: %s", pdf_path.name)
                self._write_parse_history(
                    pdf_path=pdf_path,
                    resume_id=resume_id,
                    extraction_status=base["extraction_status"],
                    trace=trace,
                    flat_result=base,
                )
                return base

            flat = flatten_extracted_data(raw)
            flat["filename"]          = pdf_path.name
            flat["extraction_status"] = "success"
            flat["error_message"]     = ""
            self._write_parse_history(
                pdf_path=pdf_path,
                resume_id=resume_id,
                extraction_status=flat["extraction_status"],
                trace=trace,
                flat_result=flat,
            )
            return flat

        except Exception as exc:
            logger.error(
                "Failed on %s: %s", pdf_path.name, exc, exc_info=True
            )
            base["error_message"] = str(exc)
            return base

    @staticmethod
    def _write_csv(results: list[dict], output_file: Path) -> None:
        df = pd.DataFrame(results).reindex(columns=csv_columns())
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        logger.info("CSV saved -> %s  (%d rows)", output_file, len(df))

def build_parser() -> argparse.ArgumentParser:
    ''' Terminal parser for flag idetnification'''
    config = _load_config()

    p = argparse.ArgumentParser(
        prog="process_resumes.py",
        description="Resume batch processor - Windows / Qwen3-0.6B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Process default directory: python process_resumes.py\n"
            "  Process specific directory: python process_resumes.py -d /path/to/pdfs -o results.csv\n"
            "  Process a single file:    python process_resumes.py -f resumes/john_doe.pdf -o result.csv\n"
            "  Fast mode (text-only):    python process_resumes.py --pdf-raw-text\n"
            "  Skip layout detection:    python process_resumes.py --no-layout-detection\n"
            "  Limit pages/GPU memory:   python process_resumes.py --page-limit 5 --gpu-memory 8\n"
            "  Better OCR extraction:    python process_resumes.py --force-ocr\n"
        ),
    )

    ## Configs
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("-d", "--directory", metavar="DIR", default="resumes/",
                             help="Process all PDFs in this directory (default: resumes/)")
    mode_group.add_argument("-f", "--file", metavar="FILE",
                             help="[Option 2] Process a single resume file")
    p.add_argument("-o", "--output", metavar="FILE", default="resume_results.csv",
                   help="Output CSV path (default: resume_results.csv)")
    p.add_argument("--gpu-memory", metavar="GB", type=float,
                   help="Cap GPU VRAM usage in GB")
    p.add_argument("--save-json", action="store_true",
                   help="Also write a full JSON file alongside the CSV")
    p.add_argument("--pdf-raw-text", action="store_true",
                   default=config.processing.use_pdf_raw_text,
                   help="Use raw PDF text extraction (skip OCR). ~2–3× faster for text-only PDFs. "
                        f"(default: {config.processing.use_pdf_raw_text}, from config.yaml)")
    p.add_argument("--force-ocr", action="store_true",
                   default=config.processing.use_force_ocr,
                   help="Force OCR on all pages. Better for garbled/scanned PDFs, slower. "
                        f"(default: {config.processing.use_force_ocr}, from config.yaml)")
    p.add_argument("--no-layout-detection", action="store_true",
                   default=not config.layout_detection.enabled,
                   help="Disable YOLO layout detection. Faster, worse on complex layouts. "
                        f"(default: {not config.layout_detection.enabled}, from config.yaml layout_detection.enabled)")
    p.add_argument("--page-limit", metavar="N", type=int,
                   default=config.processing.page_limit,
                   help="Process only first N pages per resume. Saves time on long CVs. "
                        f"(default: {config.processing.page_limit}, from config.yaml)")

    return p

def main() -> None:
    '''Main Function'''
    setup_logging()
    ensure_config_exists()
    args = build_parser().parse_args()

    if args.file:
        file_path = Path(args.file).expanduser()
        if not file_path.is_file():
            logger.error("File not found: %s", file_path)
            raise SystemExit(1)

        if args.gpu_memory:
            apply_gpu_memory_limit(args.gpu_memory)

        with WindowsResumeProcessor(
            pdf_raw_text=args.pdf_raw_text,
            force_ocr=args.force_ocr,
            no_layout_detection=args.no_layout_detection,
            page_limit=args.page_limit,
        ) as processor:
            processor.process_single(
                pdf_path=file_path,
                output_file=Path(args.output).expanduser(),
                save_json=args.save_json,
            )
        return

    directory = Path(args.directory).expanduser()
    if not directory.is_dir():
        logger.error("Directory not found: %s", directory)
        raise SystemExit(1)

    if args.gpu_memory:
        apply_gpu_memory_limit(args.gpu_memory)

    with WindowsResumeProcessor(
        pdf_raw_text=args.pdf_raw_text,
        force_ocr=args.force_ocr,
        no_layout_detection=args.no_layout_detection,
        page_limit=args.page_limit,
    ) as processor:
        processor.process_directory(
            directory=directory,
            output_file=Path(args.output).expanduser(),
            save_json=args.save_json,
        )


if __name__ == "__main__":
    main()
