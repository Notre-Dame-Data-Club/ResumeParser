"""
Gradio app for a Hugging Face Space on ZeroGPU (also runs on any GPU or CPU machine).

Three chained steps per request, so the GPU is only held for the middle one:

    read_files   CPU   PDF text + layout detection (files needing OCR are deferred)
    extract      GPU   all prompts of all files in batched generate() calls
    show         CPU   JSON, table and CSV download

To use the parser inside another Space, copy the four marked lines: import spaces
first, build ResumeParser at module level, decorate the function Gradio calls.
"""
import os
import tempfile
import time

try:
    import spaces  # must come before torch: it prepares CUDA for ZeroGPU    <-- 1
except ImportError:  # not on Spaces and `pip install spaces` not done: run undecorated
    class spaces:  # type: ignore[no-redef]
        @staticmethod
        def GPU(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda function: function

import gradio as gr
import pandas as pd

from smartresume.service import ResumeParser, to_csv_row
from smartresume.utils.schema import CSV_COLUMNS

## Loaded once, at import: ZeroGPU moves weights placed on "cuda" here to the real
## GPU when a request arrives.                                               <-- 2
parser = ResumeParser()

MAX_FILES = int(os.environ.get("RESUME_MAX_FILES", "10"))
## Columns shown in the table (the CSV has all of them)
_TABLE_COLUMNS = ["filename", "extraction_status", "full_name", "email", "phone", "location", "warnings"]


def read_files(files):
    """Stage 1 on the CPU. OCR is not started here; such files are read on the GPU."""
    if not files:
        raise gr.Error("Upload at least one resume (PDF).")
    if len(files) > MAX_FILES:
        raise gr.Error(f"Up to {MAX_FILES} files per request.")
    return [parser.read(path, allow_ocr=False) for path in files]


def _gpu_seconds(documents, *args, **kwargs):
    """GPU time to reserve: asking for less ranks the request higher in the queue."""
    documents = documents or []
    ocr_files = sum(1 for document in documents if document.get("needs_ocr"))
    return min(120, 30 + 8 * len(documents) + 15 * ocr_files)


@spaces.GPU(duration=_gpu_seconds)  # the function Gradio binds is the decorated one  <-- 3
def extract(documents):
    started = time.perf_counter()
    results = parser.extract(documents or [])  #                                       <-- 4
    for result in results:
        result["seconds"]["gpu_call"] = round(time.perf_counter() - started, 2)
    return results


def show(results):
    results = results or []
    rows = [to_csv_row(result) for result in results]
    table = pd.DataFrame(rows).reindex(columns=list(CSV_COLUMNS))
    ## A fresh file per request: concurrent users must not share an output path
    handle = tempfile.NamedTemporaryFile(prefix="resume_results_", suffix=".csv", delete=False)
    handle.close()
    table.to_csv(handle.name, index=False, encoding="utf-8-sig")

    timing = ""
    if results:
        seconds = results[0]["seconds"]
        read = sum(result["seconds"]["read"] for result in results)
        timing = (f"{len(results)} file(s): read {read:.1f}s on CPU, "
                  f"GPU call {seconds.get('gpu_call', 0):.1f}s (LLM {seconds['llm_batch']:.1f}s)")
    return [result["data"] | {"_file": result["filename"], "_status": result["status"]} for result in results], \
        table[_TABLE_COLUMNS], handle.name, timing


with gr.Blocks(title="Resume Parser") as demo:
    gr.Markdown("# Resume Parser\nUpload resume PDFs to get structured fields (JSON and CSV).")
    files = gr.File(label="Resumes (PDF)", file_count="multiple", file_types=[".pdf"])
    button = gr.Button("Parse", variant="primary")
    timing = gr.Markdown()
    table = gr.Dataframe(label="Summary", interactive=False)
    download = gr.File(label="CSV")
    parsed = gr.JSON(label="Parsed resumes")

    documents = gr.State()
    results = gr.State()
    button.click(read_files, files, documents) \
        .then(extract, documents, results) \
        .then(show, results, [parsed, table, download, timing])

if __name__ == "__main__":
    demo.launch()
