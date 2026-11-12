<div align=center>
    <img width="100" height="100" alt="DataClubLogo" src="https://github.com/user-attachments/assets/602b9792-222f-4d02-9273-0641dedbdba0" />
   <h1>Resume Parser StarterPack </h1>
   <p>Simplified English Version of <i>Alibaba/SmartResume</i><p>
  <h6>Adapted by the Data Club of Notre Dame<h6>
  <img alt="Gemma" src="https://img.shields.io/badge/Gemma-4285F4?style=flat&logo=google&logoColor=white"/> <img alt="YOLO" src="https://img.shields.io/badge/YOLO-111F68?style=flat&logo=yolo&logoColor=white"> <img alt="Transformers" src="https://img.shields.io/badge/transformers-%23FFD21E.svg?style=flat&logo=huggingface&logoColor=white"/> <img alt="Pytorch" src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white"/> <img alt="NVIDIA" src="https://img.shields.io/badge/cuda-000000.svg?style=flat&logo=NVIDIA&logoColor=green"/> 
</div>


###### Notebook Quickstart: [`Simple.ipynb`](notebooks/Simple.ipynb)
    
## Overview

Bulk resume scraper and parser, transforming .pdf content into .JSON and .CSV files for easier processing.

**Includes:**
Text Extraction, OCR Computer Vision processing, YOLOv10 layout detection, and local Gemma 4 E4B (NF4) field extraction.

### Additions & Changes

- Guided Collab notebooks.
- Simplified py commands.
- Windows .bat startup files with CUDA support.
- Project wide English translation.
- .csv file generation.

## Running

### Colab Notebooks

1. Open [`Simple.ipynb`](notebooks/Simple.ipynb) in Google Colab.
2. **Runtime → Run all**.
3. Upload Resumes to /resumes when prompted.
4. Download the generated `resume_results.csv`.

<div></br></div>

> First run takes ~10–15min on free plan CPU/GPU as model download takes place.   
> Students are encouraged to use the [free** PRO plan](https://blog.google/products-and-platforms/products/education/colab-higher-education/) and select higher power GPU's.

***Google is not accepting new students for the offer - **LAST CHECKED:** 07/25/2026*

---

### Local Run on Windows

1. Run [`setup.bat`](setup.bat) to install dependancies & models.
2. Drop your resume PDFs into the [`/resumes`](resumes/) folder.
3. Run **[`run_processor.bat`](run_processor.bat)** and select decired options when prompted.

<div></br></div>

> Optimized for NVIDIA GPU, works on CPU for other devices.

---

### Python Command Run

```bash
python process_resumes.py
```

| Flag | Effect | Default |
| ------ | -------- | -------- |
| `-d, --directory DIR` | Folder with PDFs to process | [`/resumes`](/resumes) |
| `-o, --output FILE` | Output CSV file name | `resume_results.csv` |
| --------------------------------- | --------------------------------- | --------------------------------- |
| `--pdf-raw-text` | Skip OCR for text-only PDFs (~2–3× faster) | NULL |
| `--force-ocr` | Force OCR on all pages (better for scanned PDFs) | NULL |
| `--no-layout-detection` | Disable YOLO layout detection (faster, simpler PDFs) | NULL |
| `--page-limit N` | Only process the first N pages per resume | NULL |
| `--gpu-memory GB` | Cap GPU VRAM usage | NULL |
| `--quantization MODE` | `nf4` (least VRAM), `none` (fastest) or `auto` — see [Speed and memory](#speed-and-memory) | `nf4` |
| `--batch-size N` | Most LLM prompts per `generate()` call | `16` |
| `--save-json` | Also write a full JSON alongside the CSV | NULL |
| --------------------------------- | --------------------------------- | --------------------------------- |


---

### Hugging Face Space (ZeroGPU)

[`app.py`](app.py) is a ready Gradio app for a Space on ZeroGPU hardware. It reads the PDFs on the CPU, holds the GPU only for the LLM step, and returns JSON plus a CSV.

1. Create a Gradio Space, select **ZeroGPU** hardware, and push this repository to it.
2. Put this block at the very top of the Space's `README.md`:

```yaml
---
title: Resume Parser
sdk: gradio
app_file: app.py
---
```

3. If the model repo is gated for your account, add an `HF_TOKEN` secret in the Space settings.

To use the parser inside an existing Space instead, install this repo there and call it directly:

```python
import spaces                                  # before anything that imports torch
from smartresume.service import ResumeParser

parser = ResumeParser()                        # module level: weights are placed on "cuda" here

def read_files(paths):                         # CPU
    return [parser.read(p, allow_ocr=False) for p in paths]

@spaces.GPU(duration=60)                       # the function Gradio binds
def extract(documents):
    return parser.extract(documents)           # [{"filename", "status", "warnings", "data", ...}]
```

| Variable | Meaning | Default on ZeroGPU |
| --- | --- | --- |
| `RESUME_MODEL` | Hugging Face repo id or local folder of the LLM | `google/gemma-4-E4B-it` |
| `RESUME_QUANTIZATION` | `none`, `nf4` or `auto` | `none` |
| `RESUME_DTYPE` | `bfloat16` or `float16` | `bfloat16` |
| `RESUME_BATCH_SIZE` | Most prompts per `generate()` call | `16` |
| `RESUME_MAX_FILES` | Files accepted per request (`app.py`) | `10` |

> The first start downloads the model and writes the text-only export (several minutes, ~31 GB of disk).
> To skip the export on later starts, upload `models/gemma-4-E4B-it-text/` to a model repo of your own once
> and set `RESUME_MODEL` to it.

## Speed and memory

A `generate()` call costs *steps × time per step*: steps is the token count of the longest answer in the call, and the time per step barely changes with the number of rows. Each call logs both (the numbers below only show the format):

```
generate(): 6 row(s) x 1180 prompt tokens, 412 steps in 9.8s (prompt 0.4s, then 23 ms/step); embedding lookups in RAM 0.2s; VRAM peak 4.10 GB, free now 1.20 of 6.0 GB
```

| Setting | Effect |
| --- | --- |
| `quantization: nf4` | ~4 GB VRAM. bitsandbytes unpacks every weight on every step, so steps are slower. Use on GPUs under 14 GB. |
| `quantization: none` | ~10 GB VRAM. Fastest steps and no quantization loss. Use `compute_dtype: bfloat16` on RTX 30-series or newer. |
| `--batch-size` | More rows per call means fewer calls for a folder of resumes. Calls are sized to the free VRAM automatically. |
| `processing.skip_ocr_without_images` | OCR runs only on pages that can hold text outside the text layer (pictures, outlined text, scans). |

## Output Flow

<div align=center>
<a href="smartresume/utils/prompts.py"><b>smartresume/utils/prompts.py - complete JSON schema</b></a>

<p><b>\/</b></p>

<a href="smartresume/utils/schema.py"><b>smartresume/utils/schema.py - Defines fields included as CSV columns</b></a>

<p><b>\/</b></p>

<a href="process_resumes.py"><b>process_resumes.py - Creates CSV</b></a>
</div>


## Configuration

The project's configurations live in [`configs/config.yaml`](/configs/config.yaml).
In case of error [`configs/config.yaml.example`](/configs/config.yaml.example) serves as a backup with default settings.

| Key | Meaning | Default |
|-----|---------|---------|
| `use_direct_models` | Load the model in-process (API connection is included in the original alibaba/SmartResume. It has not been tested here) | `true` |
| `direct_model_name` | Hugging Face repo id of the LLM, downloaded to `models/<name>` on first use (a local folder path also works) | `google/gemma-4-E4B-it` |
| `direct_model.quantization` | `nf4` (4-bit, bitsandbytes), `none` (full precision) or `auto` (`none` on GPUs with 14 GB or more) | `nf4` |
| `direct_model.compute_dtype` | `float16` or `bfloat16` | `float16` |
| `direct_model.offload_per_layer_embeddings` | Keep Gemma's 5.6 GB per-layer embedding table in RAM (needs ~6 GB free RAM; a one-time text-only export is written to `models/<name>-text/`) | `true` |
| `direct_model.max_new_tokens` | Output token cap per prompt | `1024` |
| `direct_model.max_new_tokens_by_type` | Lower caps for extraction types with short answers | `basic_info: 256` |
| `layout_detection.enabled` | Use YOLOv10 layout detection | `true` |
| `ocr.use_cuda` | Run OCR on GPU | `true` |
| `processing.use_pdf_raw_text` | Skip OCR for text-only PDFs | `false` |
| `processing.skip_ocr_without_images` | OCR only pages where it can add text; `false` runs OCR on every page | `true` |
| `processing.use_force_ocr` | Force OCR on every page | `false` |

## Troubleshooting

`processing.log` and `setup.log` live in the root directory.
`logs/`.include log per resume

> Only PDFs are considered. 
> All other formats will be ignored. 

## Attribution & License

Cloned from [alibaba/SmartResume](https://github.com/alibaba/SmartResume). 
Licensed under [**Apache-2.0**](LICENSE).
