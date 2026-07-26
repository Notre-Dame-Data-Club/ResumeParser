<div align=center>
    <img width="100" height="100" alt="DataClubLogo" src="https://github.com/user-attachments/assets/602b9792-222f-4d02-9273-0641dedbdba0" />
   <h1>Resume Scraping StarterPack </h1>
   <p>Simplified English Version of <i>Alibaba/SmartResume</i><p>
  <h6>Adapted by the Data Club of Notre Dame<h6>
  <img alt="Qwen" src="https://img.shields.io/badge/Qwen-6950EF?style=flat&logo=qwen&logoColor=white"/> <img alt="YOLO" src="https://img.shields.io/badge/YOLO-111F68?style=flat&logo=yolo&logoColor=white"> <img alt="Transformers" src="https://img.shields.io/badge/transformers-%23FFD21E.svg?style=flat&logo=huggingface&logoColor=white"/> <img alt="Pytorch" src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white"/> <img alt="NVIDIA" src="https://img.shields.io/badge/cuda-000000.svg?style=flat&logo=NVIDIA&logoColor=green"/> 
</div>


###### Quickstart: [`SmartResume_Colab.ipynb`](notebooks/SmartResume_Colab.ipynb) 
    
## Overview

Bulk resume scraper and parser, transforming .pdf content into .JSON and .CSV files for easier processing.

**Includes:**
Text Extraction, OCR Computer Vision processing, YOLOv10 layout detection, and local Qwen3 field extraction.

### Additions & Changes

- Guided Collab notebooks.
- Simplified py commands.
- Windows .bat startup files with CUDA support.
- Project wide English translation.
- .csv file generation.

## Running

### Colab Notebooks

1. Open [`SmartResume_Colab.ipynb`](notebooks/SmartResume_Colab.ipynb) in Google Colab.
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
| `--save-json` | Also write a full JSON alongside the CSV | NULL |
| --------------------------------- | --------------------------------- | --------------------------------- |


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
| `direct_model_name` | Path to the local LLM | `models/Qwen3-0.6B` |
| `layout_detection.enabled` | Use YOLOv10 layout detection | `true` |
| `ocr.use_cuda` | Run OCR on GPU | `true` |
| `processing.use_pdf_raw_text` | Skip OCR for text-only PDFs | `false` |
| `processing.use_force_ocr` | Force OCR on every page | `false` |

## Troubleshooting

`processing.log` and `setup.log` live in the root directory.
`logs/`.include log per resume

> Only PDFs are considered. 
> All other formats will be ignored. 

## Attribution & License

Cloned from [alibaba/SmartResume](https://github.com/alibaba/SmartResume). 
Licensed under [**Apache-2.0**](LICENSE).
