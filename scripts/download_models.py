#!/usr/bin/env python3
"""
SmartResume Model Downloader
"""
import os
import sys
import argparse
from pathlib import Path

# Set project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from smartresume.utils.models_download_utils import download_model
    from smartresume.utils.model_paths import ModelType, ModelSource
except ImportError as e:
    print(f"Import failed: {e}")
    print("Please run this script from the project root directory")
    sys.exit(1)


def _pull_with_ollama() -> bool:
    """
    With llm_backend: ollama, the LLM is an Ollama model, not a Hugging Face folder.
    Returns False (after saying why) when Ollama cannot be used, so the caller
    downloads the transformers model instead.
    """
    from smartresume.utils.config import config
    if getattr(config, "llm_backend", "transformers") != "ollama":
        return False
    from smartresume.model import ollama_backend
    if not ollama_backend.find_executable():
        print("Ollama is not installed (https://ollama.com/download): it runs the LLM several times "
              "faster on laptop GPUs.\nDownloading the slower transformers model instead...")
        return False
    try:
        print("Downloading the LLM into Ollama (about 6 GB the first time)...")
        print(f"Ollama model ready: {ollama_backend.pull_model(getattr(config, 'ollama', {}))}")
        return True
    except Exception as e:
        print(f"Ollama download failed ({e}).\nDownloading the transformers model instead...")
        return False


def download_models(model_type: str = "all", source: str = "huggingface", save_path: str = None):
    """Download models"""
    print(f"Starting to download {model_type} model from {source}")
    if save_path:
        print(f"Models will be saved to: {save_path}")
    
    try:
        model_source_enum = ModelSource(source)
        model_type_enum = ModelType(model_type)
    except ValueError as e:
        print(f"Parameter error: {e}")
        return False
    
    try:
        if model_type_enum in (ModelType.ALL, ModelType.LLM) and _pull_with_ollama():
            if model_type_enum == ModelType.LLM:
                return True
            # The LLM is in Ollama; only the layout model is left
            model_type, model_type_enum = "layout", ModelType.LAYOUT

        if model_type_enum == ModelType.ALL:
            # Download all models
            print("Downloading LLM model...")
            llm_path = download_model(ModelType.LLM, model_source_enum, save_path)
            print(f"LLM model downloaded successfully: {llm_path}")

            print("Downloading Layout model...")
            layout_path = download_model(ModelType.LAYOUT, model_source_enum, save_path)
            print(f"Layout model downloaded successfully: {layout_path}")
            
            print("All models downloaded successfully!")
            return True
        else:
            # Download specified model
            print(f"Downloading {model_type} model...")
            model_path = download_model(model_type_enum, model_source_enum, save_path)
            print(f"{model_type} model downloaded successfully: {model_path}")
            return True
            
    except Exception as e:
        print(f"Error occurred while downloading models: {str(e)}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='SmartResume Model Download Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python scripts/download_models.py                    # Download all models to models directory
  python scripts/download_models.py --model_type llm   # Download LLM model to models directory
  python scripts/download_models.py --source huggingface  # Download from Hugging Face to models directory
  python scripts/download_models.py --save_path /path/to/models  # Specify custom save path
        """
    )
    
    parser.add_argument(
        '--model_type',
        choices=['llm', 'layout', 'all'],
        default='all',
        help='Model type to download (default: all)'
    )
    
    parser.add_argument(
        '--source',
        choices=['huggingface'],
        default='huggingface',
        help='Model download source (default: huggingface)'
    )
    
    parser.add_argument(
        '--save_path',
        type=str,
        default='models',
        help='Custom path to save downloaded models (default: models)'
    )
    
    args = parser.parse_args()
    
    # Download models
    success = download_models(args.model_type, args.source, args.save_path)
    
    if success:
        print("\nNow you can analyze resumes with:")
        print("python process_resumes.py -o resume_results.csv")
        print("(or run run_processor.bat on Windows)")
        return 0
    else:
        print("\nModel download failed!")
        return 1


if __name__ == '__main__':
    exit(main())