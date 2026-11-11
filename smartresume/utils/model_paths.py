"""
Model paths configuration for SmartResume
"""
from enum import Enum


class ModelPath(Enum):
    """Model path configurations for different model sources"""

    # Hugging Face repositories (https://huggingface.co/Alibaba-EI/SmartResume)
    SMART_RESUME_ROOT_HF = "Alibaba-EI/SmartResume"
    YOLOV10_ROOT_HF = "Alibaba-EI/SmartResume"

    # Default LLM (overridden by direct_model_name in config.yaml)
    GEMMA_4_E4B_IT_HF = "google/gemma-4-E4B-it"

    # Specific model paths
    YOLOV10_MODEL = "yolov10/best.onnx"

    # Model types
    LLM_MODEL = "llm"
    LAYOUT_MODEL = "layout"


class ModelType(Enum):
    """Model types for SmartResume"""
    LLM = "llm"           # Large Language Model for text extraction
    LAYOUT = "layout"     # Layout detection model
    ALL = "all"           # All models


class ModelSource(Enum):
    """Model download sources"""
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
