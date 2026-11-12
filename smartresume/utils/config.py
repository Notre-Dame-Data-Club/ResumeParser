"""
Configuration management module
"""
import os
import yaml
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration"""
    name: str = ""
    api_url: str = ""
    api_key: str = ""
    max_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    model_source: str = "huggingface"  # huggingface, local
    local_model_path: str = ""

    def validate(self) -> list[str]:
        """Validate configuration"""
        errors = []
        if not self.name:
            errors.append("Model name cannot be empty")
        if not self.api_url:
            errors.append("API URL cannot be empty")
        # Skip API key validation for direct models
        if not self.api_key and not (hasattr(self, 'use_direct_models') and self.use_direct_models):
            errors.append("API Key cannot be empty")
        if self.max_tokens <= 0:
            errors.append("max_tokens must be greater than 0")
        if not 0 <= self.temperature <= 2:
            errors.append("temperature must be between 0 and 2")
        if not 0 <= self.top_p <= 1:
            errors.append("top_p must be between 0 and 1")
        return errors


@dataclass
class ProcessingConfig:
    """Processing configuration"""
    use_force_ocr: bool = False
    use_force_json: bool = False
    use_pdf_raw_text: bool = False

    remove_position_and_company_line: bool = False
    page_limit: int = 10
    # Run OCR only on pages where it can add text (embedded pictures, outlined text,
    # no text layer). False = OCR every page, as before.
    skip_ocr_without_images: bool = True

    def validate(self) -> list[str]:
        """Validate configuration"""
        errors = []
        if self.page_limit <= 0:
            errors.append("page_limit must be greater than 0")
        return errors


@dataclass
class OCRConfig:
    """OCR configuration"""
    enabled: bool = True
    ocr_provider: str = "default"
    use_cuda: bool = True
    confidence_threshold: float = 0.5

    def validate(self) -> list[str]:
        """Validate configuration"""
        errors = []
        if not 0 <= self.confidence_threshold <= 1:
            errors.append("confidence_threshold must be between 0 and 1")
        return errors


@dataclass
class LayoutDetectionConfig:
    """Layout detection configuration"""
    enabled: bool = False

    def validate(self) -> list[str]:
        """Validate configuration"""
        errors = []
        # No validation needed for layout detection as model is auto-downloaded
        return errors


class Config:
    """Unified configuration management"""

    def __init__(self):
        self.model = ModelConfig()
        self.processing = ProcessingConfig()
        self.ocr = OCRConfig()

        self.layout_detection = LayoutDetectionConfig()
        self.channels = {}
        self.extract_channels = None
        self.extract_channels_main = None
        self.extract_channels_backup = None

        # Model download configuration
        self.model_download = {
            'source': 'huggingface',  # huggingface
            'models_dir': {
                'llm': '',
                'layout': ''
            },
            'auto_download': True
        }

        # Direct (local transformers) model: how to load it
        self.direct_model = {
            'quantization': 'nf4',
            'compute_dtype': 'float16',
            'offload_per_layer_embeddings': True,
            'max_new_tokens': 1024,
            'gpu_memory_fraction': 0.88,
        }

    def validate(self) -> list[str]:
        """Validate all configuration sections"""
        errors = []
        use_direct = hasattr(self, 'use_direct_models') and self.use_direct_models

        # model/channels/extract_channels are only relevant to the API path
        if not use_direct:
            errors.extend(self.model.validate())

        errors.extend(self.processing.validate())
        errors.extend(self.ocr.validate())
        errors.extend(self.layout_detection.validate())

        if not use_direct and hasattr(self, 'channels') and self.channels:
            for channel_name, channel_config in self.channels.items():
                if not hasattr(channel_config, 'name') or not channel_config.name:
                    errors.append(f"Channel {channel_name} is missing 'name'")
                if not hasattr(channel_config, 'api_url') or not channel_config.api_url:
                    errors.append(f"Channel {channel_name} is missing 'api_url'")
                # Skip API key validation for local channels
                if not hasattr(channel_config, 'api_key') or not channel_config.api_key:
                    if not (channel_name.startswith('local_') or 'local' in channel_name.lower()):
                        errors.append(f"Channel {channel_name} is missing 'api_key'")
                if not hasattr(channel_config, 'max_tokens') or channel_config.max_tokens <= 0:
                    errors.append(f"Channel {channel_name} has invalid 'max_tokens'")

        if not use_direct and hasattr(self, 'extract_channels') and self.extract_channels:
            for extract_type, channel_name in self.extract_channels.__dict__.items():
                if channel_name and channel_name not in getattr(self, 'channels', {}):
                    errors.append(f"Extract type {extract_type} references non-existent channel {channel_name}")

        return errors

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> 'Config':
        """Load configuration from YAML file"""
        config = cls()

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)

                # Direct models skip the OpenAI client entirely, so the extra
                # OpenAI-specific model fields (name/max_tokens/temperature/top_p/seed)
                # are irrelevant and shouldn't be required.
                use_direct_models = yaml_config.get('use_direct_models', False)

                # Update model config
                if 'model' in yaml_config:
                    model_config = yaml_config['model']
                    required_fields = (
                        ['api_url', 'api_key'] if use_direct_models
                        else ['name', 'api_url', 'api_key', 'max_tokens', 'temperature', 'top_p', 'seed']
                    )
                    missing_fields = [f for f in required_fields if f not in model_config]

                    for field_name, value in model_config.items():
                        if hasattr(config.model, field_name):
                            setattr(config.model, field_name, value)

                    if missing_fields:
                        raise ValueError(f"Model config missing required fields: {', '.join(missing_fields)}")
                elif not use_direct_models:
                    raise ValueError("Configuration file missing required 'model' section")

                # Load channels config
                if 'channels' in yaml_config:
                    channels_config = yaml_config['channels']
                    for channel_name, channel_data in channels_config.items():
                        channel_obj = type('Channel', (), {})()
                        for field, value in channel_data.items():
                            setattr(channel_obj, field, value)
                        config.channels[channel_name] = channel_obj

                # Load extract channel mapping
                if 'extract_channels' in yaml_config:
                    extract_channels_config = yaml_config['extract_channels']
                    extract_channels_obj = type('ExtractChannels', (), {})()

                    for extract_type, channel_name in extract_channels_config.items():
                        setattr(extract_channels_obj, extract_type, channel_name)

                    config.extract_channels = extract_channels_obj

                # Load primary extract channel mapping
                if 'extract_channels_main' in yaml_config:
                    extract_channels_main_config = yaml_config['extract_channels_main']
                    extract_channels_main_obj = type('ExtractChannelsMain', {}, {})()

                    for extract_type, channel_name in extract_channels_main_config.items():
                        setattr(extract_channels_main_obj, extract_type, channel_name)

                    config.extract_channels_main = extract_channels_main_obj

                # Load backup extract channel mapping
                if 'extract_channels_backup' in yaml_config:
                    extract_channels_backup_config = yaml_config['extract_channels_backup']
                    extract_channels_backup_obj = type('ExtractChannelsBackup', {}, {})()

                    for extract_type, channel_name in extract_channels_backup_config.items():
                        setattr(extract_channels_backup_obj, extract_type, channel_name)

                    config.extract_channels_backup = extract_channels_backup_obj

                # Load other config sections
                if 'processing' in yaml_config:
                    for field, value in yaml_config['processing'].items():
                        if hasattr(config.processing, field):
                            setattr(config.processing, field, value)

                if 'ocr' in yaml_config:
                    for field, value in yaml_config['ocr'].items():
                        if hasattr(config.ocr, field):
                            setattr(config.ocr, field, value)

                if 'layout_detection' in yaml_config:
                    for field, value in yaml_config['layout_detection'].items():
                        if hasattr(config.layout_detection, field):
                            setattr(config.layout_detection, field, value)

                # Load direct model config
                if 'use_direct_models' in yaml_config:
                    config.use_direct_models = yaml_config['use_direct_models']

                if 'direct_model_name' in yaml_config:
                    config.direct_model_name = yaml_config['direct_model_name']

                if isinstance(yaml_config.get('direct_model'), dict):
                    config.direct_model.update(yaml_config['direct_model'])

                # Load model download config
                if 'model_download' in yaml_config:
                    config.model_download.update(yaml_config['model_download'])

                # Validate configuration
                validation_errors = config.validate()
                if validation_errors:
                    raise ValueError(f"Configuration validation failed: {'; '.join(validation_errors)}")

                return config

            except Exception as e:
                raise ValueError(f"Failed to load configuration file: {e}")
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")


# Create global config instance
config = Config()

# Try to load configuration from file
try:
    config = Config.from_yaml("configs/config.yaml")
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(
        "Failed to load configs/config.yaml, using defaults: %s", e
    )
