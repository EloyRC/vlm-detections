"""Configuration loader for model_config.yaml used by ROS nodes."""
from __future__ import annotations

import os
import logging
from typing import Dict, Any, List
from importlib import resources

import yaml

logger = logging.getLogger(__name__)

_MODEL_CONFIG_RESOURCE = "model_config.yaml"


def load_model_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load model configuration from YAML file.
    
    Args:
        config_path: Optional path to config file. If None, uses default from package-level config.
        
    Returns:
        Dictionary with configuration. Returns default config if loading fails.
    """
    default_config = {
        'model': 'OpenAI Vision (API)',
        'model_variant': 'Qwen/Qwen2.5-VL-32B-Instruct-AWQ',
        'threshold': 0.25,
        'device': 'auto',
        'generation_params': {},
    }
    
    # Try to load from provided path (highest priority)
    if config_path and os.path.isfile(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as fp:
                config = yaml.safe_load(fp) or {}
            logger.info(f"Loaded model config from {config_path}")
            return _validate_and_merge_config(config, default_config)
        except Exception as exc:
            logger.warning(f"Failed to load model config from {config_path}: {exc}")
    
    # Try to load from ROS package share directory
    import pathlib
    config_file = None
    try:
        from ament_index_python.packages import get_package_share_directory
        package_share = get_package_share_directory('vlm_detections')
        config_file = pathlib.Path(package_share) / "config" / _MODEL_CONFIG_RESOURCE
    except (ImportError, Exception):
        # Fallback to source tree (development)
        config_file = pathlib.Path(__file__).resolve().parents[2] / "config" / _MODEL_CONFIG_RESOURCE
    
    try:
        with config_file.open("r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp) or {}
        logger.info(f"Loaded model config from {config_file}")
        return _validate_and_merge_config(config, default_config)
    except FileNotFoundError:
        logger.warning(f"Model config file '{_MODEL_CONFIG_RESOURCE}' not found; using defaults.")
        return default_config
    except Exception as exc:
        logger.warning(f"Failed to load model config from package config ({exc}); using defaults.")
        return default_config


def _validate_and_merge_config(config: Dict[str, Any], default: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and merge loaded config with defaults."""
    merged = default.copy()
    
    # Validate and merge basic fields
    if 'model' in config and isinstance(config['model'], str):
        merged['model'] = config['model']
    
    if 'model_variant' in config and isinstance(config['model_variant'], str):
        merged['model_variant'] = config['model_variant']
    
    if 'threshold' in config and isinstance(config['threshold'], (int, float)):
        merged['threshold'] = float(config['threshold'])
    
    if 'device' in config and isinstance(config['device'], str):
        merged['device'] = config['device']
    
    # Merge generation parameters
    if 'generation_params' in config and isinstance(config['generation_params'], dict):
        merged['generation_params'] = config['generation_params']
    
    return merged


def resolve_config_path(path_param: str) -> str:
    """Resolve configuration file path, handling environment variables and relative paths."""
    if not path_param:
        return ''
    
    # Expand environment variables
    expanded = os.path.expandvars(path_param)
    
    # Expand user home directory
    expanded = os.path.expanduser(expanded)
    
    # Convert to absolute path if relative
    if not os.path.isabs(expanded):
        expanded = os.path.abspath(expanded)
    
    return expanded


__all__ = ['load_model_config', 'resolve_config_path']
