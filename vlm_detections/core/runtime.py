"""Runtime management for vision-language model (VLM) detection adapters.

This module provides the core runtime infrastructure for managing and executing
zero-shot object detection using various vision-language models. It handles:

- Model registration and variant management for supported VLM adapters
- Dynamic loading and initialization of detection models via plugin system
- Runtime state management for active detector instances
- Configuration loading from YAML for model variants and adapter registry
- Prompt parsing and validation utilities

The DetectorRuntime class serves as the main interface for:
- Loading and switching between different detection models
- Managing model-specific parameters and generation configs
- Ensuring proper device placement and model lifecycle

Adapters are loaded dynamically from adapters_config.yaml, allowing for:
- Easy addition of new adapters without modifying core code
- Separation of public and private adapter implementations
- Flexible configuration of adapter constructor arguments
"""
from __future__ import annotations

import logging
import importlib
from importlib import resources
from typing import Callable, Dict, List, Any

import yaml

from vlm_detections.core.adapter_base import (
    BaseVisionAdapter,
    ZeroShotObjectDetector,
    PromptBasedVLM,
)

logger = logging.getLogger(__name__)

_MODEL_VARIANTS_RESOURCE = "model_variants.yaml"
_ADAPTERS_CONFIG_RESOURCE = "adapters_config.yaml"

_DEFAULT_MODEL_VARIANTS: Dict[str, List[str]] = {
    "OpenAI Vision (API)": [
        "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    ],
}


def _load_model_variants() -> Dict[str, List[str]]:
    """Load model variants from config file.
    
    Priority:
    1. VLM_CONFIG_DIR environment variable (for standalone app)
    2. ROS package share directory (for ROS nodes)
    3. Development source tree (for testing)
    """
    import pathlib
    import os
    
    # For standalone app
    config_dir = os.environ.get("VLM_CONFIG_DIR")
    if config_dir:
        config_path = pathlib.Path(config_dir) / _MODEL_VARIANTS_RESOURCE
    else:
        # Try ROS package share directory first
        try:
            from ament_index_python.packages import get_package_share_directory
            package_share = get_package_share_directory('vlm_detections')
            config_path = pathlib.Path(package_share) / "config" / _MODEL_VARIANTS_RESOURCE
        except (ImportError, Exception):
            # Fallback to source tree (development)
            config_path = pathlib.Path(__file__).resolve().parents[2] / "config" / _MODEL_VARIANTS_RESOURCE
    try:
        with config_path.open("r", encoding="utf-8") as fp:
            payload = yaml.safe_load(fp) or {}
    except FileNotFoundError:
        logger.warning("Model variants file '%s' not found; using defaults.", _MODEL_VARIANTS_RESOURCE)
        return {k: list(v) for k, v in _DEFAULT_MODEL_VARIANTS.items()}
    except Exception as exc:
        logger.warning("Failed to load model variants from YAML (%s); using defaults.", exc)
        return {k: list(v) for k, v in _DEFAULT_MODEL_VARIANTS.items()}

    if not isinstance(payload, dict):
        logger.warning("Model variants YAML must be a mapping; using defaults.")
        return {k: list(v) for k, v in _DEFAULT_MODEL_VARIANTS.items()}

    normalized: Dict[str, List[str]] = {}
    for model_name, variants in payload.items():
        if isinstance(variants, (list, tuple)):
            values = [str(v) for v in variants if isinstance(v, (str, bytes))]
            normalized[str(model_name)] = values
        else:
            logger.debug("Skipping model '%s' with non-sequence variants in YAML", model_name)

    if not normalized:
        logger.warning("Model variants YAML was empty after parsing; using defaults.")
        return {k: list(v) for k, v in _DEFAULT_MODEL_VARIANTS.items()}

    merged = {k: list(v) for k, v in _DEFAULT_MODEL_VARIANTS.items()}
    merged.update(normalized)
    return merged


MODEL_VARIANTS = _load_model_variants()


def _load_adapter_registry() -> Dict[str, Callable[[str], BaseVisionAdapter]]:
    """Load adapter registry from configuration file.
    
    Adapters are defined in adapters_config.yaml with format:
    adapters:
      "Model Name":
        module: "python.module.path"
        class: "AdapterClassName"
        constructor_args:  # optional
          arg_name: value
    
    This allows easy addition of new adapters without modifying core code.
    
    Priority:
    1. VLM_CONFIG_DIR environment variable (for standalone app)
    2. ROS package share directory (for ROS nodes)
    3. Development source tree (for testing)
    """
    import pathlib
    import os
    
    # For standalone app
    config_dir = os.environ.get("VLM_CONFIG_DIR")
    if config_dir:
        config_path = pathlib.Path(config_dir) / _ADAPTERS_CONFIG_RESOURCE
    else:
        # Try ROS package share directory first
        try:
            from ament_index_python.packages import get_package_share_directory
            package_share = get_package_share_directory('vlm_detections')
            config_path = pathlib.Path(package_share) / "config" / _ADAPTERS_CONFIG_RESOURCE
        except (ImportError, Exception):
            # Fallback to source tree (development)
            config_path = pathlib.Path(__file__).resolve().parents[2] / "config" / _ADAPTERS_CONFIG_RESOURCE
    registry: Dict[str, Callable[[str], BaseVisionAdapter]] = {}
    
    try:
        with config_path.open("r", encoding="utf-8") as fp:
            config = yaml.safe_load(fp) or {}
    except FileNotFoundError:
        logger.warning("Adapters config file '%s' not found; registry will be empty.", _ADAPTERS_CONFIG_RESOURCE)
        return registry
    except Exception as exc:
        logger.warning("Failed to load adapters config from YAML (%s); registry will be empty.", exc)
        return registry

    adapters_config = config.get("adapters", {})
    if not isinstance(adapters_config, dict):
        logger.warning("Adapters config 'adapters' must be a mapping; registry will be empty.")
        return registry

    for model_name, adapter_spec in adapters_config.items():
        if not isinstance(adapter_spec, dict):
            logger.debug("Skipping adapter '%s' with invalid spec", model_name)
            continue
            
        module_path = adapter_spec.get("module")
        class_name = adapter_spec.get("class")
        constructor_args = adapter_spec.get("constructor_args", {})
        
        if not module_path or not class_name:
            logger.warning("Adapter '%s' missing 'module' or 'class' field; skipping", model_name)
            continue
        
        try:
            # Dynamically import the adapter class
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)
            
            # Create a factory function that includes both model_id and constructor_args
            def make_factory(cls: type, args: Dict[str, Any]) -> Callable[[str], BaseVisionAdapter]:
                def factory(model_id: str) -> BaseVisionAdapter:
                    return cls(model_id=model_id, **args)
                return factory
            
            registry[model_name] = make_factory(adapter_class, constructor_args)
            logger.debug("Registered adapter '%s' from %s.%s", model_name, module_path, class_name)
            
        except (ImportError, AttributeError) as exc:
            logger.warning("Failed to load adapter '%s' from %s.%s: %s", 
                         model_name, module_path, class_name, exc)
            continue
    
    return registry


MODEL_REGISTRY: Dict[str, Callable[[str], BaseVisionAdapter]] = _load_adapter_registry()


def default_variant_for(model_name: str) -> str:
    variants = MODEL_VARIANTS.get(model_name) or ["default"]
    return variants[0]


def ensure_valid_variant(model_name: str, variant: str) -> str:
    variants = MODEL_VARIANTS.get(model_name)
    if not variants:
        return variant
    return variant if variant in variants else variants[0]


def parse_prompts(text: str) -> List[str]:
    segments = [segment.strip() for segment in text.replace("\n", ",").split(",")]
    return [segment for segment in segments if segment]


def is_zero_shot_detector(adapter: BaseVisionAdapter) -> bool:
    """Check if an adapter implements the ZeroShotObjectDetector protocol.
    
    Args:
        adapter: The adapter instance to check
        
    Returns:
        True if the adapter implements ZeroShotObjectDetector protocol
    """
    return isinstance(adapter, ZeroShotObjectDetector)


def is_prompt_based_vlm(adapter: BaseVisionAdapter) -> bool:
    """Check if an adapter implements the PromptBasedVLM protocol.
    
    Args:
        adapter: The adapter instance to check
        
    Returns:
        True if the adapter implements PromptBasedVLM protocol
    """
    return isinstance(adapter, PromptBasedVLM)


class DetectorRuntime:
    """Lightweight helper that owns adapter instances and shared detector state."""

    def __init__(self) -> None:
        self.current_model_name: str | None = None
        self.current_model_id: str | None = None
        self.detector: BaseVisionAdapter | None = None
        self.model_gen_params: Dict[str, Dict[str, object]] = {}

    def ensure_model(self, model_name: str, model_id: str, device_choice: str) -> None:
        if (
            self.current_model_name == model_name
            and self.current_model_id == model_id
            and self.detector is not None
        ):
            return

        factory = MODEL_REGISTRY.get(model_name)
        if factory is None:
            raise KeyError(f"Unknown model '{model_name}'")

        logger.info("Loading model '%s' (%s)", model_name, model_id)
        detector = factory(model_id)
        detector.load(device=device_choice)

        self.detector = detector
        self.current_model_name = model_name
        self.current_model_id = model_id

    def get_generation_spec(self) -> Dict[str, Dict]:
        if self.detector is None:
            return {}
        spec_fn = getattr(self.detector, "generation_config_spec", None)
        if callable(spec_fn):
            try:
                spec = spec_fn()
                return spec or {}
            except Exception:  # pragma: no cover - adapter specific failures
                return {}
        return {}

    def apply_generation_params(self, params: Dict[str, object]) -> None:
        if self.detector is None:
            return
        update_fn = getattr(self.detector, "update_generation_params", None)
        if callable(update_fn):
            try:
                update_fn(params)
            except Exception:  # pragma: no cover - adapter specific failures
                logger.debug("Adapter rejected generation params", exc_info=True)


__all__ = [
    "DetectorRuntime",
    "MODEL_REGISTRY",
    "MODEL_VARIANTS",
    "default_variant_for",
    "ensure_valid_variant",
    "parse_prompts",
    "is_zero_shot_detector",
    "is_prompt_based_vlm",
]
