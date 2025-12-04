"""Base class for generative VLM adapters with generation parameter management."""

from __future__ import annotations

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GenerativeVLMBase:
    """
    Base class providing common generation parameter management for VLM adapters.
    
    Provides:
    - Generation parameter specification schema
    - Parameter validation and coercion
    - Default value management
    - Type conversion and range clamping
    
    Subclasses should:
    - Initialize self.gen_params: Dict[str, object] = {}
    - Initialize self._gen_spec_cache: Optional[Dict[str, Dict]] = None
    - Implement generation_config_spec() to define their parameters
    """
    
    gen_params: Dict[str, object]
    _gen_spec_cache: Optional[Dict[str, Dict]]
    
    def _gen_spec(self) -> Dict[str, Dict]:
        """Lazy-load and cache generation spec."""
        if self._gen_spec_cache is None:
            self._gen_spec_cache = self.generation_config_spec()
        return self._gen_spec_cache
    
    def _coerce_generation_value(self, name: str, value: object) -> object:
        """
        Coerce and clamp generation parameter value according to spec.
        
        Args:
            name: Parameter name
            value: Raw parameter value
            
        Returns:
            Coerced and validated value
            
        Raises:
            ValueError: If value cannot be coerced to expected type
        """
        meta = self._gen_spec().get(name)
        if meta is None:
            return value
        if value is None:
            raise ValueError(f"Generation parameter '{name}' cannot be None")
        
        ptype = meta.get("type")
        try:
            if ptype == "int":
                coerced = int(value)  # type: ignore[arg-type]
                if "min" in meta:
                    coerced = max(int(meta["min"]), coerced)
                if "max" in meta:
                    coerced = min(int(meta["max"]), coerced)
                return coerced
            
            if ptype == "float":
                coerced = float(value)  # type: ignore[arg-type]
                if "min" in meta:
                    coerced = max(float(meta["min"]), coerced)
                if "max" in meta:
                    coerced = min(float(meta["max"]), coerced)
                return coerced
            
            if ptype == "bool":
                if isinstance(value, str):
                    coerced = value.strip().lower() in {"1", "true", "yes", "on"}
                else:
                    coerced = bool(value)
                return coerced
        except (TypeError, ValueError) as exc:
            raise exc
        
        return value
    
    def _get_generation_value(self, name: str) -> object:
        """
        Get generation parameter value with fallback to default.
        
        Args:
            name: Parameter name
            
        Returns:
            Sanitized parameter value
        """
        spec = self._gen_spec()
        if name not in spec:
            return self.gen_params.get(name)
        
        raw_value = self.gen_params.get(name, spec[name].get("default"))
        try:
            coerced = self._coerce_generation_value(name, raw_value)
        except (TypeError, ValueError):
            default_val = spec[name].get("default")
            try:
                coerced = self._coerce_generation_value(name, default_val)
            except (TypeError, ValueError):
                logger.warning("Falling back to unsanitized default for generation param %s", name)
                coerced = default_val
        
        self.gen_params[name] = coerced
        return coerced
    
    def _reset_generation_defaults(self) -> None:
        """Reset gen_params to spec defaults. Should be called after load()."""
        defaults: Dict[str, object] = {}
        for key, meta in self._gen_spec().items():
            default_val = meta.get("default")
            try:
                defaults[key] = self._coerce_generation_value(key, default_val)
            except (TypeError, ValueError):
                logger.warning("Invalid default %r for generation param %s", default_val, key)
                defaults[key] = default_val
        self.gen_params = defaults
    
    def update_generation_params(self, params: Dict[str, object]) -> None:
        """
        Update generation parameters from external source (UI/API).
        
        Args:
            params: Dictionary of parameter name -> value
        """
        spec = self._gen_spec()
        for k, v in params.items():
            if k not in spec:
                continue
            try:
                self.gen_params[k] = self._coerce_generation_value(k, v)
            except (TypeError, ValueError):
                logger.warning("Invalid value %r for generation param %s", v, k)
    
    def generation_config_spec(self) -> Dict[str, Dict[str, Any]]:
        """
        Return schema of tunable generation parameters.
        
        Subclasses should override this method to define their parameters.
        
        Returns:
            Mapping of param_name -> metadata dict with keys:
            - type: "int", "float", or "bool"
            - default: default value
            - min, max, step: for numeric parameters
            - label: human-readable label
            - help: optional help text
        """
        raise NotImplementedError("Subclasses must implement generation_config_spec()")
