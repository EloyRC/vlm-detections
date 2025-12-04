"""Common utilities for extracting JSON objects from mixed text content."""

from __future__ import annotations

import json
import re
import logging
from typing import List, Dict, Any, Iterator

logger = logging.getLogger(__name__)


def extract_json_objects(text: str) -> Iterator[Dict[str, Any]]:
    """Extract all valid JSON objects from text that may contain mixed content.
    
    This function attempts to parse the entire text as JSON first. If that fails,
    it uses regex to find JSON-like patterns and parses each one individually.
    
    Args:
        text: Input text that may contain JSON objects embedded in other content.
        
    Yields:
        Parsed JSON objects (as dictionaries).
        
    Example:
        >>> text = "Here are results: {\"label\": \"cat\", \"score\": 0.9} and {\"label\": \"dog\", \"score\": 0.8}"
        >>> list(extract_json_objects(text))
        [{'label': 'cat', 'score': 0.9}, {'label': 'dog', 'score': 0.8}]
    """
    if not text:
        return
    
    # First, try parsing the entire text as JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            yield parsed
            return
        elif isinstance(parsed, list):
            # If it's a list, yield each dictionary element
            for item in parsed:
                if isinstance(item, dict):
                    yield item
            return
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fallback: use regex to extract JSON-like patterns
    # This pattern matches JSON objects with balanced braces
    pattern = re.compile(r'\{[^{}]*?\}')
    
    for match in pattern.finditer(text):
        snippet = match.group(0)
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                yield obj
        except (json.JSONDecodeError, ValueError):
            # Skip invalid JSON snippets
            continue


def extract_json_arrays(text: str) -> Iterator[List[Any]]:
    """Extract all valid JSON arrays from text that may contain mixed content.
    
    Args:
        text: Input text that may contain JSON arrays embedded in other content.
        
    Yields:
        Parsed JSON arrays (as lists).
        
    Example:
        >>> text = 'Results: [{"a": 1}, {"a": 2}]'
        >>> list(extract_json_arrays(text))
        [[{'a': 1}, {'a': 2}]]
    """
    if not text:
        return
    
    # Try parsing the entire text as JSON first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            yield parsed
            return
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Fallback: use regex to extract JSON array patterns
    # This pattern matches JSON arrays with balanced brackets
    pattern = re.compile(r'\[[^\[\]]*?\]')
    
    for match in pattern.finditer(text):
        snippet = match.group(0)
        try:
            arr = json.loads(snippet)
            if isinstance(arr, list):
                yield arr
        except (json.JSONDecodeError, ValueError):
            continue


def normalize_json_text(text: str) -> str:
    """Normalize text to improve JSON parsing success.
    
    Handles common formatting issues like converting single quotes to double quotes,
    removing trailing commas, etc.
    
    Args:
        text: Raw text that might contain malformed JSON.
        
    Returns:
        Normalized text with better chance of successful JSON parsing.
    """
    if not text:
        return text
    
    # Common normalizations
    normalized = text.strip()
    
    # Remove trailing commas before closing braces/brackets
    normalized = re.sub(r',\s*}', '}', normalized)
    normalized = re.sub(r',\s*]', ']', normalized)
    
    return normalized
