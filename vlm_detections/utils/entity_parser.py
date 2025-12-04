"""Utilities for parsing entity properties and relations from VLM text outputs."""

from __future__ import annotations

import logging
from typing import List, Set, Optional

from vlm_detections.core.parsed_items import EntityProperty, EntityRelation
from vlm_detections.utils.json_parser import extract_json_objects

logger = logging.getLogger(__name__)


def parse_entity_properties(
    text: str,
    valid_properties: Optional[Set[str]] = None,
    threshold: float = 0.0
) -> List[EntityProperty]:
    """Parse entity properties from VLM output text.
    
    Expected JSON structure:
    - entity/subject: string identifier for the entity (e.g., "person_1", "robot")
    - property/property_name: property name (e.g., "emotion", "color", "state")
    - value/property_value: property value (e.g., "happy", "red", "sitting")
    - score/confidence: optional confidence score (0.0 to 1.0, defaults to 1.0)
    
    Handles nested structures like {"properties": [{...}, {...}]} or flat arrays.
    
    Args:
        text: Model output text containing JSON-formatted entity properties.
        valid_properties: Optional set of valid property names to filter by.
                         If None, all properties are accepted.
        threshold: Minimum confidence score to include a property (default 0.0).
        
    Returns:
        List of EntityProperty objects.
        
    Example:
        >>> text = '[{"entity": "person_1", "property": "emotion", "value": "happy", "score": 0.92}]'
        >>> props = parse_entity_properties(text)
        >>> len(props)
        1
        >>> props[0].entity
        'person_1'
    """
    properties: List[EntityProperty] = []
    
    # Collect candidate objects to parse
    candidates = []
    for obj in extract_json_objects(text):
        # Check if this is a wrapper object with nested arrays
        if isinstance(obj, dict):
            # Look for common wrapper keys
            for key in ['properties', 'attributes', 'entity_properties']:
                if key in obj and isinstance(obj[key], list):
                    candidates.extend([item for item in obj[key] if isinstance(item, dict)])
            # If no wrapper key found, treat the object itself as a candidate
            if not any(key in obj for key in ['properties', 'attributes', 'entity_properties']):
                candidates.append(obj)
    
    # Parse each candidate object
    for obj in candidates:
        # Extract entity identifier
        entity = obj.get("entity") or obj.get("subject")
        if not entity:
            continue
        
        # Extract property name
        property_name = obj.get("property") or obj.get("property_name")
        if not property_name:
            continue
        
        # Filter by valid properties if specified
        if valid_properties is not None and property_name not in valid_properties:
            continue
        
        # Extract property value
        property_value = obj.get("value") or obj.get("property_value")
        if not property_value:
            continue
        
        # Extract score
        score_val = obj.get("score") or obj.get("confidence")
        try:
            score = float(score_val) if score_val is not None else 1.0
        except (TypeError, ValueError):
            score = 1.0
        
        # Skip if below threshold
        if score < threshold:
            continue
        
        # Create entity property
        properties.append(EntityProperty(
            entity=str(entity),
            property_name=str(property_name),
            property_value=str(property_value),
            score=score
        ))
    
    return properties


def parse_entity_relations(
    text: str,
    valid_predicates: Optional[Set[str]] = None,
    threshold: float = 0.0
) -> List[EntityRelation]:
    """Parse entity relations from VLM output text.
    
    Expected JSON structure:
    - subject: string identifier for the subject entity (e.g., "person_1", "robot")
    - predicate/relation: relation type (e.g., "looking_at", "holding", "near")
    - object: string identifier for the object entity (e.g., "cup", "person_2")
    - score/confidence: optional confidence score (0.0 to 1.0, defaults to 1.0)
    
    Handles nested structures like {"relations": [{...}, {...}]} or flat arrays.
    
    Args:
        text: Model output text containing JSON-formatted entity relations.
        valid_predicates: Optional set of valid predicate names to filter by.
                         If None, all predicates are accepted.
        threshold: Minimum confidence score to include a relation (default 0.0).
        
    Returns:
        List of EntityRelation objects.
        
    Example:
        >>> text = '[{"subject": "person_1", "predicate": "holding", "object": "cup", "score": 0.88}]'
        >>> relations = parse_entity_relations(text)
        >>> len(relations)
        1
        >>> relations[0].predicate
        'holding'
    """
    relations: List[EntityRelation] = []
    
    # Collect candidate objects to parse
    candidates = []
    for obj in extract_json_objects(text):
        # Check if this is a wrapper object with nested arrays
        if isinstance(obj, dict):
            # Look for common wrapper keys
            for key in ['relations', 'relationships', 'entity_relations']:
                if key in obj and isinstance(obj[key], list):
                    candidates.extend([item for item in obj[key] if isinstance(item, dict)])
            # If no wrapper key found, treat the object itself as a candidate
            if not any(key in obj for key in ['relations', 'relationships', 'entity_relations']):
                candidates.append(obj)
    
    # Parse each candidate object
    for obj in candidates:
        # Extract subject
        subject = obj.get("subject")
        if not subject:
            continue
        
        # Extract predicate
        predicate = obj.get("predicate") or obj.get("relation")
        if not predicate:
            continue
        
        # Filter by valid predicates if specified
        if valid_predicates is not None and predicate not in valid_predicates:
            continue
        
        # Extract object
        object_entity = obj.get("object")
        if not object_entity:
            continue
        
        # Extract score
        score_val = obj.get("score") or obj.get("confidence")
        try:
            score = float(score_val) if score_val is not None else 1.0
        except (TypeError, ValueError):
            score = 1.0
        
        # Skip if below threshold
        if score < threshold:
            continue
        
        # Create entity relation
        relations.append(EntityRelation(
            subject=str(subject),
            predicate=str(predicate),
            object=str(object_entity),
            score=score
        ))
    
    return relations


def parse_all_entities(
    text: str,
    valid_properties: Optional[Set[str]] = None,
    valid_predicates: Optional[Set[str]] = None,
    threshold: float = 0.0
) -> tuple[List[EntityProperty], List[EntityRelation]]:
    """Parse both entity properties and relations from VLM output text.
    
    This is a convenience function that parses both types in a single pass
    over the JSON objects in the text.
    
    Args:
        text: Model output text containing JSON-formatted entities.
        valid_properties: Optional set of valid property names to filter by.
        valid_predicates: Optional set of valid predicate names to filter by.
        threshold: Minimum confidence score to include items (default 0.0).
        
    Returns:
        Tuple of (properties, relations).
    """
    properties = parse_entity_properties(text, valid_properties, threshold)
    relations = parse_entity_relations(text, valid_predicates, threshold)
    return properties, relations
