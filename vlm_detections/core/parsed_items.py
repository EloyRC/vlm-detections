"""Data structures for items that can be parsed from VLM outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Detection:
    """A bounding box detection with optional polygon segmentation and textual content.
    
    Attributes:
        xyxy: Bounding box coordinates (x1, y1, x2, y2) in absolute pixel coordinates.
        score: Confidence score (typically 0.0 to 1.0).
        label: Class/category label for this detection.
        polygon: Optional polygon segmentation as list of (x,y) points or list of lists for multi-part regions.
        text: Optional textual content (e.g., OCR text, region caption).
    """
    xyxy: Tuple[float, float, float, float]
    score: float
    label: str
    polygon: Optional[List[Tuple[float, float]]] = None
    text: Optional[str] = None


@dataclass
class EntityProperty:
    """A property attributed to an entity (subject).
    
    Attributes:
        entity: The subject entity (e.g., "person", "robot", "object_id_5").
        property_name: The property name from a predefined set (e.g., "emotion", "color", "state").
        property_value: The value of the property (e.g., "happy", "red", "sitting").
        score: Confidence score for this attribution (typically 0.0 to 1.0).
    """
    entity: str
    property_name: str
    property_value: str
    score: float = 1.0


@dataclass
class EntityRelation:
    """A relation between two entities: <subject, predicate, object>.
    
    Attributes:
        subject: The subject entity (e.g., "person_1", "robot").
        predicate: The relation type from a predefined set (e.g., "looking_at", "holding", "near").
        object: The object entity (e.g., "cup", "person_2", "door").
        score: Confidence score for this relation (typically 0.0 to 1.0).
    """
    subject: str
    predicate: str
    object: str
    score: float = 1.0
