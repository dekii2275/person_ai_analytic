"""Framework-independent M04 body attribute contracts."""

from src.body_attributes.base import BaseBodyAttributeRecognizer
from src.body_attributes.schemas import (
    BODY_ATTRIBUTE_NAMESPACE,
    BODY_ATTRIBUTE_SCHEMA_VERSION,
    BodyAttributeKey,
    BodyAttributePrediction,
)

__all__ = [
    "BODY_ATTRIBUTE_NAMESPACE",
    "BODY_ATTRIBUTE_SCHEMA_VERSION",
    "BaseBodyAttributeRecognizer",
    "BodyAttributeKey",
    "BodyAttributePrediction",
]
