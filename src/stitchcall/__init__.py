"""stitchcall — single-schema LLM extraction with Pydantic validation and JSON-Patch repair.

Trust the model to be mostly right. Stitch the gaps.
"""
from stitchcall.exceptions import AggregatedValidationError
from stitchcall.extractor import Extractor, Result

__all__ = ["Extractor", "Result", "AggregatedValidationError"]
__version__ = "0.0.1"
