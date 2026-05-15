"""Public exception types raised inside user-supplied Pydantic validators."""
from __future__ import annotations


class AggregatedValidationError(ValueError):
    """Raise from inside a Pydantic validator that aggregates N underlying problems
    into a single error message, to declare the true weight to stitcher's
    catastrophic-re-extract threshold.

    Example:
        >>> from pydantic import BaseModel, model_validator
        >>> from stitcher import AggregatedValidationError
        >>>
        >>> class MyOutput(BaseModel):
        ...     items: list[str]
        ...
        ...     @model_validator(mode="after")
        ...     def _check(self):
        ...         missing = expected_items() - set(self.items)
        ...         if missing:
        ...             raise AggregatedValidationError(
        ...                 f"{len(missing)} items missing: {sorted(missing)}",
        ...                 count=len(missing),
        ...             )
        ...         return self

    Stitcher sums the ``count`` of each AggregatedValidationError raised
    in a single validation pass; if the total exceeds
    ``max_validation_error_weight``, the patch loop is abandoned and a
    fresh extract is performed instead. Vanilla ``ValueError`` instances
    raised by validators contribute weight 1 each.
    """

    def __init__(self, message: str, *, count: int):
        super().__init__(message)
        self.count = count
