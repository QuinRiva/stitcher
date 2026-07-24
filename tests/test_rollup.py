"""Behaviour 4 — validation-error rollup in ``_format_validation_errors``.

The renderer groups byte-identical ``(type, message)`` errors: the explanation
is shown ONCE (the validator's own prose, verbatim), followed by the list of
every JSON-Pointer path it occurs at. A lone error keeps its full prose with
its path inline. First-seen order is preserved, and the Pydantic
``"Value error, "`` prefix is stripped exactly once so a leading ``###`` heading
renders instead of leaking as literal text.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, model_validator

from stitcher.extractor import (
    _build_patch_prompt,
    _format_validation_errors,
)


_LIFECYCLE = (
    "\n### `lifecycle_events` must be present\n\n"
    "`lifecycle_events` was omitted, which is indistinguishable from "
    "forgetting to look. Emit an empty list `[]`."
)


def _err(loc, msg, err_type="value_error"):
    return {"loc": loc, "msg": msg, "type": err_type}


def test_empty_errors():
    assert _format_validation_errors([]) == "(no specific errors reported)"


def test_identical_class_rolled_up_once_with_all_paths():
    errs = [_err(("entities", i), f"Value error, {_LIFECYCLE}") for i in range(7)]
    out = _format_validation_errors(errs)
    assert out.count("was omitted, which is indistinguishable") == 1
    assert "**Occurs at these 7 paths:**" in out
    for i in range(7):
        assert f"- `/entities/{i}`" in out


def test_value_error_prefix_stripped_and_heading_renders():
    out = _format_validation_errors([_err(("entities", 0), f"Value error, {_LIFECYCLE}")])
    assert "Value error," not in out
    # A blank line precedes the `###` heading so it renders (not literal text).
    assert "\n\n### `lifecycle_events` must be present" in out


def test_single_occurrence_keeps_full_prose_and_inline_path():
    out = _format_validation_errors([_err(("entities", 3), f"Value error, {_LIFECYCLE}")])
    assert "at `/entities/3`" in out
    assert "occurrences" not in out
    assert "was omitted, which is indistinguishable" in out


def test_distinct_messages_not_merged():
    errs = [
        _err(("entities", 0), "Value error, \n### A\n\nfirst class."),
        _err(("entities", 1), "Value error, \n### B\n\nsecond class."),
    ]
    out = _format_validation_errors(errs)
    assert "first class." in out
    assert "second class." in out
    assert "occurrences" not in out


def test_grouping_preserves_first_seen_order():
    errs = [
        _err(("entities", 0), "Value error, \n### First\n\nalpha."),
        _err(("entities", 1), "Value error, \n### Second\n\nbeta."),
        _err(("entities", 2), "Value error, \n### First\n\nalpha."),
    ]
    out = _format_validation_errors(errs)
    assert out.index("alpha.") < out.index("beta.")
    assert "[1]" in out and "[2]" in out


def test_root_loc_renders_as_root_label():
    out = _format_validation_errors([_err((), "boom", err_type="value_error")])
    assert "at `(root)`" in out


@pytest.mark.asyncio
async def test_build_patch_prompt_uses_rollup_end_to_end(fake_llm):
    """A real multi-occurrence ValidationError routed through the base
    extractor's ``_build_patch_prompt`` collapses to one explanation with
    exactly one ``## Validation Errors`` prefix."""

    class Item(BaseModel):
        n: int

        @model_validator(mode="after")
        def _check(self):
            if self.n < 0:
                raise ValueError(
                    "\n### `n` must be non-negative\n\nGot a negative value."
                )
            return self

    class Bag(BaseModel):
        items: list[Item]

    errs = []
    try:
        Bag.model_validate({"items": [{"n": -1}, {"n": -2}, {"n": -3}]})
    except Exception as e:  # ValidationError
        prompt = _build_patch_prompt(e, {"type": "object"})
        errs = e.errors()

    assert len(errs) == 3
    assert prompt.count("must be non-negative") == 1
    assert "**Occurs at these 3 paths:**" in prompt
    assert prompt.count("## Validation Errors") == 1
    assert "Value error," not in prompt
    for i in range(3):
        assert f"- `/items/{i}`" in prompt
