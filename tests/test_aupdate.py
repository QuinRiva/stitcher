"""``aupdate`` applies a user update to an existing instance via the patch loop.

Behavioural contract (mirrors the docstring on ``Extractor.aupdate``):

- Always runs at least one LLM call. There is no zero-LLM-call optimisation
  for "existing already validates" — that would conflate update with
  verify-and-repair.
- The initial JSON-mode extract is **never** called.
- The first patch turn carries no validator framing (no ``<errors>`` block,
  no "your previous output failed validation" prefix). The user's update
  intent in ``messages`` directs *what* to patch; the prompt only specifies
  *how*.
- Subsequent turns (if validation fails after the first patch) prepend the
  validator-failure prefix exactly as in ``ainvoke``.
- Accepts either a Pydantic ``BaseModel`` instance or a JSON-serialisable
  ``dict`` as ``existing``.
- ``run_name`` threads through to every patch turn as ``f"{run_name}.patch"``;
  initial-extract names are unused (no initial extract in update mode).
- ``max_attempts`` exhaustion raises ``RuntimeError``.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, model_validator

from stitcher import Extractor
from stitcher.extractor import JsonPatchResponse


pytestmark = pytest.mark.asyncio


class Person(BaseModel):
    name: str
    age: int

    @model_validator(mode="after")
    def _positive(self):
        if self.age < 0:
            raise ValueError("age must be non-negative")
        return self


async def test_aupdate_first_turn_skips_initial_extract(fake_llm):
    """Even when ``existing`` validates cleanly, aupdate runs a patch turn —
    it must never call the initial-extract runnable."""
    fake_llm.set_scripts(
        initial=[],  # initial-extract path must NOT be exercised
        patch=[JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": 31}])],
    )
    extractor = Extractor(fake_llm, Person)

    result = await extractor.aupdate(
        existing={"name": "Alice", "age": 30},
        messages=[],
    )

    assert result.value == Person(name="Alice", age=31)
    assert result.attempts == 1
    assert result.was_re_extracted is False
    assert fake_llm.initial_runnable.calls == []
    assert len(fake_llm.patch_runnable.calls) == 1


async def test_aupdate_first_turn_prompt_has_no_repair_prefix(fake_llm):
    """The first-turn patch prompt presents the prior with the *how* (JSON
    Patch instructions) but no *what* (no validator-failure framing)."""
    fake_llm.set_scripts(
        initial=[],
        patch=[JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": 25}])],
    )
    extractor = Extractor(fake_llm, Person)

    await extractor.aupdate(
        existing={"name": "Bob", "age": 30},
        messages=[],
    )

    # The HumanMessage carrying the patch prompt is the last message in the
    # sent input — assert it carries the target schema and the prior, but
    # NOT the repair prefix (no validation failure on aupdate's first turn).
    sent_input = fake_llm.patch_runnable.calls[0]["input"]
    patch_prompt = sent_input[-1].content
    assert "<schema>" in patch_prompt
    assert "<previous>" in patch_prompt
    assert "failed validation" not in patch_prompt
    assert "<errors>" not in patch_prompt


async def test_aupdate_retry_uses_repair_prefix(fake_llm):
    """When the first patch produces an invalid object, the *next* turn's
    prompt prepends the validator-failure prefix."""
    fake_llm.set_scripts(
        initial=[],
        patch=[
            # First patch: still invalid (age -2 still < 0)
            JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": -2}]),
            # Second patch: valid
            JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": 18}]),
        ],
    )
    extractor = Extractor(fake_llm, Person, max_attempts=5)

    result = await extractor.aupdate(
        existing={"name": "Carol", "age": 30},
        messages=[],
    )
    assert result.value.age == 18
    assert result.attempts == 2

    # First turn: no repair prefix (user-driven).
    first_prompt = fake_llm.patch_runnable.calls[0]["input"][-1].content
    assert "failed validation" not in first_prompt

    # Second turn: repair prefix present (validator-driven retry).
    second_prompt = fake_llm.patch_runnable.calls[1]["input"][-1].content
    assert "Your previous JSON output failed validation" in second_prompt
    assert "<errors>" in second_prompt


async def test_aupdate_accepts_pydantic_instance(fake_llm):
    """``existing`` can be a Pydantic instance; it gets ``model_dump``'d."""
    fake_llm.set_scripts(
        initial=[],
        patch=[JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": 50}])],
    )
    extractor = Extractor(fake_llm, Person)

    result = await extractor.aupdate(
        existing=Person(name="Dave", age=40),
        messages=[],
    )
    assert result.value == Person(name="Dave", age=50)


async def test_aupdate_run_name_threads_to_every_patch_turn(fake_llm):
    """Every patch call uses ``f"{run_name}.patch"``; the initial-extract
    runnable is not touched (no initial-extract name to assert)."""
    fake_llm.set_scripts(
        initial=[],
        patch=[
            JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": -1}]),  # still fails
            JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": 7}]),
        ],
    )
    extractor = Extractor(fake_llm, Person, max_attempts=5)

    await extractor.aupdate(
        existing={"name": "Eve", "age": 30},
        messages=[],
        run_name="reconciliation_batch_3",
    )

    assert fake_llm.initial_runnable.calls == []
    assert len(fake_llm.patch_runnable.calls) == 2
    for call in fake_llm.patch_runnable.calls:
        assert call["config"]["run_name"] == "patch"


async def test_aupdate_run_name_default_when_unset(fake_llm):
    """Without ``run_name``, patch calls fall back to the fixed child name ``patch``."""
    fake_llm.set_scripts(
        initial=[],
        patch=[JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": 22}])],
    )
    extractor = Extractor(fake_llm, Person)

    await extractor.aupdate(existing={"name": "Frank", "age": 30}, messages=[])

    assert fake_llm.patch_runnable.calls[0]["config"]["run_name"] == "patch"


async def test_aupdate_exhausts_attempts_and_raises(fake_llm):
    """If every patch keeps the object invalid, raise after max_attempts."""
    fake_llm.set_scripts(
        initial=[],
        patch=[
            # Three patches, all leaving age negative → still invalid
            JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": -10}]),
            JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": -20}]),
            JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": -30}]),
        ],
    )
    extractor = Extractor(fake_llm, Person, max_attempts=3)

    with pytest.raises(RuntimeError, match="exhausted 3 attempts"):
        await extractor.aupdate(existing={"name": "Gina", "age": 30}, messages=[])

    assert len(fake_llm.patch_runnable.calls) == 3


async def test_aupdate_first_turn_prompt_embeds_target_schema(fake_llm):
    """The patch prompt embeds the target schema so the model can reason
    about what shape to patch toward — needed because on patch turns the
    structured-output binding is JsonPatchResponse (not the user's schema)."""
    fake_llm.set_scripts(
        initial=[],
        patch=[JsonPatchResponse(reasoning="(test)", operations=[{"op": "replace", "path": "/age", "value": 25}])],
    )
    extractor = Extractor(fake_llm, Person)

    await extractor.aupdate(existing={"name": "Hank", "age": 30}, messages=[])

    patch_prompt = fake_llm.patch_runnable.calls[0]["input"][-1].content
    # Schema content present (Pydantic emits the field names verbatim)
    assert '"name"' in patch_prompt
    assert '"age"' in patch_prompt
    # Wrapped in the <schema> block, not just leaked elsewhere
    schema_start = patch_prompt.find("<schema>")
    schema_end = patch_prompt.find("</schema>")
    assert schema_start < schema_end
    assert '"name"' in patch_prompt[schema_start:schema_end]
