"""``Result.metadata`` aggregates token usage and wall time.

Verifies:

- ``TokenUsage`` field semantics: ``cached_input_tokens`` is a subset of
  ``input_tokens``; ``reasoning_tokens`` is a subset of the underlying
  ``output_tokens`` and ``output_payload_tokens = output_tokens -
  reasoning_tokens``.
- ``Metadata.initial`` tracks only the first happy-path AIMessage \u2014 the
  call whose output seeded the eventually-returned value.
- ``Metadata.total`` sums every AIMessage that hit ``raw_messages``,
  including discarded ones (catastrophic re-extract, failed patch apply).
- ``Metadata.duration_seconds`` is non-negative and broadly reflects the
  loop time.
- Auto-wrapped scripts (no real ``usage_metadata``) produce all-zero
  ``TokenUsage`` without raising \u2014 the ``usage_metadata is None`` path
  is exercised by every existing test, this just locks it in.

Tests script the include_raw envelope explicitly when they need to attach
``usage_metadata`` to drive the aggregation.
"""
from __future__ import annotations

import time

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, model_validator

from stitcher import Extractor, Metadata, TokenUsage
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


def _envelope(
    parsed,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    reasoning: int = 0,
) -> dict:
    """Build a successful include_raw envelope with the requested usage_metadata.

    The ``raw`` AIMessage carries ``usage_metadata`` shaped per LangChain's
    convention: top-level ``input_tokens``/``output_tokens``/``total_tokens``
    plus optional ``input_token_details.cache_read`` and
    ``output_token_details.reasoning`` subset breakdowns.
    """
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cache_read:
        usage["input_token_details"] = {"cache_read": cache_read}
    if reasoning:
        usage["output_token_details"] = {"reasoning": reasoning}
    content = (
        parsed.model_dump_json()
        if isinstance(parsed, BaseModel)
        else __import__("json").dumps(parsed)
    )
    return {
        "raw": AIMessage(content=content, usage_metadata=usage),
        "parsed": parsed,
        "parsing_error": None,
    }


async def test_metadata_zero_when_no_usage_attached(fake_llm):
    """Auto-wrapped scripts (the common test path) yield all-zero TokenUsage."""
    fake_llm.set_scripts(
        initial=[{"name": "Alice", "age": 30}],
        patch=[],
    )
    extractor = Extractor(fake_llm, Person)
    result = await extractor.ainvoke([])

    assert isinstance(result.metadata, Metadata)
    assert isinstance(result.metadata.initial, TokenUsage)
    assert result.metadata.initial == TokenUsage(
        input_tokens=0, cached_input_tokens=0,
        reasoning_tokens=0, output_payload_tokens=0,
    )
    assert result.metadata.total == result.metadata.initial


async def test_happy_path_single_extract(fake_llm):
    """attempts==1: initial == total, and output_payload_tokens IS the size
    of the final value as the model emitted it (the documented shortcut)."""
    fake_llm.set_scripts(
        initial=[_envelope({"name": "Alice", "age": 30},
                           input_tokens=120, output_tokens=18)],
        patch=[],
    )
    extractor = Extractor(fake_llm, Person)
    result = await extractor.ainvoke([])

    assert result.attempts == 1
    assert result.metadata.initial.input_tokens == 120
    assert result.metadata.initial.output_payload_tokens == 18
    assert result.metadata.initial == result.metadata.total


async def test_total_sums_initial_plus_patch(fake_llm):
    """After a successful repair, total = initial + patch turn tokens, and
    initial reflects ONLY the seed extract."""
    fake_llm.set_scripts(
        initial=[_envelope({"name": "Alice", "age": -1},
                           input_tokens=100, output_tokens=20)],
        patch=[_envelope(
            JsonPatchResponse(reasoning="(test)",
                              operations=[{"op": "replace", "path": "/age", "value": 30}]),
            input_tokens=250, output_tokens=35,
        )],
    )
    extractor = Extractor(fake_llm, Person)
    result = await extractor.ainvoke([])

    assert result.attempts == 2
    assert result.metadata.initial.input_tokens == 100
    assert result.metadata.initial.output_payload_tokens == 20
    assert result.metadata.total.input_tokens == 350
    assert result.metadata.total.output_payload_tokens == 55


async def test_initial_skips_discarded_extract_on_re_extract(fake_llm):
    """Catastrophic re-extract path: the discarded first extract is in
    ``total`` (and in ``raw_messages``) but NOT in ``initial`` \u2014 ``initial``
    is the re-extract whose output actually seeded ``Result.value``."""
    fake_llm.set_scripts(
        initial=[
            # discarded by parse-error path: 80 input / 5 output
            {"raw": AIMessage(content="garbage",
                              usage_metadata={"input_tokens": 80, "output_tokens": 5,
                                              "total_tokens": 85}),
             "parsed": None,
             "parsing_error": ValueError("nope")},
            # the successful re-extract that seeds the final value
            _envelope({"name": "Alice", "age": 30},
                      input_tokens=110, output_tokens=22),
        ],
        patch=[],
    )
    extractor = Extractor(fake_llm, Person, max_attempts=5)
    result = await extractor.ainvoke([])

    assert result.was_re_extracted is True
    assert result.attempts == 2
    # initial = ONLY the successful re-extract
    assert result.metadata.initial.input_tokens == 110
    assert result.metadata.initial.output_payload_tokens == 22
    # total = both extracts (discarded + successful)
    assert result.metadata.total.input_tokens == 80 + 110
    assert result.metadata.total.output_payload_tokens == 5 + 22


async def test_failed_patch_apply_excluded_from_initial_but_in_total(fake_llm):
    """A patch whose ops fail jsonpatch.apply contributes to ``total`` (it's
    still an LLM call we paid for) but NOT to ``initial`` (only the seed
    extract is initial; patches contribute to a separate happy-path slot
    that initial doesn't read from)."""
    fake_llm.set_scripts(
        initial=[_envelope({"name": "Alice", "age": -1},
                           input_tokens=100, output_tokens=20)],
        patch=[
            # First patch: bad pointer that jsonpatch.apply will reject
            _envelope(
                JsonPatchResponse(reasoning="(bad)",
                                  operations=[{"op": "replace", "path": "/nonexistent", "value": 1}]),
                input_tokens=200, output_tokens=30,
            ),
            # Second patch: correct repair
            _envelope(
                JsonPatchResponse(reasoning="(fix)",
                                  operations=[{"op": "replace", "path": "/age", "value": 30}]),
                input_tokens=250, output_tokens=35,
            ),
        ],
    )
    extractor = Extractor(fake_llm, Person, max_attempts=5)
    result = await extractor.ainvoke([])

    assert result.attempts == 3
    # initial still tracks the seed extract only
    assert result.metadata.initial.input_tokens == 100
    # total includes both patches (failed and successful)
    assert result.metadata.total.input_tokens == 100 + 200 + 250
    assert result.metadata.total.output_payload_tokens == 20 + 30 + 35


async def test_cached_and_reasoning_are_subsets(fake_llm):
    """cached_input_tokens is a subset of input_tokens; reasoning_tokens
    is a subset of the underlying output_tokens, and output_payload_tokens
    excludes reasoning."""
    fake_llm.set_scripts(
        initial=[_envelope({"name": "Alice", "age": 30},
                           input_tokens=500, output_tokens=250,
                           cache_read=400, reasoning=200)],
        patch=[],
    )
    extractor = Extractor(fake_llm, Person)
    result = await extractor.ainvoke([])

    m = result.metadata.initial
    assert m.input_tokens == 500
    assert m.cached_input_tokens == 400        # subset of input
    assert m.reasoning_tokens == 200           # subset of output
    assert m.output_payload_tokens == 50       # 250 output - 200 reasoning
    # "Uncached input" derivation works for the caller
    assert m.input_tokens - m.cached_input_tokens == 100


async def test_duration_seconds_non_negative_and_plausible(fake_llm):
    """duration_seconds is measured around the patch loop. We only assert
    non-negativity and an upper bound to catch obviously-wrong clocks; the
    scripted loop should complete in milliseconds."""
    fake_llm.set_scripts(
        initial=[{"name": "Alice", "age": 30}],
        patch=[],
    )
    extractor = Extractor(fake_llm, Person)
    t_before = time.monotonic()
    result = await extractor.ainvoke([])
    elapsed = time.monotonic() - t_before

    assert result.metadata.duration_seconds >= 0
    assert result.metadata.duration_seconds <= elapsed + 0.001  # small slack for clock jitter


async def test_aupdate_initial_is_first_applied_patch(fake_llm):
    """Under aupdate, the first LLM call is a patch turn. ``initial`` tracks
    that first applied patch (NOT the existing seed, which had no LLM cost).
    The documented warning: ``initial.output_payload_tokens`` here is the
    size of the ``{reasoning, operations}`` envelope, not the size of
    ``Result.value``."""
    fake_llm.set_scripts(
        initial=[],
        patch=[_envelope(
            JsonPatchResponse(reasoning="(test)",
                              operations=[{"op": "replace", "path": "/age", "value": 31}]),
            input_tokens=180, output_tokens=40,
        )],
    )
    extractor = Extractor(fake_llm, Person)
    result = await extractor.aupdate(
        existing=Person(name="Alice", age=30),
        messages=[],
    )

    assert result.attempts == 1
    assert result.metadata.initial.input_tokens == 180
    assert result.metadata.initial.output_payload_tokens == 40
    assert result.metadata.initial == result.metadata.total
