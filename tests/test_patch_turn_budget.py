"""Behaviour 2 — the apply-retry budget is separate from the validation budget.

A patch that fails to APPLY is retried within a single outer turn with its own
cap ``_MAX_APPLY_ATTEMPTS`` and does NOT consume a validation attempt. Only
inner-cap exhaustion, or a protocol/parse error, hands an error back to the
outer loop (consuming exactly one attempt). Total patch generations are bounded
by ``max_attempts * _MAX_APPLY_ATTEMPTS``.

The fake here scripts the patch LLM as real ``AIMessage``s that flow through
stitcher's real ``_invoke_patch`` -> ``_parse_content`` seam (never a
pre-parsed envelope) — the same seam production uses.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from stitcher import Extractor as PublicExtractor
from stitcher.extractor import Extractor, JsonPatchResponse, _MAX_APPLY_ATTEMPTS


class _FakePatchLLM:
    """Queued patch responses as real AIMessages; records every call's history.

    A truthy ``parsing_errors`` entry queues non-JSON content so the real parse
    path raises a ``parsing_error`` (rather than faking the envelope).
    """

    def __init__(self, ops_sequence, parsing_errors=None):
        self._ops = list(ops_sequence)
        self._parse = list(parsing_errors or [None] * len(ops_sequence))
        self.calls: list = []

    async def ainvoke(self, history, config=None):
        self.calls.append(history)
        ops = self._ops.pop(0)
        parse_error = self._parse.pop(0)
        if parse_error is not None:
            return AIMessage(content="this is not valid json")
        return AIMessage(content=json.dumps({"reasoning": "fix", "operations": ops}))


def _make_extractor(fake_llm) -> Extractor:
    """An Extractor with only the attrs ``_run_patch_turn`` touches."""
    ext = object.__new__(Extractor)
    ext._patch_llm = fake_llm
    ext._schema_json = {"type": "object"}
    return ext


def _run_turn(ext, prev_dict, patch_prompt="<initial patch prompt>"):
    raw_messages: list = []
    result = asyncio.run(
        ext._run_patch_turn(
            prev_dict=prev_dict,
            patch_prompt=patch_prompt,
            original_messages=[HumanMessage(content="orig")],
            patch_config={},
            raw_messages=raw_messages,
        )
    )
    return result, raw_messages


def test_apply_retry_then_success_is_one_outer_turn():
    prev = {"scope": "NONE_IN_SCOPE"}
    bad = [{"op": "add", "path": "/scope/-", "value": {"x": 1}}]
    good = [{"op": "replace", "path": "/scope", "value": [{"x": 1}]}]
    fake = _FakePatchLLM([bad, good])
    (new_dict, error, applied), raw = _run_turn(_make_extractor(fake), prev)

    # The bad patch was retried WITHIN this single outer turn and then
    # succeeded — the outer loop (validation budget) is untouched.
    assert error is None
    assert new_dict == {"scope": [{"x": 1}]}
    assert applied is not None
    assert len(fake.calls) == 2
    # The retry prompt carried the shape-enriched apply feedback.
    retry_prompt = fake.calls[1][-1].content
    assert "could not be applied" in retry_prompt
    assert "NONE_IN_SCOPE" in retry_prompt


def test_apply_failures_capped_and_do_not_loop_forever():
    prev = {"scope": "NONE_IN_SCOPE"}
    bad = [{"op": "add", "path": "/scope/-", "value": {"x": 1}}]
    # More bad patches than the cap: the turn must give up, not loop.
    fake = _FakePatchLLM([bad] * 10)
    (new_dict, error, applied), raw = _run_turn(_make_extractor(fake), prev)

    assert applied is None
    assert new_dict is prev  # unchanged
    assert isinstance(error, ValueError)
    assert "could not be applied" in str(error)
    # Exactly the apply cap of generations — never the full 10.
    assert len(fake.calls) == _MAX_APPLY_ATTEMPTS


def test_parse_error_propagates_without_apply_retry():
    prev = {"scope": "NONE_IN_SCOPE"}
    fake = _FakePatchLLM([None], parsing_errors=[ValueError("not json")])
    (new_dict, error, applied), raw = _run_turn(_make_extractor(fake), prev)

    assert applied is None
    assert new_dict is prev
    assert "reasoning, operations" in str(error)
    assert len(fake.calls) == 1  # no apply retry for a protocol error


def test_first_patch_applies_single_call():
    prev = {"scope": "NONE_IN_SCOPE"}
    good = [{"op": "replace", "path": "/scope", "value": []}]
    fake = _FakePatchLLM([good])
    (new_dict, error, applied), raw = _run_turn(_make_extractor(fake), prev)

    assert error is None
    assert new_dict == {"scope": []}
    assert len(fake.calls) == 1


class _Doc(BaseModel):
    age: int


@pytest.mark.asyncio
async def test_termination_bounded_by_max_attempts_times_apply_cap(fake_llm):
    """End-to-end: a patch that never applies exhausts the inner cap each outer
    attempt (consuming one attempt), so the loop terminates after exactly
    ``max_attempts`` outer attempts and ``max_attempts * _MAX_APPLY_ATTEMPTS``
    patch generations, then raises."""
    max_attempts = 2
    bad = JsonPatchResponse(
        reasoning="(t)",
        operations=[{"op": "replace", "path": "/nonexistent", "value": 1}],
    )
    fake_llm.set_scripts(
        initial=[],
        patch=[bad] * (max_attempts * _MAX_APPLY_ATTEMPTS),
    )
    ext = PublicExtractor(fake_llm, _Doc, max_attempts=max_attempts)

    with pytest.raises(RuntimeError, match=f"exhausted {max_attempts} attempts"):
        await ext.aupdate(existing={"age": 1}, messages=[])

    # Every scripted patch was consumed: no more, no fewer.
    assert len(fake_llm.patch_runnable.calls) == max_attempts * _MAX_APPLY_ATTEMPTS
