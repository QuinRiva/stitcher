"""Shared pytest fixtures and the FakeLLM used across the suite.

These tests do not call any real LLM. ``FakeLLM`` is a hand-rolled
``BaseChatModel`` subclass that returns scripted responses, lets each test
assert on the exact ``config`` (i.e. ``run_name``, ``callbacks``) that
stitcher threads through, and counts how many times each underlying call
was made.

There is no fixture for an Extractor instance because tests typically need
to override ``with_structured_output`` per-test (different scripted outputs
for the initial extract vs the patch turn).
"""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.config import ensure_config
from pydantic import BaseModel


class _ScriptedRunnable(Runnable):
    """A Runnable that returns scripted outputs in order, recording each call's
    config so tests can assert what stitcher passed.

    Each scripted output is either:
    - a dict (returned as-is — simulates the JSON-mode initial extract path
      where ``with_structured_output(schema=dict)`` returns a parsed dict), or
    - a Pydantic ``BaseModel`` instance (returned as-is — simulates the patch
      path where ``with_structured_output(schema=JsonPatchResponse)`` returns
      a ``JsonPatchResponse``).

    If the script runs out, ``ainvoke`` raises ``AssertionError`` so an
    over-eager extractor doesn't silently re-use the last output.
    """

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def invoke(self, input, config=None, **kwargs):  # type: ignore[override]
        raise AssertionError("stitcher is async-only; sync invoke shouldn't be used")

    async def ainvoke(self, input, config: RunnableConfig | None = None, **kwargs):  # type: ignore[override]
        if not self._script:
            raise AssertionError("scripted runnable exhausted; extractor called more times than expected")
        out = self._script.pop(0)
        # ensure_config merges the explicit config arg with the LangChain
        # context vars (parent run's callbacks, tags, metadata, etc.). Real
        # Runnables do this implicitly via their default ainvoke; we have
        # to do it explicitly because _ScriptedRunnable's ainvoke is custom.
        # Without this the fake would only see what stitcher passed
        # explicitly, missing context-inherited fields and breaking tests
        # that verify the parent-runnable wrapping does its job.
        merged = ensure_config(config)
        self.calls.append({"input": input, "config": dict(merged)})
        return out


class FakeLLM(BaseChatModel):
    """A minimal BaseChatModel whose ``with_structured_output`` returns
    ``_ScriptedRunnable`` instances controlled per-test.

    Two scripts are tracked separately, mirroring the two ``with_structured_output``
    calls stitcher makes in ``Extractor.__init__``:

    - ``initial_script`` \u2192 used by the ``self._initial_llm`` runnable
    - ``patch_script`` \u2192 used by the ``self._patch_llm`` runnable

    Tests populate them via ``llm.set_scripts(initial=[...], patch=[...])``
    *before* constructing the Extractor, since stitcher captures the
    structured-output runnables eagerly in ``__init__``.
    """

    initial_runnable: _ScriptedRunnable | None = None
    patch_runnable: _ScriptedRunnable | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        # Required abstract method; we never use direct (non-structured) calls.
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

    def with_structured_output(self, schema, *, method=None, **kwargs):  # type: ignore[override]
        # Distinguish the two stitcher call sites by schema shape.
        # Initial: ``schema`` is a dict (a JSON Schema). Patch: ``schema`` is
        # the ``JsonPatchResponse`` Pydantic class.
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            assert self.patch_runnable is not None, "test forgot to set patch script"
            return self.patch_runnable
        else:
            assert self.initial_runnable is not None, "test forgot to set initial script"
            return self.initial_runnable

    def set_scripts(self, *, initial: list[Any] | None = None, patch: list[Any] | None = None) -> None:
        self.initial_runnable = _ScriptedRunnable(initial or [])
        self.patch_runnable = _ScriptedRunnable(patch or [])


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


class CapturingCallback(BaseCallbackHandler):
    """Records every chain/LLM start so tests can assert on the parent/child
    run hierarchy stitcher establishes via its RunnableLambda wrapper.

    Each event is a dict with ``name``, ``run_id``, ``parent_run_id`` so tests
    can verify e.g. ``parent.name == 'my-pipeline'`` and ``child.parent_run_id
    == parent.run_id``.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def _record(self, kind: str, *, run_id, parent_run_id=None, name=None, **_kwargs) -> None:
        self.events.append(
            {"kind": kind, "name": name, "run_id": run_id, "parent_run_id": parent_run_id}
        )

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, name=None, **kwargs):
        self._record("chain", run_id=run_id, parent_run_id=parent_run_id, name=name)

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, name=None, **kwargs):
        self._record("llm", run_id=run_id, parent_run_id=parent_run_id, name=name)


@pytest.fixture
def capturing_callback() -> CapturingCallback:
    return CapturingCallback()
