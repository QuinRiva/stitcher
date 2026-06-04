"""Thinking-model responses arrive as a list of content blocks, not a string.

Gemini 3 (and Claude extended thinking, OpenAI o-series via LangChain) return
``AIMessage.content`` as ``[{"type": "thinking", ...}, {"type": "text",
"text": "<the JSON>"}]``. ``_content_text`` must pull the JSON out of the text
block and ignore the thinking block.

This is a regression guard for a real shipped bug: ``str(content)`` on the
list produced a Python repr, every ``json.loads`` failed, and the extractor
silently re-extracted until it exhausted ``max_attempts`` (looping forever on
"initial" with zero patch turns). High blast radius, invisible without a
trace \u2014 worth the test.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from stitcher import Extractor
from stitcher.extractor import _content_text, _parse_content


class Person(BaseModel):
    name: str
    age: int


def test_content_text_extracts_text_block_ignoring_thinking():
    msg = AIMessage(content=[
        {"type": "thinking", "thinking": "secret reasoning, not JSON"},
        {"type": "text", "text": '{"name": "Alice", "age": 30}', "extras": {}},
    ])
    assert _content_text(msg.content) == '{"name": "Alice", "age": 30}'


def test_parse_content_handles_thinking_block_list():
    msg = AIMessage(content=[
        {"type": "thinking", "thinking": "..."},
        {"type": "text", "text": '{"name": "Bob", "age": 25}'},
    ])
    env = _parse_content(msg, model=None)
    assert env["parsing_error"] is None
    assert env["parsed"] == {"name": "Bob", "age": 25}


@pytest.mark.asyncio
async def test_ainvoke_does_not_loop_on_thinking_content(fake_llm):
    """End-to-end: a thinking-block initial response validates on attempt 1 \u2014
    no re-extract loop, no patch turn."""
    fake_llm.set_scripts(
        initial=[AIMessage(content=[
            {"type": "thinking", "thinking": "classifying the person"},
            {"type": "text", "text": '{"name": "Carol", "age": 40}'},
        ])],
        patch=[],  # must NOT be reached
    )
    extractor = Extractor(fake_llm, Person)

    result = await extractor.ainvoke([])

    assert result.value == Person(name="Carol", age=40)
    assert result.attempts == 1
    assert fake_llm.patch_runnable.calls == []
