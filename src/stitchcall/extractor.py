"""Extractor — single-schema LLM extraction with native JSON-mode + JSON-Patch repair.

Pipeline per ``ainvoke``:

    1. Initial extract via the model's native JSON mode
       (``with_structured_output(method="json_schema")``). Returns a parsed dict.
    2. ``schema.model_validate(dict, context=validation_context)``.
    3. On validation failure:
       - sum per-entry weight (``AggregatedValidationError(count=N)`` -> N, else 1).
       - if cumulative weight > ``max_validation_error_weight``: re-extract from
         the original system+user, increment ``attempts``, restart the loop.
       - else: ask the model for a JSON Patch (RFC 6902) on a plain
         ``HumanMessage`` chain, apply via ``jsonpatch``, re-validate.
    4. Bounded by ``max_attempts``.

Design rationale: see ``docs/intent.md``.
"""
from __future__ import annotations

import json
from typing import Any, NamedTuple

import jsonpatch
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from stitchcall.exceptions import AggregatedValidationError


class JsonPatchResponse(BaseModel):
    """Wrapper for the model's JSON Patch repair response."""

    operations: list[dict] = Field(
        ...,
        description=(
            "An RFC 6902 JSON Patch — an array of {op, path, value} operations "
            "that, when applied to the previous JSON output, will produce a "
            "JSON object that passes validation."
        ),
    )


class Result(NamedTuple):
    """Result of a successful ``Extractor.ainvoke`` call."""
    value: BaseModel
    attempts: int
    was_re_extracted: bool
    raw_messages: list


_PATCH_PROMPT_TEMPLATE = (
    "Your previous JSON output failed validation.\n\n"
    "<errors>\n{errors}\n</errors>\n\n"
    "The previous JSON output was:\n\n"
    "<previous>\n{previous}\n</previous>\n\n"
    "Return a JSON Patch (RFC 6902) — an `operations` array of "
    "{{op, path, value}} entries — that, when applied to the previous "
    "output, produces a JSON object that passes validation. "
    "Common ops: 'add', 'replace', 'remove'. "
    "Paths are JSON Pointers (e.g. '/items/0' to address the first list item, "
    "'/items/-' to append). Return ONLY the patch operations; do not echo the "
    "full object."
)


class Extractor:
    """Single-schema, JSON-mode + JSON-Patch-repair extractor.

    Args:
        llm: any LangChain ``BaseChatModel`` whose
            ``with_structured_output(method="json_schema")`` is supported.
        schema: the Pydantic v2 model to extract.
        max_attempts: total attempts (initial + patches + re-extracts).
        max_validation_error_weight: when summed validation error weight on
            a single pass exceeds this, the patch loop is abandoned and a
            fresh extract is performed instead. ``None`` disables the
            catastrophic-re-extract path entirely.

    Example:
        >>> from pydantic import BaseModel
        >>> from langchain_google_genai import ChatGoogleGenerativeAI
        >>> from stitchcall import Extractor
        >>>
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>>
        >>> llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
        >>> extractor = Extractor(llm, Person)
        >>> result = await extractor.ainvoke([
        ...     {"role": "user", "content": "Extract the person from: Alice, 30"}
        ... ])
        >>> result.value
        Person(name='Alice', age=30)
    """

    def __init__(
        self,
        llm: BaseChatModel,
        schema: type[BaseModel],
        *,
        max_attempts: int = 5,
        max_validation_error_weight: int | None = 40,
    ) -> None:
        self.llm = llm
        self.schema = schema
        self.max_attempts = max_attempts
        self.max_validation_error_weight = max_validation_error_weight
        # Bind the user's schema as a JSON-schema dict so the structured-output
        # endpoint returns a parsed dict (which we re-validate ourselves with
        # the user-supplied validation context — model_validate via langchain's
        # built-in path skips that context).
        self._initial_llm = llm.with_structured_output(
            schema=schema.model_json_schema(),
            method="json_schema",
        )
        self._patch_llm = llm.with_structured_output(
            schema=JsonPatchResponse,
            method="json_schema",
        )

    async def ainvoke(
        self,
        messages: list,
        *,
        validation_context: dict | None = None,
        callbacks: list | None = None,
    ) -> Result:
        """Extract a validated instance of ``self.schema`` from ``messages``.

        Args:
            messages: conversation history to send to the model. Accepts either
                LangChain ``BaseMessage`` instances or plain ``{role, content}``
                dicts (with role in ``system | human | user | ai | assistant``).
            validation_context: passed verbatim to ``schema.model_validate(...,
                context=...)`` on every validation pass. Use this for any
                cross-field invariants the schema's validators need.
            callbacks: LangChain callback handlers, threaded into every LLM call
                (initial extract, every patch turn).

        Returns:
            ``Result(value, attempts, was_re_extracted, raw_messages)``.

        Raises:
            RuntimeError: if ``max_attempts`` is exhausted without a valid
                extraction.
        """
        original_messages = _coerce_messages(messages)
        attempts = 0
        was_re_extracted = False
        raw_messages: list = list(original_messages)

        prev_dict: dict | None = None
        last_validation_error: BaseException | None = None

        while attempts < self.max_attempts:
            attempts += 1

            if prev_dict is None:
                prev_dict = await self._initial_extract(original_messages, callbacks=callbacks)
                raw_messages.append(AIMessage(content=json.dumps(prev_dict, default=str)))
            else:
                patch_msg = HumanMessage(
                    content=_build_patch_prompt(prev_dict, last_validation_error)
                )
                patch_history = list(original_messages) + [
                    AIMessage(content=json.dumps(prev_dict, default=str)),
                    patch_msg,
                ]
                raw_messages.append(patch_msg)
                patch_resp: JsonPatchResponse = await self._patch_llm.ainvoke(
                    patch_history,
                    config={"callbacks": callbacks or [], "run_name": "stitchcall_patch"},
                )
                ops = patch_resp.operations
                raw_messages.append(
                    AIMessage(content=json.dumps({"operations": ops}, default=str))
                )
                try:
                    prev_dict = jsonpatch.JsonPatch(ops).apply(prev_dict)
                except (jsonpatch.JsonPatchException, jsonpatch.JsonPointerException) as e:
                    # Treat patch-application failure as a validation failure for
                    # the next loop turn — feed the error back to the model.
                    last_validation_error = ValueError(
                        f"Your JSON Patch could not be applied: {type(e).__name__}: {e}. "
                        "Re-issue a corrected patch against the previous JSON output."
                    )
                    continue

            try:
                value = self.schema.model_validate(prev_dict, context=validation_context)
                return Result(
                    value=value,
                    attempts=attempts,
                    was_re_extracted=was_re_extracted,
                    raw_messages=raw_messages,
                )
            except ValidationError as e:
                last_validation_error = e
                weight = _error_weight(e)
                if (
                    self.max_validation_error_weight is not None
                    and weight > self.max_validation_error_weight
                ):
                    was_re_extracted = True
                    prev_dict = None
                continue
            except AggregatedValidationError as e:
                last_validation_error = e
                if (
                    self.max_validation_error_weight is not None
                    and e.count > self.max_validation_error_weight
                ):
                    was_re_extracted = True
                    prev_dict = None
                continue

        raise RuntimeError(
            f"Extractor exhausted {self.max_attempts} attempts. "
            f"Last validation error: {last_validation_error!r}"
        )

    async def _initial_extract(
        self, messages: list[BaseMessage], *, callbacks: list | None
    ) -> dict:
        out = await self._initial_llm.ainvoke(
            messages,
            config={"callbacks": callbacks or [], "run_name": "stitchcall_initial"},
        )
        if isinstance(out, BaseModel):
            return out.model_dump()
        if not isinstance(out, dict):
            raise TypeError(
                f"with_structured_output returned unexpected type: {type(out).__name__}"
            )
        return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _coerce_messages(messages: list) -> list[BaseMessage]:
    """Accept either dict-form or BaseMessage-form, return BaseMessages."""
    out: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, BaseMessage):
            out.append(m)
        elif isinstance(m, dict):
            role = m.get("role") or m.get("type")
            content = m.get("content", "")
            if role == "system":
                out.append(SystemMessage(content=content))
            elif role in ("human", "user"):
                out.append(HumanMessage(content=content))
            elif role in ("ai", "assistant"):
                out.append(AIMessage(content=content))
            else:
                raise ValueError(f"Unknown message role: {role!r}")
        else:
            raise TypeError(f"Unsupported message type: {type(m).__name__}")
    return out


def _error_weight(e: ValidationError) -> int:
    """Sum per-entry weights. AggregatedValidationError(count=N) -> N, else 1."""
    total = 0
    for err in e.errors():
        ctx = err.get("ctx") or {}
        cause = ctx.get("error") if isinstance(ctx, dict) else None
        if isinstance(cause, AggregatedValidationError):
            total += cause.count
        else:
            total += 1
    return total or 1


def _build_patch_prompt(prev_dict: dict, error: BaseException | None) -> str:
    if isinstance(error, ValidationError):
        try:
            errors_text = json.dumps(
                [
                    {
                        "loc": list(err.get("loc", [])),
                        "msg": err.get("msg", ""),
                        "type": err.get("type", ""),
                    }
                    for err in error.errors()
                ],
                indent=2,
                default=str,
            )
        except Exception:
            errors_text = str(error)
    else:
        errors_text = str(error) if error else "(no error captured)"
    previous_text = json.dumps(prev_dict, indent=2, default=str)
    return _PATCH_PROMPT_TEMPLATE.format(errors=errors_text, previous=previous_text)
