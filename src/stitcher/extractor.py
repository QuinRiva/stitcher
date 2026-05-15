"""Extractor — single-schema LLM extraction with native JSON-mode + JSON-Patch repair.

Two public operations, both built on the same JSON-Patch repair loop:

``ainvoke(messages, ...)`` — produce a fresh validated instance:

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

``aupdate(existing, messages, ...)`` — transform an existing instance per user
intent (replaces trustcall's ``existing=`` flow for the single-schema case):

    1. Seed ``prev_dict`` from ``existing``; skip the initial extract.
    2. Patch loop runs as in ``ainvoke``. The first turn carries no validator
       header — the user's update intent in ``messages`` directs *what* to
       patch; the prompt only specifies *how* (JSON Patch / JSON Pointer
       format). Subsequent turns (if validation fails) prepend the validation
       header exactly as in ``ainvoke``.
    3. No catastrophic-re-extract path (there is no fresh-extract fallback);
       bounded by ``max_attempts``.

Design rationale: see ``docs/intent.md``.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable
from typing import Any, Callable, NamedTuple, TypeAlias

import jsonpatch
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field, ValidationError

from stitcher.exceptions import AggregatedValidationError


ValidationContext: TypeAlias = dict[str, Any]
Callbacks: TypeAlias = list[BaseCallbackHandler]


class AttemptInfo(NamedTuple):
    """Per-attempt observability payload, fired by ``Extractor.on_attempt``.

    Differs from trustcall's ``AttemptInfo`` by omitting ``ai_message``:
    stitcher uses LangChain's ``with_structured_output``, which returns a
    parsed dict rather than a raw ``AIMessage`` — synthesizing a fake one
    here would lose ``response_metadata`` / ``usage_metadata`` and silently
    break callers who need that data. If you need raw response metadata
    (token counts, finish_reason, etc.), pass a ``BaseCallbackHandler`` via
    ``callbacks=`` — it fires on the underlying LLM call.

    Fields:
        attempt_number: 1 on the first validation, incrementing per attempt
            (matches the ``attempt_count`` injected into validation_context).
        parsed: the parsed dict that was validated, or ``None`` if the
            model's JSON Patch could not be applied (no validate happened).
        validation_errors: list of formatted ``loc: msg`` strings, one per
            entry in ``ValidationError.errors()``. Empty list on success.
        is_success: convenience boolean (``not validation_errors`` is
            equivalent).
    """
    attempt_number: int
    parsed: dict[str, Any] | None
    validation_errors: list[str]
    is_success: bool


# Both sync (returns None) and async (returns Awaitable[None]) callables
# are accepted; stitcher awaits if the return is a coroutine.
OnAttempt: TypeAlias = Callable[["AttemptInfo"], None | Awaitable[None]]


class JsonPatchResponse(BaseModel):
    """Wrapper for the model's JSON Patch repair response."""

    operations: list[dict[str, Any]] = Field(
        ...,
        description=(
            "An RFC 6902 JSON Patch — an array of {op, path, value} operations "
            "that, when applied to the previous JSON output, will produce a "
            "JSON object that passes validation."
        ),
    )


class Result(NamedTuple):
    """Result of a successful ``ainvoke`` or ``aupdate`` call."""
    value: BaseModel
    attempts: int
    was_re_extracted: bool
    raw_messages: list[BaseMessage]


# Body of every patch-turn prompt: state + how-to-patch instruction.
# The caller's messages own the *what* (extraction prompt for ainvoke,
# update intent for aupdate); this template owns the *how*.
_PATCH_PROMPT_TEMPLATE = (
    "<previous>\n{previous}\n</previous>\n\n"
    "Return a JSON Patch (RFC 6902) — an `operations` array of "
    "{{op, path, value}} entries — that, when applied to the previous "
    "output, produces a JSON object that passes validation. "
    "Common ops: 'add', 'replace', 'remove'. "
    "Paths are JSON Pointers (e.g. '/items/0' to address the first list item, "
    "'/items/-' to append). Return ONLY the patch operations; do not echo the "
    "full object."
)

# Prepended to the body when stitcher owns the *what* — i.e. when the
# trigger is an internal validator failure rather than a user-supplied intent.
_PATCH_REPAIR_PREFIX = (
    "Your previous JSON output failed validation.\n\n"
    "<errors>\n{errors}\n</errors>\n\n"
    "The previous JSON output was:\n\n"
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
        on_attempt: optional observability hook fired once per validation
            attempt with an ``AttemptInfo``. Sync or async callables both
            work (async ones are awaited). Fired on validation success,
            validation failure, and patch-application failure (the third
            with ``parsed=None``). Mirrors trustcall's ``on_attempt`` for
            migration; see ``AttemptInfo`` for shape differences.

    Example:
        >>> from pydantic import BaseModel
        >>> from langchain_google_genai import ChatGoogleGenerativeAI
        >>> from stitcher import Extractor
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
        on_attempt: OnAttempt | None = None,
    ) -> None:
        self.llm = llm
        self.schema = schema
        self.max_attempts = max_attempts
        self.max_validation_error_weight = max_validation_error_weight
        self._on_attempt = on_attempt
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
        messages: list[BaseMessage],
        *,
        validation_context: ValidationContext | None = None,
        callbacks: Callbacks | None = None,
        run_name: str | None = None,
    ) -> Result:
        """Extract a validated instance of ``self.schema`` from ``messages``.

        Args:
            messages: conversation history to send to the model. Accepts either
                LangChain ``BaseMessage`` instances or plain ``{role, content}``
                dicts (with role in ``system | human | user | ai | assistant``).
            validation_context: passed to ``schema.model_validate(...,
                context=...)`` on every validation pass, with one
                stitcher-supplied key merged in: ``attempt_count`` (1 on
                the first validation, 2 after the first patch, ...). User
                keys override on collision (matches trustcall's contract).
                Use this for any cross-field invariants the schema's
                validators need.
            callbacks: LangChain callback handlers, threaded into every LLM call
                (initial extract, every patch turn).
            run_name: optional prefix for internal LLM run names
                (``<prefix>.initial`` / ``<prefix>.patch``). Defaults to
                ``stitcher_initial`` / ``stitcher_patch`` when unset.

        Returns:
            ``Result(value, attempts, was_re_extracted, raw_messages)``.

        Raises:
            RuntimeError: if ``max_attempts`` is exhausted without a valid
                extraction.
        """
        return await self._patch_loop(
            original_messages=messages,
            prev_dict=None,
            run_name=run_name,
            callbacks=callbacks,
            validation_context=validation_context,
            allow_re_extract=True,
        )

    async def aupdate(
        self,
        existing: BaseModel | dict[str, Any],
        messages: list[BaseMessage],
        *,
        validation_context: ValidationContext | None = None,
        callbacks: Callbacks | None = None,
        run_name: str | None = None,
    ) -> Result:
        """Apply an update intent to an existing instance.

        ``messages`` carry the user's update request (the *what*). ``existing``
        seeds the patch loop directly; the initial JSON-mode extract is
        skipped. The first patch turn presents the prior with no validator
        header — the model is expected to read the update intent from the
        preceding messages. Subsequent turns (if validation fails) are
        validator-driven exactly as in ``ainvoke``. No catastrophic-re-extract
        path exists in update mode; ``max_attempts`` exhaustion raises.

        Always runs at least one LLM call. If the existing object is already
        valid and no update is needed, do not call ``aupdate`` — there is no
        zero-LLM-call optimisation by design (that would conflate update with
        verify-and-repair, which are different operations).

        Args:
            existing: the prior instance — either a Pydantic model instance
                (``model_dump(mode='json')`` is called internally) or a
                JSON-serialisable dict (used directly; not copied).
            messages: conversation history carrying the update intent.
            validation_context: as for ``ainvoke``.
            callbacks: as for ``ainvoke``.
            run_name: as for ``ainvoke``; only the ``.patch`` suffix is used
                (no initial extract in update mode).

        Returns:
            ``Result``; ``was_re_extracted`` is always ``False``.

        Raises:
            RuntimeError: if ``max_attempts`` is exhausted.
        """
        prev_dict: dict[str, Any] = (
            existing.model_dump(mode="json")
            if isinstance(existing, BaseModel)
            else existing
        )
        return await self._patch_loop(
            original_messages=messages,
            prev_dict=prev_dict,
            run_name=run_name,
            callbacks=callbacks,
            validation_context=validation_context,
            allow_re_extract=False,
        )

    async def _patch_loop(
        self,
        *,
        original_messages: list[BaseMessage],
        prev_dict: dict[str, Any] | None,
        run_name: str | None,
        callbacks: Callbacks | None,
        validation_context: ValidationContext | None,
        allow_re_extract: bool,
    ) -> Result:
        """Shared validate-and-patch loop for ``ainvoke`` and ``aupdate``.

        State machine on each iteration, driven by ``(prev_dict, last_error)``:

        - ``prev_dict is None`` → initial extract via JSON mode (only ever
          reached from ``ainvoke``: either the first iteration or after the
          catastrophic-re-extract path resets ``prev_dict`` to ``None``).
        - ``prev_dict`` set, ``last_error is None`` → headerless patch turn;
          the user's messages own the *what* (``aupdate``'s first turn).
        - ``prev_dict`` set, ``last_error`` set → repair patch turn; the
          validator-failure prefix is prepended to the patch prompt.

        ``allow_re_extract`` gates the catastrophic-re-extract path — only
        ``ainvoke`` enables it because only ``ainvoke`` has a fresh-extract
        path to fall back to.
        """
        initial_run_name = f"{run_name}.initial" if run_name else "stitcher_initial"
        patch_run_name = f"{run_name}.patch" if run_name else "stitcher_patch"

        raw_messages: list[BaseMessage] = list(original_messages)
        attempts = 0
        was_re_extracted = False
        last_validation_error: BaseException | None = None

        while attempts < self.max_attempts:
            attempts += 1

            if prev_dict is None:
                prev_dict = await self._initial_llm.ainvoke(
                    original_messages,
                    config={"callbacks": callbacks or [], "run_name": initial_run_name},
                )
                raw_messages.append(AIMessage(content=json.dumps(prev_dict)))
            else:
                prev_dict, patch_error = await self._run_patch_turn(
                    prev_dict=prev_dict,
                    patch_prompt=_build_patch_prompt(prev_dict, last_validation_error),
                    original_messages=original_messages,
                    patch_run_name=patch_run_name,
                    callbacks=callbacks,
                    raw_messages=raw_messages,
                )
                if patch_error is not None:
                    last_validation_error = patch_error
                    await self._fire_attempt(attempts, None, [str(patch_error)], False)
                    continue

            # Inject attempt_count so validators can implement
            # first-attempt-strict / later-lenient patterns. Matches trustcall's
            # contract: user-supplied keys win, so callers can override
            # (typically only useful for testing).
            ctx = {"attempt_count": attempts, **(validation_context or {})}
            try:
                value = self.schema.model_validate(prev_dict, context=ctx)
                await self._fire_attempt(attempts, prev_dict, [], True)
                return Result(
                    value=value,
                    attempts=attempts,
                    was_re_extracted=was_re_extracted,
                    raw_messages=raw_messages,
                )
            except ValidationError as e:
                last_validation_error = e
                await self._fire_attempt(attempts, prev_dict, _format_validation_errors(e), False)
                if (
                    allow_re_extract
                    and self.max_validation_error_weight is not None
                    and _error_weight(e) > self.max_validation_error_weight
                ):
                    was_re_extracted = True
                    prev_dict = None
                continue
            except AggregatedValidationError as e:
                last_validation_error = e
                await self._fire_attempt(attempts, prev_dict, [str(e)], False)
                if (
                    allow_re_extract
                    and self.max_validation_error_weight is not None
                    and e.count > self.max_validation_error_weight
                ):
                    was_re_extracted = True
                    prev_dict = None
                continue

        raise RuntimeError(
            f"Extractor exhausted {self.max_attempts} attempts. "
            f"Last validation error: {last_validation_error!r}"
        )

    async def _fire_attempt(
        self,
        attempt_number: int,
        parsed: dict[str, Any] | None,
        validation_errors: list[str],
        is_success: bool,
    ) -> None:
        """Fire the on_attempt hook if set; await if it returns a coroutine."""
        if self._on_attempt is None:
            return
        result = self._on_attempt(
            AttemptInfo(
                attempt_number=attempt_number,
                parsed=parsed,
                validation_errors=validation_errors,
                is_success=is_success,
            )
        )
        if asyncio.iscoroutine(result):
            await result

    async def _run_patch_turn(
        self,
        *,
        prev_dict: dict[str, Any],
        patch_prompt: str,
        original_messages: list[BaseMessage],
        patch_run_name: str,
        callbacks: Callbacks | None,
        raw_messages: list[BaseMessage],
    ) -> tuple[dict[str, Any], BaseException | None]:
        """One patch turn: build the request, call the model, apply the patch.

        Returns ``(new_prev_dict, error)``. On success ``error`` is ``None`` and
        ``new_prev_dict`` is the patched dict. If the model's patch can't be
        applied (malformed op, bad pointer, etc.), ``new_prev_dict`` is the
        *unchanged* input and ``error`` carries a synthetic ``ValueError``
        the caller should feed back to the next turn as the validation
        trigger.

        Mutates ``raw_messages`` in place (appends the outgoing
        ``HumanMessage`` and the model's ``AIMessage``).
        """
        patch_msg = HumanMessage(content=patch_prompt)
        patch_history = list(original_messages) + [
            AIMessage(content=json.dumps(prev_dict)),
            patch_msg,
        ]
        raw_messages.append(patch_msg)
        patch_resp: JsonPatchResponse = await self._patch_llm.ainvoke(
            patch_history,
            config={"callbacks": callbacks or [], "run_name": patch_run_name},
        )
        ops = patch_resp.operations
        raw_messages.append(
            AIMessage(content=json.dumps({"operations": ops}))
        )
        try:
            new_dict = jsonpatch.JsonPatch(ops).apply(prev_dict)
            return new_dict, None
        except (jsonpatch.JsonPatchException, jsonpatch.JsonPointerException) as e:
            return prev_dict, ValueError(
                f"Your JSON Patch could not be applied: {type(e).__name__}: {e}. "
                "Re-issue a corrected patch against the previous JSON output."
            )

def _format_validation_errors(e: ValidationError) -> list[str]:
    """Format a Pydantic ValidationError as a list of ``loc: msg`` strings.

    Used as the ``validation_errors`` field on AttemptInfo so callers don't
    have to re-parse Pydantic's nested error dicts to log per-attempt
    failures.
    """
    formatted: list[str] = []
    for err in e.errors():
        loc = ".".join(str(p) for p in err.get("loc", []))
        msg = err.get("msg", "")
        formatted.append(f"{loc}: {msg}" if loc else msg)
    return formatted


def _error_weight(e: ValidationError) -> int:
    """Sum per-entry weights. AggregatedValidationError(count=N) -> N, else 1."""
    total = 0
    for err in e.errors():
        cause = (err.get("ctx") or {}).get("error")
        total += cause.count if isinstance(cause, AggregatedValidationError) else 1
    return total


def _build_patch_prompt(prev_dict: dict[str, Any], error: BaseException | None) -> str:
    """Build the patch-turn prompt.

    With ``error`` set, prepends the validator-failure prefix — stitcher
    owns the *what* (the validation errors). With ``error=None``, body only
    — the *what* is in the caller's messages (aupdate's first turn).
    """
    body = _PATCH_PROMPT_TEMPLATE.format(
        previous=json.dumps(prev_dict, indent=2),
    )
    if error is None:
        return body
    if isinstance(error, ValidationError):
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
        )
    else:
        errors_text = str(error)
    return _PATCH_REPAIR_PREFIX.format(errors=errors_text) + body
