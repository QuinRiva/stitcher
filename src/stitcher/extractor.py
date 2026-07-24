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
import time
import types
from collections.abc import Awaitable
from typing import Any, Callable, TypeAlias, Union, get_args, get_origin

import jsonpatch
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    convert_to_messages,
)
from langchain_core.runnables import RunnableLambda
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    ValidationError,
    field_serializer,
)

from stitcher.exceptions import AggregatedValidationError


ValidationContext: TypeAlias = dict[str, Any]


# Apply failures get their own budget, separate from the validation-attempt
# budget. Small by design: a pointer that will not resolve after a shape-
# informed retry or two is not going to resolve on the fifth. Keeping it tight
# preserves LLM calls while still giving the model a couple of shape-informed
# chances within one outer attempt.
_MAX_APPLY_ATTEMPTS = 3

# Cap the current-shape blob fed back so a large document cannot bloat the
# repair prompt; the model needs the shape at the failing location, not the
# whole tree.
_SHAPE_CHAR_BUDGET = 800


class AttemptInfo(BaseModel):
    """Per-attempt observability payload, fired by ``Extractor.on_attempt``.

    Diverges from trustcall's ``AttemptInfo`` in one place — ``error: BaseException
    | None`` instead of ``validation_errors: list[str]``. Adjudicators classify
    failures by ``error.errors()`` structure (``type``, ``loc``, ``ctx["error"]``
    for nested ``AggregatedValidationError``) — stitcher pre-formatting to strings
    would throw away the structure right where the consumer needs it. In-memory
    you get the live exception; ``model_dump()`` projects it to
    ``{"type": ..., "errors": ...}`` for Pydantic ``ValidationError`` (preserving
    the structured payload adjudicators key off) and ``{"type": ..., "message":
    ...}`` otherwise, so wide-logs serialise without bespoke helpers.

    Per-attempt token usage / finish_reason / provider metadata are available via
    ``Result.raw_messages`` (which carries the real LangChain ``AIMessage``s
    populated by ``with_structured_output(..., include_raw=True)``) and via the
    aggregated ``Result.metadata`` headline numbers. ``AttemptInfo`` is
    deliberately not duplicated with an ``ai_message`` field; if a consumer
    needs per-attempt metadata pre-correlated with hook firings, pass
    ``callbacks=`` to ``ainvoke``/``aupdate``.

    Fields:
        attempt_number: 1 on the first validation, incrementing per attempt
            (matches the ``attempt_count`` injected into validation_context).
        parsed: the parsed dict that was validated, or ``None`` if the
            model's JSON Patch could not be applied / its response did not
            match the patch protocol (no validate happened).
        error: the failure that drove this attempt's classification —
            ``ValidationError`` for a Pydantic failure, ``ValueError`` for a
            patch-apply or patch-response-shape failure, ``None`` on success.
        is_success: convenience boolean (equivalent to ``error is None``).
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    attempt_number: int
    parsed: dict[str, Any] | None
    error: BaseException | None
    is_success: bool

    @field_serializer("error")
    def _serialize_error(self, e: BaseException | None) -> dict[str, Any] | None:
        if e is None:
            return None
        if isinstance(e, ValidationError):
            # e.errors() can carry live exception objects in ctx (e.g. a nested
            # AggregatedValidationError), which breaks json.dumps downstream;
            # e.json() is pydantic's JSON-safe projection of the same list.
            return {"type": "ValidationError", "errors": json.loads(e.json())}
        return {"type": type(e).__name__, "message": str(e)}


# Both sync (returns None) and async (returns Awaitable[None]) callables
# are accepted; stitcher awaits if the return is a coroutine.
OnAttempt: TypeAlias = Callable[["AttemptInfo"], None | Awaitable[None]]


class JsonPatchResponse(BaseModel):
    """Model's response on a patch turn: structured reasoning followed by JSON Patch ops.

    The two-field shape (reasoning then operations) forces a chain-of-thought step
    before the model emits its operations — same idea as trustcall's ``planned_edits``.
    Stitcher reads ``operations`` for the patch apply; ``reasoning`` is preserved in
    ``raw_messages`` for observability but otherwise discarded.
    """

    reasoning: str = Field(
        ...,
        description=(
            "First, walk through what needs to change and the JSON Patch operations "
            "needed. For a validation-driven patch: cite each error and the operation "
            "to heal it. For a user-driven update: explain what the user requested and "
            "how the patch encodes it. Reason step by step — this grounds the "
            "operations below."
        ),
    )
    operations: list[dict[str, Any]] = Field(
        ...,
        description=(
            "Then, the RFC 6902 JSON Patch — an array of {op, path, value} entries. "
            "Operations are applied SEQUENTIALLY: each operates on the state produced "
            "by the previous one.\n\n"
            "Key rules:\n"
            "- `add` is atomic: deliver the COMPLETE final value in one operation. Do "
            "  NOT add a placeholder (`{}` or `[]`) intending to fill it with later "
            "  `replace` ops — the placeholder fails validation immediately and the "
            "  follow-ups have no path to target.\n"
            "- `replace` requires the path to already exist. If you see `input_value={}` "
            "  in a validation error, the parent is empty and the field doesn't exist "
            "  yet — use `add`, not `replace`.\n"
            "- For multiple `remove`s on the same list, order them HIGHEST-INDEX-FIRST "
            "  to avoid index shift between sequential applies.\n"
            "- For appending to a list, use `\"path\": \"/items/-\"`.\n"
            "- Paths are JSON Pointers: `/items/0` to address the first list item, "
            "  `/foo/bar` to address a nested object field."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "reasoning": "Field 'age' has wrong type ('thirty' string vs int). Replace with parsed int.",
                    "operations": [{"op": "replace", "path": "/age", "value": 30}],
                },
                {
                    "reasoning": "Item at /items/2 duplicates /items/0; remove the dupe.",
                    "operations": [{"op": "remove", "path": "/items/2"}],
                },
                {
                    "reasoning": "User requested adding a tax line; append a complete entry as one atomic add.",
                    "operations": [
                        {
                            "op": "add",
                            "path": "/lines/-",
                            "value": {"name": "tax", "amount_cents": 825},
                        }
                    ],
                },
                {
                    "reasoning": "Three duplicates at /items/2, /items/5, /items/8; remove highest-index-first to keep indices stable.",
                    "operations": [
                        {"op": "remove", "path": "/items/8"},
                        {"op": "remove", "path": "/items/5"},
                        {"op": "remove", "path": "/items/2"},
                    ],
                },
            ]
        }
    )


class TokenUsage(BaseModel):
    """Aggregated token counts over one or more LLM calls.

    Fields:
        input_tokens: billable input tokens. Includes cached input on
            providers that bill cached reads (Anthropic, OpenAI) — the
            ``cached_input_tokens`` field is a *subset* breakdown, not
            additive.
        cached_input_tokens: subset of ``input_tokens`` that was served
            from a provider-side prompt cache. Sum from LangChain's
            ``UsageMetadata.input_token_details.cache_read``; ``0`` for
            providers that don't cache. Use ``input_tokens -
            cached_input_tokens`` for uncached input volume.
        reasoning_tokens: subset of the underlying ``output_tokens`` spent
            on hidden chain-of-thought (Gemini thinking, OpenAI o-series).
            Sum from ``UsageMetadata.output_token_details.reasoning``;
            ``0`` for non-thinking models. Note: ``reasoning_tokens`` are
            already part of what the provider bills as output, so the
            wall-clock cost is ``input_tokens + reasoning_tokens +
            output_payload_tokens``.
        output_payload_tokens: ``output_tokens - reasoning_tokens`` — the
            user-visible response payload (the JSON the model actually
            emitted as the structured output, before stitcher's
            re-validation). For thinking models this is the non-reasoning
            slice of output; for non-thinking models this equals
            ``output_tokens`` verbatim.
    """
    model_config = ConfigDict(frozen=True)

    input_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int
    output_payload_tokens: int


class Metadata(BaseModel):
    """Aggregated observability headline numbers attached to ``Result``.

    Fields:
        initial: token usage for the *first LLM call that contributed to
            ``Result.value``*. For ``ainvoke`` this is the initial extract
            whose output seeded the patch loop (post-catastrophic-re-extract
            if that fired). For ``aupdate`` this is the first patch turn
            whose ops successfully applied to the existing seed. Discarded
            calls (parse failures that triggered re-extract, catastrophic
            initial extracts, patch turns whose ops failed to apply) are
            *not* counted here.
        total: token usage summed across every LLM call stitcher made,
            including any that were discarded. ``total - initial`` is the
            cost of repair/retry; ``initial`` alone is the cost of the
            seed.
        duration_seconds: wall time of the whole ``ainvoke`` / ``aupdate``
            call, measured around the patch loop (excludes the
            ``Result``/``Metadata`` construction below it but includes
            every LLM call, normaliser pass, and validation pass).

    Output-payload size shortcut: if you called ``ainvoke`` and
    ``Result.attempts == 1``, then ``metadata.initial.output_payload_tokens``
    is the token size of ``Result.value`` as the model emitted it (no patches
    happened). In every other case — ``attempts > 1`` on ``ainvoke``, or any
    ``aupdate`` call — ``initial.output_payload_tokens`` reflects the seed
    extract / first applied patch envelope and is *not* the size of
    ``Result.value``. Tokenize ``Result.value.model_dump_json()`` with your
    provider's tokenizer if you need the final payload size in those cases.

    Deferred fields (not exposed; added if a real consumer appears):
        ``num_validation_failures``, ``num_patch_apply_failures``,
        ``num_parse_failures`` — counters of *why* the loop retried.
        Currently derivable by passing ``on_attempt=`` and counting; not
        worth duplicating until a pipeline shows up that wants them as a
        wide-log signal.
    """
    model_config = ConfigDict(frozen=True)

    initial: TokenUsage
    total: TokenUsage
    duration_seconds: float


class Result(BaseModel):
    """Result of a successful ``ainvoke`` or ``aupdate`` call.

    ``value`` is annotated ``SerializeAsAny[BaseModel]`` so that
    ``Result.model_dump()`` emits the user schema's full field set, not just
    the abstract ``BaseModel`` base (Pydantic v2's default is to serialise
    against the declared type, which would strip every subclass field).
    """
    model_config = ConfigDict(frozen=True)

    value: SerializeAsAny[BaseModel]
    attempts: int
    was_re_extracted: bool
    raw_messages: list[BaseMessage]
    metadata: Metadata


# Body of every patch-turn prompt: target schema + how-to-patch instruction.
# The caller's messages own the *what* (extraction prompt for ainvoke,
# update intent for aupdate); this template owns the *how*. The detailed
# format and ops rules live on JsonPatchResponse's field descriptions,
# which the model sees as part of the structured-output binding — NOT
# duplicated here, by design (see commit d947c0e).
#
# Markdown structure: ``## Headings``, ``---`` separators, and
# ```` ```json ``` ```` code fences render natively in Langfuse and
# similar trace viewers, so a human reading a stuck-patch trial can scan
# the sections at a glance instead of squinting at XML-style tags.
#
# Schema is dumped as compact (no indent) JSON: the model handles it fine
# and the trace panel doesn't waste vertical space on a multi-line dump
# the human almost never needs to eyeball. Copy-paste into a JSON
# formatter if you do.
#
# prev_dict is NOT embedded here — it lives in the AIMessage immediately
# above this HumanMessage in patch_history. The two are contiguous so the
# model has full attention on it; embedding it again would burn tokens
# for no benefit.
_PATCH_PROMPT_TEMPLATE = (
    "## Target Schema\n\n"
    "The patched output must conform to:\n\n"
    "```json\n{schema}\n```\n\n"
    "---\n\n"
    "## Instructions\n\n"
    "Produce a JSON Patch against your previous output (the assistant "
    "message immediately above). Return `reasoning` then `operations` — "
    "their format and ops rules are in the structured-output schema.\n"
)

# Prepended to the body when stitcher owns the *what* — i.e. when the
# trigger is an internal validator failure rather than a user-supplied intent.
#
# The errors block is deliberately NOT fenced: consumers author validator
# messages as markdown (bold headers, code-spans, bullets) meant to render
# in Langfuse-style trace viewers. A fence would show them as literal text.
_PATCH_REPAIR_PREFIX = (
    "## Validation Errors\n\n"
    "The previous JSON output failed validation. The errors below are "
    "symptoms — diagnose what your previous output got wrong, then patch "
    "the root cause, not just the surface message.\n\n"
    "{errors}\n\n"
    "---\n\n"
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
        # built-in path skips that context). Cache the JSON-schema dict so the
        # patch prompt can embed it: on patch turns we bind JsonPatchResponse
        # to the structured output, so the user's schema is no longer in the
        # API-level constraint and the model needs it inline to know what
        # shape it's patching toward.
        self._schema_json = schema.model_json_schema()
        # Bind the JSON schema straight onto the model (Gemini:
        # ``response_mime_type="application/json"`` + ``response_json_schema``)
        # and parse the raw AIMessage ourselves in ``_parse_content``. This is
        # exactly what ``with_structured_output(method="json_schema")`` binds
        # under the hood, minus its output-parser chain — so each LLM call is a
        # SINGLE generation span in Langfuse (carrying the real
        # ``usage_metadata`` / ``response_metadata`` / ``finish_reason`` on the
        # AIMessage), instead of the ~7 nested runnables ``include_raw=True``
        # expands one call into (RunnableSequence / RunnableParallel<raw> /
        # RunnableWithFallbacks / RunnableAssign / JsonOutputParser / internal
        # lambdas). Stitcher already owns validation + JSON-Patch repair, so
        # owning the initial JSON parse too folds parse failures into the same
        # repair loop and lets us drop the langfuse-version-coupled
        # ``langsmith:hidden`` trace hack entirely. ``_parse_content`` returns
        # the same ``{raw, parsed, parsing_error}`` envelope include_raw did,
        # so the patch loop below is unchanged.
        self._initial_llm = llm.bind(
            response_mime_type="application/json",
            response_json_schema=self._schema_json,
        )
        self._patch_llm = llm.bind(
            response_mime_type="application/json",
            response_json_schema=JsonPatchResponse.model_json_schema(),
        )

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        *,
        validation_context: ValidationContext | None = None,
        run_name: str | None = None,
        **config: Any,
    ) -> Result:
        """Extract a validated instance of ``self.schema`` from ``messages``.

        Args:
            messages: conversation history to send to the model.
            validation_context: passed to ``schema.model_validate(...,
                context=...)`` on every validation pass, with one
                stitcher-supplied key merged in: ``attempt_count`` (1 on
                the first validation, 2 after the first patch, ...). User
                keys override on collision (matches trustcall's contract).
                Use this for any cross-field invariants the schema's
                validators need.
            run_name: optional name for the parent extraction run. Stitcher
                wraps each call in a ``RunnableLambda`` so the initial
                extract and any patch turns appear as children of one
                parent in the trace tree (one Langfuse trace per
                ``ainvoke`` call). The user-supplied ``run_name`` becomes
                the parent's name; children are always ``"initial"`` and
                ``"patch"``. Defaults to ``"stitcher"`` when unset.
            **config: forwarded verbatim to LangChain's ``RunnableConfig``
                on every LLM call (initial extract, every patch turn).
                Common fields: ``callbacks=[handler, ...]`` for trace
                handlers, ``tags=["batch_42", ...]`` for trace filtering,
                ``metadata={"trace_id": "...", ...}`` for trace context.
                See LangChain's ``RunnableConfig`` for the full list.
                Stitcher does not validate these keys — misspellings will
                silently no-op.

        Returns:
            ``Result(value, attempts, was_re_extracted, raw_messages, metadata)``.

        Raises:
            RuntimeError: if ``max_attempts`` is exhausted without a valid
                extraction.
            Exception: if LangChain's structured-output binding raises a
                ``parsing_error`` on the initial extract while
                ``allow_re_extract`` is disabled. ``ainvoke`` always enables
                it, so this only surfaces if a future caller threads
                ``allow_re_extract=False`` through.
            Note that for ``aupdate`` (where ``allow_re_extract`` is always
                ``False``) the initial-extract branch is never entered, so
                ``parsing_error`` from that path is unreachable; patch-turn
                ``parsing_error``s are folded into the patch loop's
                bad-patch retry path and consume an attempt rather than
                raising.
        """
        # Wrap the patch loop in a RunnableLambda so LangChain's callback
        # machinery establishes a parent run — the initial extract and any
        # patch turns become children in the trace tree (one Langfuse trace
        # per ainvoke call instead of N separate ones). User-supplied
        # callbacks/tags/metadata are attached to the parent and inherited
        # by children via LangChain's context-var propagation.
        async def _execute(_input: Any) -> Result:
            return await self._patch_loop(
                original_messages=messages,
                prev_dict=None,
                validation_context=validation_context,
                allow_re_extract=True,
            )

        parent_config: dict[str, Any] = {
            **config,
            "run_name": run_name or "stitcher",
        }
        return await RunnableLambda(_execute).ainvoke(None, config=parent_config)

    async def aupdate(
        self,
        existing: BaseModel | dict[str, Any],
        messages: list[BaseMessage],
        *,
        validation_context: ValidationContext | None = None,
        run_name: str | None = None,
        initial_error: BaseException | None = None,
        **config: Any,
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
            run_name: as for ``ainvoke``; only the ``.patch`` suffix is used
                (no initial extract in update mode).
            initial_error: optional per-call error that seeds the FIRST patch
                turn's prompt via the same ``## Validation Errors`` framing a
                validator failure would produce. ``aupdate`` seeds the loop
                with an existing document, so without this the first patch
                turn carries no validator header — the *what* is expected to
                come through ``messages``. Handing the initial adjudication
                failure here delivers it through the per-turn patch channel
                instead, which is rebuilt — and so naturally REPLACED by the
                real current error — every subsequent turn. It never touches
                the system channel / ``original_messages``, so validation state
                cannot accumulate stale there. With ``initial_error=None`` the
                first patch prompt is byte-identical to today's headerless one.
            **config: as for ``ainvoke`` — forwarded to LangChain's
                ``RunnableConfig`` on every patch turn.

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
        # Normalise the seed before the first patch turn so the model's
        # patch prompt shows the clean shape (otherwise the model would
        # write JSON Pointer paths against the stringified form, which
        # then fail to apply). For BaseModel seeds this is a no-op;
        # for raw dicts it catches Gemini-style stringified-JSON-in-list-slot.
        prev_dict = _normalise_stringified_json(prev_dict, self.schema)

        # Same parent-runnable wrapping as ainvoke — see comment there.
        async def _execute(_input: Any) -> Result:
            return await self._patch_loop(
                original_messages=messages,
                prev_dict=prev_dict,
                validation_context=validation_context,
                allow_re_extract=False,
                initial_error=initial_error,
            )

        parent_config: dict[str, Any] = {
            **config,
            "run_name": run_name or "stitcher",
        }
        return await RunnableLambda(_execute).ainvoke(None, config=parent_config)

    async def _patch_loop(
        self,
        *,
        original_messages: list[BaseMessage],
        prev_dict: dict[str, Any] | None,
        validation_context: ValidationContext | None,
        allow_re_extract: bool,
        initial_error: BaseException | None = None,
    ) -> Result:
        """Shared validate-and-patch loop for ``ainvoke`` and ``aupdate``.

        Always runs INSIDE a ``RunnableLambda`` parent set up by ``ainvoke``
        / ``aupdate`` — this means inner LLM calls inherit the parent run's
        callbacks/tags/metadata/parent_run_id via LangChain's context vars,
        so we only need to set ``run_name`` on each child config to
        differentiate them in the trace tree.

        State machine on each iteration, driven by ``(prev_dict, last_error)``:

        - ``prev_dict is None`` → initial extract via JSON mode (only ever
          reached from ``ainvoke``: either the first iteration or after the
          catastrophic-re-extract path resets ``prev_dict`` to ``None``).
        - ``prev_dict`` set, ``last_error is None`` → headerless patch turn;
          the user's messages own the *what* (``aupdate``'s first turn).
        - ``prev_dict`` set, ``last_error`` set → repair patch turn; the
          validator-failure prefix is prepended to the patch prompt.

        ``initial_error`` (``aupdate`` only) seeds ``last_validation_error``
        before the loop, so the FIRST patch turn is a repair turn carrying the
        ``## Validation Errors`` framing around that error instead of a
        headerless turn. It is replaced by the real current error every turn
        after (the loop already rewrites ``last_validation_error`` each pass),
        so nothing accumulates. With ``initial_error=None`` the first turn is
        headerless exactly as before.

        ``allow_re_extract`` gates the catastrophic-re-extract path — only
        ``ainvoke`` enables it because only ``ainvoke`` has a fresh-extract
        path to fall back to.

        Happy-path tracking: ``happy_path_messages`` records the AIMessages
        whose outputs are reflected in the eventually-returned ``Result.value``.
        It is rebuilt from scratch every time a fresh extract seeds
        ``prev_dict`` (initial extract, or any re-extract after a
        catastrophic-weight or parse-failure trigger) and appended to every
        time a patch successfully applies. ``Metadata.initial`` reads
        ``happy_path_messages[0]``; ``Metadata.total`` reads every AIMessage
        in ``raw_messages`` regardless of whether its output survived.

        Parsing-error semantics (when LangChain's structured-output binding
        emits ``result['parsing_error']``):

        - On the initial extract: treated as a hard failure of the seed —
          analogous to a catastrophic-weight ValidationError. If
          ``allow_re_extract`` (``ainvoke``), reset and try again, consuming
          an attempt. If not (``aupdate`` cannot reach this branch because
          its ``prev_dict`` is never ``None``), raise.
        - On a patch turn: treated as a patch-protocol failure, structurally
          identical to ``jsonpatch.JsonPatch(...).apply(...)`` raising on
          bad ops. Fed back to the next iteration as a corrective
          ``ValueError`` ("your response didn't match the required shape"),
          consuming an attempt. The malformed AIMessage is still appended
          to ``raw_messages`` so the caller sees what the model emitted.
        """
        # Child run names — just enough to distinguish initial extract from
        # patch turns inside the parent trace. The user's run_name lives on
        # the parent (set in ainvoke/aupdate), not on these children. ``run_name``
        # here names the LLM generation span itself (no wrapper span), so the
        # Langfuse tree is one visible ``initial`` / ``patch`` generation per
        # LLM round-trip.
        initial_config: dict[str, Any] = {"run_name": "initial"}
        patch_config: dict[str, Any] = {"run_name": "patch"}

        t_start = time.monotonic()
        # Accept LangChain chat-dicts ({"role": ..., "content": ...}) as well as
        # BaseMessage instances — normalised here so ``Result.raw_messages``
        # (a validated ``list[BaseMessage]``) holds real message objects.
        original_messages = convert_to_messages(original_messages)
        raw_messages: list[BaseMessage] = list(original_messages)
        happy_path_messages: list[AIMessage] = []
        attempts = 0
        was_re_extracted = False
        # Seeded from initial_error (aupdate) so the first patch turn carries
        # its ## Validation Errors framing; None for ainvoke (headerless first
        # turn / initial extract). Replaced by the real current error each pass.
        last_validation_error: BaseException | None = initial_error

        while attempts < self.max_attempts:
            attempts += 1

            if prev_dict is None:
                init_result = await self._invoke_initial(
                    original_messages, initial_config
                )
                init_raw: AIMessage = init_result["raw"]
                raw_messages.append(init_raw)
                init_parse_error = init_result["parsing_error"]
                if init_parse_error is not None:
                    # The model gave us something we can't use as a seed.
                    # Symmetric with catastrophic-weight ValidationError:
                    # discard, reset happy path, re-extract on next loop.
                    last_validation_error = init_parse_error
                    await self._fire_attempt(attempts, None, init_parse_error)
                    if allow_re_extract:
                        was_re_extracted = True
                        happy_path_messages = []
                        continue
                    # aupdate never reaches the prev_dict-is-None branch, but
                    # if a future caller wires allow_re_extract=False here,
                    # surface the failure rather than loop indefinitely.
                    raise init_parse_error
                prev_dict = init_result["parsed"]
                # Fresh seed: the happy path starts from this AIMessage.
                happy_path_messages = [init_raw]
            else:
                prev_dict, patch_error, applied_msg = await self._run_patch_turn(
                    prev_dict=prev_dict,
                    patch_prompt=_build_patch_prompt(
                        last_validation_error, self._schema_json
                    ),
                    original_messages=original_messages,
                    patch_config=patch_config,
                    raw_messages=raw_messages,
                )
                if applied_msg is not None:
                    happy_path_messages.append(applied_msg)
                if patch_error is not None:
                    last_validation_error = patch_error
                    await self._fire_attempt(attempts, None, patch_error)
                    continue

            # Inject attempt_count so validators can implement
            # first-attempt-strict / later-lenient patterns. Matches trustcall's
            # contract: user-supplied keys win, so callers can override
            # (typically only useful for testing).
            ctx = {"attempt_count": attempts, **(validation_context or {})}
            # Normalise common LLM-side malformations before validation —
            # specifically: stringified-JSON in object/array slots, the
            # most common Gemini soft-enforcement artefact. Safe-by-
            # construction (only fires when the slot's annotation is
            # structural and the parsed shape matches); see
            # _normalise_stringified_json for the full rule.
            prev_dict = _normalise_stringified_json(prev_dict, self.schema)
            try:
                value = self.schema.model_validate(prev_dict, context=ctx)
            except ValidationError as e:
                last_validation_error = e
                await self._fire_attempt(attempts, prev_dict, e)
                if (
                    allow_re_extract
                    and self.max_validation_error_weight is not None
                    and _error_weight(e) > self.max_validation_error_weight
                ):
                    was_re_extracted = True
                    happy_path_messages = []
                    prev_dict = None
                continue
            except AggregatedValidationError as e:
                last_validation_error = e
                await self._fire_attempt(attempts, prev_dict, e)
                if (
                    allow_re_extract
                    and self.max_validation_error_weight is not None
                    and e.count > self.max_validation_error_weight
                ):
                    was_re_extracted = True
                    happy_path_messages = []
                    prev_dict = None
                continue

            # Result construction sits OUTSIDE the try above so a bug in
            # stitcher's own envelope can never masquerade as a schema
            # validation failure and silently burn patch rounds.
            await self._fire_attempt(attempts, prev_dict, None)
            return Result(
                value=value,
                attempts=attempts,
                was_re_extracted=was_re_extracted,
                raw_messages=raw_messages,
                metadata=_build_metadata(
                    happy=happy_path_messages,
                    all_messages=raw_messages,
                    t_start=t_start,
                ),
            )

        raise RuntimeError(
            f"Extractor exhausted {self.max_attempts} attempts. "
            f"Last validation error: {last_validation_error!r}"
        )

    async def _invoke_initial(
        self,
        messages: list[BaseMessage],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Call the schema-bound model for the initial extract and self-parse.

        Returns the ``{raw, parsed, parsing_error}`` envelope the patch loop
        consumes (``parsed`` is the raw dict here). ``config`` carries
        ``run_name="initial"``, which names the generation span itself — no
        wrapper span, so Langfuse shows one visible ``initial`` generation.
        Callbacks / tags / metadata reach the call via context vars set by
        the parent ``RunnableLambda`` in ``ainvoke`` / ``aupdate``.
        """
        raw = await self._initial_llm.ainvoke(messages, config=config)
        return _parse_content(raw, model=None)

    async def _invoke_patch(
        self,
        history: list[BaseMessage],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Mirror of ``_invoke_initial`` for the patch path; parses the model's
        JSON into a ``JsonPatchResponse`` (surfacing a ``parsing_error`` if it
        doesn't match), and names the generation span ``patch``.
        """
        raw = await self._patch_llm.ainvoke(history, config=config)
        return _parse_content(raw, model=JsonPatchResponse)

    async def _fire_attempt(
        self,
        attempt_number: int,
        parsed: dict[str, Any] | None,
        error: BaseException | None,
    ) -> None:
        """Fire the on_attempt hook if set; await if it returns a coroutine."""
        if self._on_attempt is None:
            return
        result = self._on_attempt(
            AttemptInfo(
                attempt_number=attempt_number,
                parsed=parsed,
                error=error,
                is_success=error is None,
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
        patch_config: dict[str, Any],
        raw_messages: list[BaseMessage],
    ) -> tuple[dict[str, Any], BaseException | None, AIMessage | None]:
        # patch_config carries only run_name ("patch") — callbacks, tags,
        # metadata, and parent_run_id come from the surrounding RunnableLambda
        # context set up by ainvoke/aupdate.
        """One outer patch turn, with an inner shape-informed apply-retry loop.

        Returns ``(new_prev_dict, error, applied_message)``:

        - ``error is None``: patch applied. ``new_prev_dict`` is the patched
          dict; ``applied_message`` is the model's real ``AIMessage`` (now on
          the happy path — its output is reflected in the patched state).
        - ``error`` set: the patch could not be applied, either because the
          model returned a ``parsing_error`` (response didn't match the
          ``{reasoning, operations}`` shape) or because the apply-retry cap
          ``_MAX_APPLY_ATTEMPTS`` was exhausted. ``new_prev_dict`` is the
          *unchanged* input; ``applied_message`` is ``None`` (the messages are
          still on ``raw_messages`` for debugging, just not on the happy path).
          The caller feeds ``error`` back as the next turn's validation trigger
          and consumes exactly one validation attempt.

        Apply-failure hardening (see the module functions ``_apply_failure_``
        ``feedback`` etc.): a patch that fails to *apply* is retried in place
        with a shape-enriched prompt — the current value at the deepest
        existing ancestor of the failing pointer — up to ``_MAX_APPLY_ATTEMPTS``
        times *within this single outer turn*. Those inner retries do NOT
        consume a validation attempt; only inner-cap exhaustion (or a non-apply
        parse error) propagates an error to the outer loop. A purely mechanical
        pointer error therefore can no longer exhaust the semantic-repair
        budget.

        Termination: the outer ``_patch_loop`` bounds total outer attempts at
        ``max_attempts``; each outer attempt runs at most ``_MAX_APPLY_ATTEMPTS``
        patch generations here, so total LLM calls are bounded by
        ``max_attempts * _MAX_APPLY_ATTEMPTS`` (e.g. a caller with
        ``max_attempts=7`` bounds worst-case patch generations at 21) and the
        loop always terminates.

        ``on_attempt`` semantics: inner apply retries are sub-attempts and do
        NOT fire ``on_attempt`` — only the outer-attempt outcome (applied then
        validated, or inner-cap exhaustion) fires it once from the outer loop.

        Mutates ``raw_messages`` in place: appends the outgoing
        ``HumanMessage`` and the model's ``AIMessage`` for every inner attempt
        (whether the call succeeded, hit a ``parsing_error``, or failed to
        apply).
        """
        current_prompt = patch_prompt
        for apply_attempt in range(_MAX_APPLY_ATTEMPTS):
            patch_msg = HumanMessage(content=current_prompt)
            patch_history = list(original_messages) + [
                AIMessage(content=json.dumps(prev_dict)),
                patch_msg,
            ]
            raw_messages.append(patch_msg)
            patch_result = await self._invoke_patch(patch_history, patch_config)
            patch_raw: AIMessage = patch_result["raw"]
            raw_messages.append(patch_raw)

            parse_error = patch_result["parsing_error"]
            if parse_error is not None:
                # The model returned something that didn't match
                # JsonPatchResponse (missing operations, wrong types, prose
                # instead of JSON, ...). This is a protocol error, not an apply
                # failure — hand it straight back to the outer loop (no apply
                # retry), consuming one attempt.
                return prev_dict, ValueError(
                    f"Your response did not match the required "
                    f"{{reasoning, operations}} shape: "
                    f"{type(parse_error).__name__}: {parse_error}. "
                    "Reissue with both fields populated."
                ), None

            ops = patch_result["parsed"].operations
            try:
                new_dict = jsonpatch.JsonPatch(ops).apply(prev_dict)
                return new_dict, None, patch_raw
            except (
                jsonpatch.JsonPatchException,
                jsonpatch.JsonPointerException,
            ) as exc:
                feedback = _apply_failure_feedback(prev_dict, ops, exc)
                if apply_attempt < _MAX_APPLY_ATTEMPTS - 1:
                    # Retry within this outer attempt: rebuild the patch prompt
                    # around the shape-enriched apply error. Does NOT consume a
                    # validation attempt.
                    current_prompt = _build_patch_prompt(
                        feedback, self._schema_json
                    )
                    continue
                # Apply budget exhausted for this outer attempt: hand the
                # shape-enriched error back to the outer loop (consumes exactly
                # one validation attempt), which keeps overall termination.
                return prev_dict, feedback, None

        # Unreachable: the loop always returns. Present for type-checkers.
        raise AssertionError("apply-retry loop exited without returning")

def _parse_content(
    raw: AIMessage, model: type[BaseModel] | None
) -> dict[str, Any]:
    """Parse a schema-bound model's JSON response into the include_raw envelope.

    Stitcher binds the schema directly and parses here rather than delegating
    to LangChain's ``with_structured_output`` parser chain, so each LLM call
    stays a single Langfuse generation span. The returned shape
    ``{raw, parsed, parsing_error}`` matches what ``include_raw=True`` used to
    yield, so the patch loop is unchanged.

    ``model=None`` (initial extract) returns the parsed dict. A BaseModel
    subclass (patch turn) validates into that model. A JSON-decode failure or
    a model-validation failure both surface as ``parsing_error`` — folded into
    the patch loop exactly like LangChain's old ``parsing_error``.
    """
    text = _content_text(raw.content)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        return {"raw": raw, "parsed": None, "parsing_error": e}
    if model is None:
        return {"raw": raw, "parsed": data, "parsing_error": None}
    try:
        return {"raw": raw, "parsed": model.model_validate(data), "parsing_error": None}
    except ValidationError as e:
        return {"raw": raw, "parsed": None, "parsing_error": e}


def _content_text(content: Any) -> str:
    """Extract the JSON text payload from an ``AIMessage.content``.

    Plain-string content is returned as-is. Thinking models (Gemini 3,
    Claude extended thinking, OpenAI o-series via LangChain) return ``content``
    as a list of blocks — e.g. ``[{"type": "thinking", ...}, {"type": "text",
    "text": "<the JSON>"}]``. The structured-output payload lives in the
    ``text`` block(s), never the thinking block, so we concatenate the text of
    every ``type == "text"`` block and ignore the rest. (LangChain's
    ``with_structured_output`` parser did this for us; self-parsing means we
    do it here.)
    """
    if isinstance(content, str):
        return content
    return "".join(
        block["text"]
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _normalise_stringified_json(value: Any, annotation: Any) -> Any:
    """Walk ``value`` against ``annotation``; re-parse stringified-JSON
    values that sit in non-string structural slots.

    Motivating case: Gemini soft-enforcement on nested object types. A slot
    typed ``list[Item]`` occasionally receives ``["{...}", "{...}"]`` instead
    of ``[{...}, {...}]``. Without this helper, Pydantic rejects the strings
    and stitcher pays a patch-turn round-trip to repair. With it, we re-parse
    the strings in place and validation passes on the first attempt.

    Safe-by-construction: only fires when ALL of:

    1. ``value`` is a ``str``
    2. The annotation requires a structural type (``BaseModel`` / ``list[X]``
       / ``dict[K, V]``, or one of those inside a ``Union``)
    3. The string starts with ``{`` or ``[`` after stripping whitespace
    4. ``json.loads`` succeeds and the parsed value matches the expected
       container type (``dict`` for object-typed slots, ``list`` for arrays)

    Slots whose annotation is ``str`` (or ``Union[str, ...]``) are left alone
    even if the string content happens to look like JSON — the schema
    explicitly allows strings there.

    Limitations:

    - Discriminated unions: when ``value`` is a dict and the annotation is
      ``Union[ModelA, ModelB, ...]``, we recurse into the first non-None
      member (no discriminator-based dispatch). Worst case we miss
      normalisation opportunities deeper in; the patch loop catches anything
      we miss.
    - ``Annotated[T, ...]`` is not unwrapped explicitly; Pydantic v2 already
      strips it from ``FieldInfo.annotation`` so this is moot in practice.
    """
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[T] / Union[A, B, ...]
    if origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        if isinstance(value, str) and str in args:
            # Schema explicitly allows strings here; respect that.
            return value
        for arg in args:
            if arg is not type(None):
                return _normalise_stringified_json(value, arg)
        return value

    # list[T]
    if origin is list:
        item_type = args[0] if args else Any
        if isinstance(value, str):
            parsed = _try_parse_json_string(value, list)
            if parsed is not None:
                return [_normalise_stringified_json(item, item_type) for item in parsed]
            return value
        if isinstance(value, list):
            return [_normalise_stringified_json(item, item_type) for item in value]
        return value

    # dict[K, V]
    if origin is dict:
        v_type = args[1] if len(args) >= 2 else Any
        if isinstance(value, str):
            parsed = _try_parse_json_string(value, dict)
            if parsed is not None:
                return {k: _normalise_stringified_json(v, v_type) for k, v in parsed.items()}
            return value
        if isinstance(value, dict):
            return {k: _normalise_stringified_json(v, v_type) for k, v in value.items()}
        return value

    # BaseModel subclass
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if isinstance(value, str):
            parsed = _try_parse_json_string(value, dict)
            if parsed is not None:
                return _normalise_stringified_json(parsed, annotation)
            return value
        if isinstance(value, dict):
            result = dict(value)  # preserve unknown keys for Pydantic to handle
            for fname, finfo in annotation.model_fields.items():
                if fname in result:
                    result[fname] = _normalise_stringified_json(
                        result[fname], finfo.annotation
                    )
            return result
        return value

    # Primitive, Any, or unrecognised — leave alone.
    return value


def _try_parse_json_string(s: str, expected_container: type) -> Any | None:
    """Try ``json.loads(s.strip())``; return the parsed value if it's the
    expected container type (``dict`` or ``list``), else ``None``.

    Cheap prefix check (must start with ``{`` or ``[``) avoids parsing
    strings that obviously aren't JSON.
    """
    stripped = s.strip()
    if not stripped:
        return None
    expected_char = "{" if expected_container is dict else "["
    if not stripped.startswith(expected_char):
        return None
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, expected_container):
        return None
    return parsed


def _error_weight(e: ValidationError) -> int:
    """Sum per-entry weights. AggregatedValidationError(count=N) -> N, else 1."""
    total = 0
    for err in e.errors():
        cause = (err.get("ctx") or {}).get("error")
        total += cause.count if isinstance(cause, AggregatedValidationError) else 1
    return total


def _aggregate_usage(messages: list[AIMessage]) -> TokenUsage:
    """Sum ``UsageMetadata`` across ``messages`` into a ``TokenUsage``.

    LangChain convention: ``input_tokens`` and ``output_tokens`` are billable
    totals; ``input_token_details.cache_read`` and
    ``output_token_details.reasoning`` are *subset* breakdowns (already
    counted inside the parent total, not additive). We surface the subsets
    as peer fields and compute ``output_payload_tokens = output_tokens -
    reasoning_tokens`` so the user-visible response payload is one read away.

    Messages whose ``usage_metadata`` is ``None`` (some test fakes, some
    providers when caching is hot, some non-final stream chunks) contribute
    zero and are skipped without error.
    """
    input_total = 0
    cached_total = 0
    reasoning_total = 0
    payload_total = 0
    for m in messages:
        usage = m.usage_metadata
        if usage is None:
            continue
        input_total += usage.get("input_tokens", 0)
        output_raw = usage.get("output_tokens", 0)
        in_details = usage.get("input_token_details") or {}
        out_details = usage.get("output_token_details") or {}
        cached_total += in_details.get("cache_read", 0)
        msg_reasoning = out_details.get("reasoning", 0)
        reasoning_total += msg_reasoning
        payload_total += output_raw - msg_reasoning
    return TokenUsage(
        input_tokens=input_total,
        cached_input_tokens=cached_total,
        reasoning_tokens=reasoning_total,
        output_payload_tokens=payload_total,
    )


def _build_metadata(
    *,
    happy: list[AIMessage],
    all_messages: list[BaseMessage],
    t_start: float,
) -> Metadata:
    """Roll the per-call AIMessages into a ``Metadata`` for ``Result``.

    ``initial`` aggregates only the first AIMessage on the happy path — the
    call whose output seeded ``Result.value`` (post-catastrophic-re-extract
    if that fired). ``total`` aggregates every AIMessage in
    ``all_messages``, including any discarded by re-extract or by failed
    patch applies.
    """
    ai_only = [m for m in all_messages if isinstance(m, AIMessage)]
    initial = _aggregate_usage(happy[:1])
    total = _aggregate_usage(ai_only)
    return Metadata(
        initial=initial,
        total=total,
        duration_seconds=time.monotonic() - t_start,
    )


def _build_patch_prompt(
    error: BaseException | None,
    schema_json: dict[str, Any],
) -> str:
    """Build the patch-turn prompt.

    With ``error`` set, prepends the validator-failure prefix — stitcher
    owns the *what* (the validation errors). With ``error=None``, body only
    — the *what* is in the caller's messages (aupdate's first turn).

    The body always embeds ``schema_json`` so the model has the target
    schema available on the patch turn (where the structured-output
    binding is on JsonPatchResponse, not the user's schema).

    The model's previous output is NOT embedded here. It sits in the
    AIMessage immediately preceding this HumanMessage in patch_history;
    embedding it again would just burn tokens for no attention benefit.
    """
    body = _PATCH_PROMPT_TEMPLATE.format(
        schema=json.dumps(schema_json),
    )
    if error is None:
        return body
    if isinstance(error, ValidationError):
        errors_text = _format_validation_errors(error.errors())
    else:
        errors_text = str(error)
    return _PATCH_REPAIR_PREFIX.format(errors=errors_text) + body


def _loc_to_json_pointer(loc: tuple[str | int, ...]) -> str:
    """Convert a Pydantic ``loc`` tuple to an RFC 6901 JSON Pointer.

    Empty loc → ``""`` (root). Each segment is escaped per RFC 6901:
    ``~`` → ``~0``, ``/`` → ``~1``. The model can paste the result
    verbatim into a patch op's ``path`` field.
    """
    if not loc:
        return ""
    parts = [str(p).replace("~", "~0").replace("/", "~1") for p in loc]
    return "/" + "/".join(parts)


def _format_validation_errors(errs: list[dict[str, Any]]) -> str:
    """Render ``ValidationError.errors()``, rolling identical error classes up.

    Pydantic emits one entry per error, so when the SAME validator fires on
    many list items — e.g. a presence check on seven ``/entities/N`` — its full
    multi-line explanation is repeated verbatim once per occurrence, burying
    the only per-occurrence signal (the paths) and wasting repair-prompt
    tokens.

    Errors sharing an identical ``(type, message)`` are grouped: the
    explanation is shown ONCE — the validator's own prose kept verbatim, so the
    schema stays the single source of truth — followed by a concise list of
    every JSON-Pointer path it occurs at. A lone error keeps its full prose
    with its path inline. Only byte-identical messages merge, so any
    per-occurrence variance (a differing interpolated value) keeps its own
    block and no information is lost. First-seen order is preserved.

    Markdown throughout (bold, inline-code paths, the validators' own ``###``
    headings) so it renders in Langfuse-style trace viewers. The Pydantic
    ``"Value error, "`` prefix is stripped and a blank line precedes each
    message, so a leading ``###`` heading renders as a heading instead of
    leaking as literal text. This preserves the previous renderer's contract
    for single errors (real newlines, JSON-Pointer paths, bracket numbering)
    while collapsing repeated classes.
    """
    if not errs:
        return "(no specific errors reported)"
    groups: dict[tuple[str, str], list[str]] = {}
    for err in errs:
        path = _loc_to_json_pointer(tuple(err.get("loc", ()))) or "(root)"
        key = (
            err.get("type", ""),
            err.get("msg", "").removeprefix("Value error, "),
        )
        groups.setdefault(key, []).append(path)
    blocks = []
    for i, ((err_type, msg), paths) in enumerate(groups.items(), 1):
        body = msg.strip()
        if len(paths) == 1:
            blocks.append(f"[{i}] {err_type} at `{paths[0]}`\n\n{body}")
        else:
            locs = "\n".join(f"- `{p}`" for p in paths)
            blocks.append(
                f"[{i}] {err_type} ({len(paths)} occurrences)\n\n{body}\n\n"
                f"**Occurs at these {len(paths)} paths:**\n{locs}"
            )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# JSON-Patch apply-failure feedback (shape-informed repair on an apply error)
# ---------------------------------------------------------------------------


def _apply_failure_feedback(
    doc: dict[str, Any], ops: list[Any], exc: BaseException
) -> ValueError:
    """Build a shape-enriched apply-failure error for the next patch turn.

    Locates the operation that could not be applied AND the document state
    immediately before it (earlier ops in the same patch have already been
    applied — JSON-Patch applies sequentially — so the failing op sees that
    transformed state, not the original). Resolves the deepest part of the
    relevant pointer(s) that exists in THAT state and renders the current
    value/shape there, so the model can retarget instead of re-issuing the
    same doomed op. Markdown, Langfuse-readable.
    """
    state_before, failing_op, applied_before = _find_failing_op(doc, ops)

    if failing_op is None:
        op_desc = "one of your operations"
        pointers = [""]
    else:
        op_desc = f"`{failing_op.get('op', '?')}` at `{failing_op.get('path', '')}`"
        pointers = _failure_pointers(failing_op, state_before)

    shape_blocks = []
    for pointer in pointers:
        resolved, value = _nearest_existing(state_before, _pointer_parts(pointer))
        label = _pointer_label(failing_op, pointer)
        shape_blocks.append(
            f"The deepest part of {label} `{pointer or '/'}` that currently "
            f"exists is `{resolved or '/'}`, whose current value is:\n\n"
            f"{_describe_shape(value)}"
        )

    provenance = (
        " This reflects the state AFTER your earlier operation(s) in this same "
        "patch were applied (JSON Patch applies operations in order), so a path "
        "an earlier op removed or re-indexed may no longer exist."
        if applied_before
        else ""
    )
    return ValueError(
        "\n### Your JSON Patch could not be applied\n\n"
        f"{type(exc).__name__}: {exc}\n\n"
        f"The operation that failed was {op_desc}.{provenance}\n\n"
        + "\n\n".join(shape_blocks)
        + "\n\nRe-derive your patch paths from THIS shape (shown above and in "
        "the previous JSON output), not from paths quoted in an earlier error. "
        "Note that a scope field holding `\"NONE_IN_SCOPE\"` (or an empty list) "
        "is empty and has no elements to index or append to — target the field "
        "itself, not an element within it. List indices are positional; when "
        "removing several items from one list, order the removals "
        "highest-index-first so earlier removals do not shift later indices."
    )


def _find_failing_op(
    doc: dict[str, Any], ops: list[Any]
) -> tuple[Any, dict[str, Any] | None, bool]:
    """Locate the first un-appliable op and the doc state it actually saw.

    ``jsonpatch`` applies a patch atomically and does not say *which* op
    failed. Replaying the ops one at a time against a running copy identifies
    the culprit precisely AND yields the running state immediately before it
    (the state that op was applied against). Returns
    ``(state_before_failure, failing_op, any_ops_applied_first)``.
    """
    running = doc
    applied_before = False
    for op in ops:
        op_dict = op if isinstance(op, dict) else op.model_dump()
        try:
            running = jsonpatch.JsonPatch([op_dict]).apply(running)
        except Exception:
            # Any op that will not replay is the culprit. Broad by design: this
            # runs only AFTER the atomic apply already failed, purely to build
            # feedback, so it must never itself raise (some malformed ops map
            # to TypeError/KeyError rather than a jsonpatch exception).
            return running, op_dict, applied_before
        applied_before = True
    return running, None, applied_before


def _failure_pointers(op: dict[str, Any], state: Any) -> list[str]:
    """The pointer(s) whose shape explains why ``op`` failed against ``state``.

    For move/copy there are two arms: the ``from`` source and the ``path``
    destination. Disambiguate by resolution — if the source resolves fully the
    fault is at the destination, otherwise at the source; when neither is
    clearly at fault, surface both. Every other op is anchored on ``path``.
    """
    if op.get("op") in ("move", "copy"):
        src = op.get("from", "")
        dst = op.get("path", "")
        resolved_src, _ = _nearest_existing(state, _pointer_parts(src))
        if _pointer_parts(src) and resolved_src == _render_pointer(_pointer_parts(src)):
            return [dst]  # source resolves cleanly → the destination is at fault
        return [src, dst]
    return [op.get("path", "")]


def _pointer_label(op: dict[str, Any] | None, pointer: str) -> str:
    """Name a pointer as the source/destination/target for the feedback text."""
    if op is not None and op.get("op") in ("move", "copy"):
        if pointer == op.get("from"):
            return "the source path"
        if pointer == op.get("path"):
            return "the destination path"
    return "that path"


def _pointer_parts(pointer: str) -> list[str]:
    """Decode an RFC 6901 JSON Pointer into its unescaped segments."""
    if not pointer or pointer == "/":
        return []
    return [
        seg.replace("~1", "/").replace("~0", "~")
        for seg in pointer.lstrip("/").split("/")
    ]


def _render_pointer(parts: list[str]) -> str:
    """Render unescaped segments back into an RFC 6901 JSON Pointer.

    Re-escapes ``~`` and ``/`` so a segment that contains either (a filename
    with a slash, say) produces a valid pointer the model can patch against.
    """
    if not parts:
        return ""
    return "/" + "/".join(
        seg.replace("~", "~0").replace("/", "~1") for seg in parts
    )


def _nearest_existing(doc: Any, parts: list[str]) -> tuple[str, Any]:
    """Descend ``parts`` through ``doc`` as far as they actually resolve.

    Returns ``(pointer, value)`` for the deepest existing ancestor: descent
    stops at the first segment that cannot be followed (a missing key, an
    out-of-range or ``-`` list index, or any attempt to index into a scalar —
    the sentinel-string case). This deliberately does NOT index into strings
    (RFC-pointer libraries would return a character), because the useful signal
    is exactly that the field is a scalar where the model expected a container.
    """
    cur = doc
    resolved: list[str] = []
    for part in parts:
        if isinstance(cur, dict):
            if part in cur:
                cur = cur[part]
                resolved.append(part)
                continue
            break
        if isinstance(cur, list):
            if part == "-":
                break
            try:
                idx = int(part)
            except ValueError:
                break
            if idx < 0 or idx >= len(cur):
                break
            cur = cur[idx]
            resolved.append(part)
            continue
        break
    return _render_pointer(resolved), cur


def _fenced(text: str) -> str:
    """Cap ``text`` to the shape budget and wrap it in a code fence.

    A ``json`` fence is used only when the (uncapped) content is genuine JSON;
    once truncated it is no longer valid JSON, so a plain fence is used to
    avoid mislabelling prose/partial content as JSON.
    """
    if len(text) > _SHAPE_CHAR_BUDGET:
        return "```\n" + text[:_SHAPE_CHAR_BUDGET] + "\n…(truncated)\n```"
    return "```json\n" + text + "\n```"


def _describe_shape(value: Any) -> str:
    """Render ``value`` as a budgeted markdown block for the feedback message.

    For a list the model most needs its length (the valid indices), so a
    one-line prose summary precedes the fenced value — the prose sits OUTSIDE
    the fence so it is never mislabelled as JSON. Every container's rendered
    value is capped at ``_SHAPE_CHAR_BUDGET`` so a large document cannot bloat
    the repair prompt (a 3×10k-element list would otherwise dominate it).
    """
    if isinstance(value, list):
        if not value:
            return "an empty list `[]`"
        summary = (
            f"a list with {len(value)} element(s) "
            f"(valid indices 0..{len(value) - 1}); value:"
        )
        return summary + "\n\n" + _fenced(json.dumps(value, ensure_ascii=False))
    return _fenced(json.dumps(value, ensure_ascii=False))
