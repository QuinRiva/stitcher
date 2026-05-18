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
from typing import Any, Callable, NamedTuple, TypeAlias, Union, get_args, get_origin

import jsonpatch
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from stitcher.exceptions import AggregatedValidationError


ValidationContext: TypeAlias = dict[str, Any]


class AttemptInfo(NamedTuple):
    """Per-attempt observability payload, fired by ``Extractor.on_attempt``.

    Diverges from trustcall's ``AttemptInfo`` in one place — ``error: BaseException
    | None`` instead of ``validation_errors: list[str]``. Adjudicators classify
    failures by ``error.errors()`` structure (``type``, ``loc``, ``ctx["error"]``
    for nested ``AggregatedValidationError``) — stitcher pre-formatting to strings
    would throw away the structure right where the consumer needs it. Format the
    exception however your wide-log expects.

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
    attempt_number: int
    parsed: dict[str, Any] | None
    error: BaseException | None
    is_success: bool


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


class TokenUsage(NamedTuple):
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
    input_tokens: int
    cached_input_tokens: int
    reasoning_tokens: int
    output_payload_tokens: int


class Metadata(NamedTuple):
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
    initial: TokenUsage
    total: TokenUsage
    duration_seconds: float


class Result(NamedTuple):
    """Result of a successful ``ainvoke`` or ``aupdate`` call."""
    value: BaseModel
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
_PATCH_REPAIR_PREFIX = (
    "## Validation Errors\n\n"
    "The previous JSON output failed validation. The errors below are "
    "symptoms — diagnose what your previous output got wrong, then patch "
    "the root cause, not just the surface message.\n\n"
    "```\n{errors}\n```\n\n"
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
        # include_raw=True returns {"raw": AIMessage, "parsed": ..., "parsing_error":
        # ...} instead of the bare parsed value. This is the idiomatic LangChain
        # path for keeping the real AIMessage (with usage_metadata,
        # response_metadata, finish_reason, provider message id) while still
        # getting the parsed structured output. Stitcher uses "parsed" for its
        # loop logic and appends "raw" to raw_messages — same position as the
        # synthesized AIMessage used to occupy, now with the real metadata
        # attached. See ``_aggregate_usage`` for how metadata is rolled up.
        #
        # ``langsmith:hidden`` tag demotes the LangChain include_raw plumbing
        # (RunnableSequence / RunnableParallel<raw> / RunnableWithFallbacks /
        # RunnableAssign / JsonOutputParser / internal RunnableLambdas) to
        # ``level="DEBUG"`` in Langfuse, which the Langfuse trace UI hides by
        # default. The actual LLM generation is structurally exempt from the
        # demotion (Langfuse's ``__on_llm_action`` doesn't apply level-from-tags),
        # so the model call stays at DEFAULT and visible. The user-meaningful
        # ``initial`` / ``patch`` labels also stay visible because they sit on
        # outer ``RunnableLambda`` wrappers in ``_patch_loop`` and
        # ``_run_patch_turn`` that are NOT tagged hidden. See
        # https://github.com/langfuse/langfuse-python/pull/1077.
        self._initial_llm = llm.with_structured_output(
            schema=self._schema_json,
            method="json_schema",
            include_raw=True,
        ).with_config(tags=["langsmith:hidden"])
        self._patch_llm = llm.with_structured_output(
            schema=JsonPatchResponse,
            method="json_schema",
            include_raw=True,
        ).with_config(tags=["langsmith:hidden"])

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
        # the parent (set in ainvoke/aupdate), not on these children. Each
        # child name is set on a thin ``RunnableLambda`` wrapper around the
        # LLM call (see ``_invoke_initial`` / ``_invoke_patch`` below) so the
        # label sits at Langfuse's DEFAULT level — visible — while the
        # ``langsmith:hidden``-tagged ``with_structured_output`` plumbing
        # underneath collapses to DEBUG.
        initial_config: dict[str, Any] = {"run_name": "initial"}
        patch_config: dict[str, Any] = {"run_name": "patch"}

        t_start = time.monotonic()
        raw_messages: list[BaseMessage] = list(original_messages)
        happy_path_messages: list[AIMessage] = []
        attempts = 0
        was_re_extracted = False
        last_validation_error: BaseException | None = None

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

        raise RuntimeError(
            f"Extractor exhausted {self.max_attempts} attempts. "
            f"Last validation error: {last_validation_error!r}"
        )

    async def _invoke_initial(
        self,
        messages: list[BaseMessage],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke the initial-extract chain inside a thin labeling wrapper.

        The wrapper exists solely to put the user-meaningful ``initial`` label
        at Langfuse's DEFAULT level (visible). Without it, the label would
        attach to the outermost span of the ``with_structured_output`` chain,
        which is tagged ``langsmith:hidden`` and so renders at DEBUG (hidden
        by default in Langfuse's trace UI). We also re-pass ``config`` to the
        inner ``ainvoke`` so the per-call run_name reaches the LLM via
        context vars (the FakeLLM-based tests assert on this).
        """
        async def _step(msgs: list[BaseMessage]) -> dict[str, Any]:
            return await self._initial_llm.ainvoke(msgs, config=config)
        return await RunnableLambda(_step).ainvoke(messages, config=config)

    async def _invoke_patch(
        self,
        history: list[BaseMessage],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke the patch chain inside a thin labeling wrapper.

        Mirror of ``_invoke_initial`` for the patch path — same reason, same
        shape. The wrapper keeps the ``patch`` label visible at DEFAULT level
        in Langfuse while the include_raw plumbing underneath stays hidden.
        """
        async def _step(hist: list[BaseMessage]) -> dict[str, Any]:
            return await self._patch_llm.ainvoke(hist, config=config)
        return await RunnableLambda(_step).ainvoke(history, config=config)

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
        """One patch turn: build the request, call the model, apply the patch.

        Returns ``(new_prev_dict, error, applied_message)``:

        - ``error is None``: patch applied. ``new_prev_dict`` is the patched
          dict; ``applied_message`` is the model's real ``AIMessage`` (now on
          the happy path — its output is reflected in the patched state).
        - ``error`` set: patch could not be applied, either because the model
          returned a ``parsing_error`` (response didn't match the
          ``{reasoning, operations}`` shape) or because ``jsonpatch.apply``
          rejected the ops (bad pointer, malformed op). ``new_prev_dict`` is
          the *unchanged* input; ``applied_message`` is ``None`` (the message
          is still on ``raw_messages`` for debugging, just not on the happy
          path). The caller feeds ``error`` back as the next turn's
          validation trigger.

        Mutates ``raw_messages`` in place: appends the outgoing
        ``HumanMessage`` and the model's ``AIMessage`` (whether the call
        succeeded or hit a ``parsing_error``).
        """
        patch_msg = HumanMessage(content=patch_prompt)
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
            # The model returned something that didn't match JsonPatchResponse
            # (missing operations, wrong types, prose instead of JSON, ...).
            # Symmetric with a jsonpatch-apply failure: feed back, consume an
            # attempt, let the model retry the patch protocol.
            return prev_dict, ValueError(
                f"Your response did not match the required "
                f"{{reasoning, operations}} shape: "
                f"{type(parse_error).__name__}: {parse_error}. "
                "Reissue with both fields populated."
            ), None
        patch_resp: JsonPatchResponse = patch_result["parsed"]
        ops = patch_resp.operations
        try:
            new_dict = jsonpatch.JsonPatch(ops).apply(prev_dict)
            return new_dict, None, patch_raw
        except (jsonpatch.JsonPatchException, jsonpatch.JsonPointerException) as e:
            return prev_dict, ValueError(
                f"Your JSON Patch could not be applied: {type(e).__name__}: {e}. "
                "Re-issue a corrected patch against the previous JSON output."
            ), None

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
    """Render Pydantic ``ValidationError.errors()`` as a readable text block.

    Replaces the previous ``json.dumps(errors)`` rendering, which escaped
    every ``\\n`` in user-supplied validator messages into literal ``\\n``
    text — turning multi-line messages (markdown, examples, hints) into
    unreadable single-line walls. Real newlines are preserved here.

    Each error gets a one-line header (index, JSON Pointer path, Pydantic
    error type) followed by the message verbatim. Errors are separated by
    a blank line.
    """
    if not errs:
        return "(no specific errors reported)"
    blocks = []
    for i, err in enumerate(errs, 1):
        path = _loc_to_json_pointer(tuple(err.get("loc", ())))
        err_type = err.get("type", "")
        msg = err.get("msg", "")
        header = f"[{i}] path {path or '(root)'} \u2014 {err_type}:"
        blocks.append(f"{header}\n{msg}")
    return "\n\n".join(blocks)
