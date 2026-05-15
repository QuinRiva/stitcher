---
manager_sessions:
  - id: 019e1e72-9606-72da-ad52-3c2942e9f6cd
    name: trustcall-multitool-response
    role: intent
    authored_at: 2026-05-14T12:31:58.797Z
---

# Intent: stitcher

A small library for **single-schema** structured LLM extraction with native
JSON-mode initial extract and JSON-Patch (RFC 6902) repair on validation
failure, plus a single-instance update primitive (`aupdate`) that reuses
the same patch loop. Complements [trustcall](https://github.com/hwchase17/trustcall)
— which remains the right tool for multi-schema, multi-call, and
multi-instance patch-existing flows. `stitcher` is for the more common
case where you bind one schema, expect one validated object back, and want
surgical repair when the model's first try is *almost* right — or where
you have a single prior instance and want to apply a user-described update
to it.

## Why this exists

Trustcall's design committed to tool-calling as the unifying transport
because it has to cover several use cases:

- multi-schema heterogeneous extraction (`tools=[A, B, C]`)
- multi-call extraction (`tool_choice="any"` with many calls back)
- patch-existing-instances (`existing=…` keyed by `tool_call_id`)
- single-schema extraction (degenerate case of the above)

For the single-schema case, tool-calling is paying for surface area you
don't use. It also opens specific failure modes — the model emitting the
same tool call twice in one turn, or emitting one valid + one empty
duplicate — that don't exist in JSON mode at all.

The trade gets worse on weaker models. On a large extraction prompt with
a wrapping schema that requires N-element list invariants, trustcall +
tools at ~80 % per-trial success on Gemini 3 Flash improved to 100 % at
roughly half the wall-time when the *same schema* was switched to native
JSON mode + JSON-Patch repair (see "Empirical evidence" below).

## What stitcher is

An extractor with a small public API — two operations sharing one
patch-loop mechanism:

```python
extractor = Extractor(llm, MySchema, max_attempts=5,
                      max_validation_error_weight=40)

# Operation A: produce a fresh validated instance
result = await extractor.ainvoke(messages,
                                 validation_context={...},
                                 callbacks=[...],
                                 run_name="my_call_site")

# Operation B: transform a prior instance per a user intent
result = await extractor.aupdate(existing=prior,
                                 messages=messages,
                                 validation_context={...},
                                 callbacks=[...],
                                 run_name="my_call_site")

# result.value : MySchema (validated)
# result.attempts, result.was_re_extracted, result.raw_messages
```

`ainvoke` does exactly four things:

1. **Initial extract via native JSON mode.**
   `llm.with_structured_output(schema=MySchema.model_json_schema(),
   method="json_schema")`. Returns a parsed dict.
2. **Validate via `MySchema.model_validate(data, context=…)`.** The user's
   validation context flows in unchanged so cross-field validators that
   depend on it (e.g. "every input id must be accounted for") work normally.
3. **On `ValidationError`, ask the model for a JSON Patch.** The repair
   conversation is a plain `HumanMessage` chain containing the previous
   JSON output, the validation errors, and an instruction to return RFC
   6902 patch operations. Apply via the `jsonpatch` library, re-validate.
4. **Catastrophic-error re-extract.** If the summed error weight on a
   single validation pass exceeds `max_validation_error_weight`, abandon
   the patch loop and start a fresh extract from the original system+user.
   Bounded by `max_attempts`. The weight-summing logic supports
   `AggregatedValidationError(count=N)` so user validators can declare
   "this single error covers N underlying problems".

`aupdate` reuses the patch loop with a different first-turn shape:

1. **Skip the initial extract**; seed `prev_dict` from `existing`
   (`model_dump(mode="json")` for Pydantic instances, the dict directly
   for raw dicts — jsonpatch returns a fresh object on apply, so no copy
   is needed).
2. **First patch turn carries no validator framing.** The prior is presented
   to the model and the same patch-instruction template as `ainvoke`'s
   repair turn is used — minus the `<errors>` prefix. The user's update
   intent in `messages` directs *what* to patch; the prompt only specifies
   *how* (JSON Patch / JSON Pointer format). Schema stays on the
   `with_structured_output` binding, as usual — not stuffed into the
   prompt.
3. **Subsequent turns are validator-driven** — the validator-failure
   prefix is prepended to the same patch template, identical to
   `ainvoke`'s repair turns.
4. **Always at least one LLM call.** No zero-LLM-call optimisation: that
   would conflate update with verify-and-repair, two different operations.
   Callers who want verify-and-repair should pre-validate themselves and
   only invoke `aupdate` when an update is actually requested.
5. **No catastrophic-re-extract path** — there's no fresh-extract
   fallback to revert to. Bounded by `max_attempts`; exhaustion raises.

### Validation context contract

On every `model_validate` call (in either `ainvoke` or `aupdate`), stitcher
merges one library-supplied key into the user's `validation_context`:

- `attempt_count: int` — 1 on the first validation, incrementing by one
  per validation attempt. Mirrors trustcall's contract so validators can
  implement first-attempt-strict / later-lenient patterns (e.g. an
  adjudicator that pushes back on the LLM once but accepts its judgment on
  retry).

User-supplied keys win on collision — callers can pass
`validation_context={"attempt_count": 5}` to simulate a later attempt in
tests, also matching trustcall.

### Per-attempt observability hook

`Extractor(llm, schema, ..., on_attempt=callable)` registers an
observability hook fired once per validation attempt with an
`AttemptInfo(attempt_number, parsed, error, is_success)`. Mirrors
trustcall's `on_attempt` for migration; the canonical use case is
wide-logging per-attempt failure classifications (validation_failure
vs patch_apply_failure vs success) for observability tools.

Fires at three points inside the patch loop:

- After successful `model_validate` — `is_success=True`, `error=None`,
  `parsed=<the validated dict>`.
- After failed `model_validate` — `is_success=False`,
  `error=<the ValidationError>`, `parsed=<the dict that failed>`. Callers
  classify by inspecting `error.errors()` for `type`, `loc`,
  `ctx["error"]` (the latter carries any nested
  `AggregatedValidationError`).
- After a patch the model returned could not be applied (bad pointer,
  malformed op) — `is_success=False`, `parsed=None`, `error=<a synthetic
  ValueError>`. `isinstance(error, ValidationError)` discriminates
  validation-failures from patch-apply-failures.

Sync (`def`) and async (`async def`) callables both work; stitcher
awaits if the return is a coroutine.

Two deliberate divergences from trustcall, both for the same reason —
stitcher passes the consumer the real underlying object rather than a
flattened/synthesized stand-in:

- `AttemptInfo` does **not** carry an `ai_message` field. Stitcher uses
  `with_structured_output()`, which returns a parsed dict and hides the
  underlying `AIMessage` — synthesizing a fake one would lose
  `response_metadata` / `usage_metadata` and silently break callers who
  depend on them. If you need raw response metadata (token counts,
  finish_reason, etc.), pass a `BaseCallbackHandler` via `callbacks=`
  instead.
- `AttemptInfo.error` is a raw `BaseException`, not a `list[str]` of
  pre-formatted messages. Adjudicators classify failures by structured
  fields (`type`, `loc`, `ctx["error"]`) — stitcher pre-formatting to
  strings would throw away the structure right where the consumer needs
  it. Format however your wide-log expects.

## What stitcher is **not**

- **Not a replacement for trustcall.** Multi-schema, multi-call, and
  *multi-instance* patch-existing flows belong in trustcall — those flows
  need `tool_call_id` correlation, multi-schema routing, and
  `ToolMessage`-based history that stitcher's JSON-mode + HumanMessage
  shape cannot express. **Single-instance** update is supported via
  `aupdate` (see above) because the architectural objection to trustcall's
  flow is specific to the multi-instance case.
- **Not a multi-instance update primitive.** `aupdate` accepts exactly
  one `existing` object. Multi-instance patching
  (`existing={id1: ..., id2: ...}`) stays in trustcall by design — do not
  add it here.
- **Not a verify-and-repair primitive.** `aupdate` always runs at least
  one LLM call. Callers who want "validate this; only call the model if
  it's broken" should pre-validate with `schema.model_validate(...)` and
  branch on `ValidationError` themselves — stitcher does not conflate
  the two operations.
- **Not a tool-calling primitive.** No `bind_tools`, no `tool_choice`, no
  `tool_call_id` correlation, no `ToolMessage`. Single response, repair via
  JSON Patch over a HumanMessage chain.
- **Not a JSON-Schema-only solver.** The repair loop relies on Pydantic
  v2's `ValidationError.errors()` shape (specifically the JSON-Pointer-like
  `loc` paths) to give the model precise targets for its patches. v2 is
  required.
- **Not a callback handler.** Callers pass their LangChain callbacks in via
  the `callbacks` parameter; lifecycle (e.g. `Langfuse.shutdown()` on
  process exit) is the caller's responsibility.

## Key design choices

### 1. Native JSON mode for the initial extract

Tool-calling for single-schema extraction is a misuse — the model has to
emit a `functionCall` part wrapping a JSON object that you then have to
unwrap. Some providers (Gemini in particular) have observable failure modes
like "two `functionCall` parts in one response" that don't exist in JSON
mode at all because there's only one response slot. Going to JSON mode
sidesteps that class of failure entirely.

### 2. Wrapping schema preserved (not flattened to multi-call)

A natural-looking alternative is "stop using the wrapper, bind the leaf
schemas with `tool_choice="any"` and let the model emit N independent
calls". This is *much* worse than the wrapping shape on weaker models —
the model can't reliably maintain "which items have I already claimed"
across many independent tool calls. In our testing the flat-multi-call
shape achieves ~3 % success on the same workload where the wrapping shape
achieves ~80 %.

The lesson: the wrapping schema is **load-bearing for the model's own
consistency**, not just for validation. Stitcher preserves it; the
transport changes, the schema doesn't.

### 3. JSON Patch (RFC 6902), not custom PatchDoc

Trustcall uses a custom `PatchDoc` schema because it needs `tool_call_id`
correlation back to the original tool call being patched. Stitcher has
only one extraction in flight, so RFC 6902 with the standard `jsonpatch`
library is sufficient. No custom primitive needed.

### 4. Repair conversation is plain HumanMessage

History on a repair turn is exactly:

```
[
  SystemMessage(original_system),
  HumanMessage(original_user),
  AIMessage(content=json.dumps(prev_dict)),
  HumanMessage(content=patch_prompt),
]
```

No `tool_calls`, no `ToolMessage`. This is the shape Gemini's JSON-mode
endpoint accepts on history replay; it also avoids the
`tool_call_id`-correlation problem trustcall's repair loop has to solve.

### 5. JSON Pointer paths are the load-bearing detail

The patch loop only works because Pydantic v2's `ValidationError.errors()`
emits `loc` tuples that translate cleanly into JSON Pointers, and because
your validator messages can include those pointers explicitly when they
matter. The model uses those pointers verbatim in its `path` operations.
Without that handover the patch loop would degenerate into "edit the whole
document".

If you write a custom `model_validator` and want stitcher's patch loop
to work well for it, *include the JSON Pointer in your error message*.
Stitcher does not synthesise pointers from validator messages — that
contract is the user's.

## Stringified-JSON normalisation (automatic)

Before every `model_validate` call, stitcher walks the parsed dict
guided by the target Pydantic schema and re-parses any string value
that sits in a non-string structural slot (a slot annotated as
`BaseModel`, `list[T]`, `dict[K, V]`, etc.). The motivating case is
Gemini's soft-enforcement of nested object types: a slot typed
`list[Item]` occasionally receives `["{...}", "{...}"]` instead of
`[{...}, {...}]`. Without the normaliser, stitcher pays a patch-turn
round-trip to repair (~$0.09 / ~20s on Gemini per affected run); with
it, validation passes on the first attempt.

Safe-by-construction: only fires when ALL of:

1. The value is a `str`
2. The annotation requires a structural type (`BaseModel`, `list[X]`,
   `dict[K, V]`, or one of those inside a `Union`)
3. The string starts with `{` or `[` after stripping whitespace
4. `json.loads` succeeds and the parsed value matches the expected
   container (`dict` for object slots, `list` for arrays)

Fields legitimately typed `str` are left alone even if the content
looks like JSON — the schema explicitly allows strings there.

This is the only post-`json.loads` normalisation stitcher applies. It
was selected after a survey of BAML's Schema-Aligned Parser (whose
other text-level recoveries are unreachable from a parsed dict) and
Instructor (which has effectively no normalisation — it delegates to
Pydantic non-strict mode and retries on failure). Pydantic non-strict
mode already covers the cheap coercions (`"42"`→`42`, `"true"`→`True`,
etc.); BAML's lossy recoveries (silently drop bad array items, round
3.5→4, etc.) are out of scope for stitcher's design.

No opt-out flag is exposed. If a future use case shows a real false
positive, an opt-out can be added then — for now the safety property
above makes one premature.

## Writing LLM-friendly validators

The patch loop only works as well as the validation errors it feeds back
to the model. A bare `ValueError("invalid")` from your validator gives
the model nothing to act on; a precise error with a JSON Pointer path
and a hint about the root cause lets the model produce a one-op patch.
These patterns are not enforced by stitcher — they're guidance for
schema authors who want the patch loop to converge quickly.

### Include JSON Pointer paths in `model_validator(mode="after")` messages

Field validators (`@field_validator`) get a useful `loc` automatically.
Model-level validators (`@model_validator(mode="after")`) don't — their
`loc` is empty because they fire at the model level, not a field level.
When a model validator finds something wrong, include the JSON Pointer
in the error message text:

```python
@model_validator(mode="after")
def _check_unique(self):
    seen: dict[str, int] = {}
    for i, item in enumerate(self.items):
        if item.id in seen:
            raise ValueError(
                f"Duplicate id '{item.id}' at /items/{i} "
                f"(first seen at /items/{seen[item.id]}). "
                f"Patch suggestion: 'remove' the duplicate at /items/{i}."
            )
        seen[item.id] = i
    return self
```

### Aggregate, don't enumerate

If your validator finds N problems, raise one combined error — not N
separate ones. Pydantic surfaces only the first raised error per
validator, so enumerating means you lose N-1 problems on each pass and
the patch loop wastes attempts. When the N errors share a common cause,
use `AggregatedValidationError(message, count=N)` to declare the true
weight to the catastrophic-re-extract threshold.

```python
@model_validator(mode="after")
def _check_all(self):
    errors = []
    for i, item in enumerate(self.items):
        if item.amount_cents < 0:
            errors.append(
                f"Negative amount at /items/{i}/amount_cents: {item.amount_cents}. "
                f"Patch suggestion: 'replace' with a non-negative integer."
            )
    if errors:
        raise ValueError(" | ".join(errors))
    return self
```

### Pre-empt the empty-object anti-pattern with a `mode="before"` validator

When the model needs to add an object to a list, it sometimes returns an
empty `{}` as a placeholder intending to fill it later — but JSON Patch
is declarative, not transactional. The placeholder fails Pydantic
immediately (one "field required" error per missing field) and the
follow-up `replace` ops have no path to target. The model often retries
with another `{}`.

Short-circuit by adding a `mode="before"` check on the child model that
detects sparse input and raises one message including the expected
shape:

```python
class Item(BaseModel):
    id: str
    name: str
    active: bool

    @model_validator(mode="before")
    @classmethod
    def _reject_empty(cls, data):
        if isinstance(data, dict) and len(data) < 2:
            raise ValueError(
                "Empty placeholder rejected. The patch `value` must contain "
                "the COMPLETE object. Required fields: id (str), name (str), "
                "active (bool). Use a single `add` op with the complete value, "
                "not an empty placeholder followed by `replace` ops."
            )
        return data
```

### Read `attempt_count` for first-attempt-strict / later-lenient patterns

Stitcher injects `attempt_count: int` into `validation_context` on every
validation pass (1 on the first, 2 after the first patch, ...). Use it
for invariants where you want to challenge the model's first answer but
accept its judgment on retry — prevents infinite repair loops on
contested judgments while still giving the LLM a chance to reconsider.

```python
@model_validator(mode="after")
def _adjudicate(self, info: ValidationInfo):
    if (
        info.context["attempt_count"] == 1
        and self.values_equivalent
        and self.unresolved_residuals
    ):
        raise ValueError(
            "Marked values_equivalent=True but unresolved_residuals is non-empty. "
            "Confirm by leaving values_equivalent=True on retry, or update it."
        )
    return self
```

### Avoid `default_factory=list` on required-shape list fields

A subtle Pydantic + Gemini interaction: `Field(default_factory=list)`
doesn't emit a JSON Schema `default: []`, so the model may treat the
field as optional. Some models (Gemini in particular) then skip the
field entirely and put the *next* field's value in its slot, producing
type-contamination errors that look unrelated to the actual root cause.

- **Bad:** `items: list[Item] = Field(default_factory=list)`
- **Good:** `items: list[Item] = Field(...)` — the model must explicitly
  produce `[]`.

## Empirical evidence

The design was validated against a single non-trivial workload: one
Pydantic schema with cross-field invariants over a list of N items, a
long user message, run on `gemini-3-flash-preview` at temperature 1, 30
trials per shape.

| Shape | n | success | mean wall (s) | mean attempts |
|---|---|---|---|---|
| trustcall + tools (wrapping schema) | 30 | 80 % (24/30) | 485 | 2.63 |
| trustcall + tools (flat multi-call) | 30 |  3 % (1/30)  | 268 | 1.90 |
| **stitcher** (wrapping schema, JSON mode) | **30** | **100 % (30/30)** | **246** | **1.55** |

Same fixture, same model, same temperature in all three. The wrapping
schema is preserved across rows 1 and 3; only the transport (and repair
primitive) change.

Notable findings:

- **JSON-mode binding works on a non-trivial schema as-is.** The schema
  used had recursive `$defs`, `anyOf` unions, and a sentinel-forced field
  accepting either `str` or `list[str]`. LangChain's
  `with_structured_output(method="json_schema")` accepted it without
  modification.
- **The validator's JSON-Pointer paths drive the patch loop.** Trials that
  succeeded on a patch turn used patches whose `path` values came directly
  from the validator's error text. Single-op patches (one `remove` to fix
  a duplicate, or one `add` to fix a missing item) were the modal repair
  shape.
- **The catastrophic-re-extract path was never triggered** in 30 trials.
  Common error weights stayed in the 1–5 range; the threshold of 40 is
  reserved for genuinely-broken initial extractions and was correctly
  dormant.
- **Wall-time saved is mostly from avoided retries.** Trustcall's failures
  consume the full `max_attempts` budget; stitcher's patch loop converges
  in 1–3 attempts on every observed case.

## Migration from trustcall (single-schema call sites)

If you have a trustcall call site that looks like this:

```python
extractor = trustcall.create_extractor(
    llm,
    tools=[MySchema],
    tool_choice="MySchema",
    max_validation_error_weight=40,
)
response = await extractor.ainvoke({
    "messages": messages,
    "validation_context": ctx,
})
my_schema_instance = response["responses"][0]
```

The stitcher equivalent:

```python
extractor = stitcher.Extractor(
    llm,
    schema=MySchema,
    max_validation_error_weight=40,
)
result = await extractor.ainvoke(
    messages,
    validation_context=ctx,
)
my_schema_instance = result.value
```

The two are not drop-in compatible (different return shape) but the
mechanics carry across cleanly.

## Open questions

- **Provider coverage.** The library is provider-neutral in shape but only
  empirically validated against Gemini 3 Flash and Pro. OpenAI and
  Anthropic should work via their respective LangChain integrations'
  `with_structured_output(method="json_schema")` paths; first-class
  testing on those providers is a follow-up.
- **Streaming.** Not supported in v0.0.1. The patch loop architecture is
  not obviously friendly to streaming (you have to wait for the full
  initial JSON to validate before deciding whether to repair); deferred
  unless evidence suggests demand.
- **Async vs sync.** Only async (`ainvoke`, `aupdate`) is exposed. Sync
  variants are trivial to add if anyone wants them; not exposed yet to
  keep the surface minimal.

## Rejected alternatives

- **Adapt trustcall in place.** Considered and rejected — JSON mode is
  incompatible with trustcall's `tool_call_id`-correlated patch repair,
  multi-schema routing, and `ToolMessage`-based history replay. Building
  a small sibling tool preserves trustcall's value where it is uniquely
  useful (multi-schema, multi-instance patch-existing) and removes its
  overhead where it is not. (Single-instance patch-existing is in scope
  for stitcher — see `aupdate` — because the architectural objection
  above is multi-instance-specific.)
- **Flat multi-call extraction** (`tools=[A, B], tool_choice="any"`).
  Empirically falsified above: 3 % vs 80 % success at fixed schema and
  workload. The model cannot maintain cross-call consistency without a
  schema-level surface to coerce it onto.
- **Reconciliation LLM call over divergent outputs.** Extra round trip,
  new failure surface, and doesn't apply at all in JSON mode (only one
  initial extract; no possibility of duplicate same-schema responses).
- **Custom PatchDoc primitive** (mirroring trustcall). RFC 6902 is the
  standard; the `jsonpatch` Python library is mature; we have no
  `tool_call_id`-correlation problem to justify a non-standard primitive.
- **Specialised "missing items" handling at v0.** The wrapping schema's
  validators already produce informative errors. Adding a specialised
  continuation path before there is evidence the model struggles to patch
  list-membership errors is premature.
