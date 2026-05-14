---
manager_sessions:
  - id: 019e1e72-9606-72da-ad52-3c2942e9f6cd
    name: trustcall-multitool-response
    role: intent
    authored_at: 2026-05-14T12:31:58.797Z
---

# Intent: stitchcall

A small library for **single-schema** structured LLM extraction with native
JSON-mode initial extract and JSON-Patch (RFC 6902) repair on validation
failure. Complements [trustcall](https://github.com/hwchase17/trustcall) —
which remains the right tool for multi-schema, multi-call, and
patch-existing-instances flows. `stitchcall` is for the more common case
where you bind one schema, expect one validated object back, and want
surgical repair when the model's first try is *almost* right.

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

The trade gets worse on weaker models. In our benchmarks, on a
large extraction prompt with a wrapping schema that requires
N-element list invariants, current trustcall + tools achieves ~80 % success
per trial on Gemini 3 Flash; switching the *same schema* to native JSON
mode + JSON-Patch repair achieves 100 % success in roughly half the
wall-time (see "Empirical evidence" below).

## What stitchcall is

A `~250-line` extractor with a small public API:

```python
extractor = Extractor(llm, MySchema, max_attempts=5,
                      max_validation_error_weight=40)
result = await extractor.ainvoke(messages,
                                 validation_context={...},
                                 callbacks=[...])
# result.value : MySchema (validated)
# result.attempts, result.was_re_extracted, result.raw_messages
```

It does exactly four things:

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

## What stitchcall is **not**

- **Not a replacement for trustcall.** Multi-schema, multi-call,
  patch-existing-instances flows belong in trustcall.
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
consistency**, not just for validation. Stitchcall preserves it; the
transport changes, the schema doesn't.

### 3. JSON Patch (RFC 6902), not custom PatchDoc

Trustcall uses a custom `PatchDoc` schema because it needs `tool_call_id`
correlation back to the original tool call being patched. Stitchcall has
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

If you write a custom `model_validator` and want stitchcall's patch loop
to work well for it, *include the JSON Pointer in your error message*.
Stitchcall does not synthesise pointers from validator messages — that
contract is the user's.

## Empirical evidence

The design was validated against a non-trivial workload:
a single Pydantic schema with cross-field invariants over a list of N
items, prompted with a ~500k-token user message, run on
`gemini-3-flash-preview` at temperature 1.

| Shape | n | success | mean wall (s) | mean attempts |
|---|---|---|---|---|
| trustcall + tools (wrapping schema) | 30 | 80 % (24/30) | 485 | 2.63 |
| trustcall + tools (flat multi-call) | 30 |  3 % (1/30)  | 268 | 1.90 |
| **stitchcall** (wrapping schema, JSON mode) | **30** | **100 % (30/30)** | **246** | **1.55** |

Same fixture, same model, same temperature in all three. The wrapping
schema is preserved across rows 1 and 3; only the transport (and repair
primitive) change.

Notable findings:

- **JSON-mode binding works on a non-trivial schema as-is.** The schema
  used here had recursive `$defs`, `anyOf` unions, and a
  sentinel-forced field (`(field redacted)` accepting either `str` or
  `list[str]`). LangChain's `with_structured_output(method="json_schema")`
  accepted it without modification.
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
  consume the full `max_attempts` budget; stitchcall's patch loop converges
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

The stitchcall equivalent:

```python
extractor = stitchcall.Extractor(
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
- **Async vs sync.** Only `ainvoke` is exposed. A sync `invoke` is trivial
  to add if anyone wants it; not in v0.0.1 to keep the surface minimal.

## Rejected alternatives

- **Adapt trustcall in place.** Considered and rejected — JSON mode is
  incompatible with trustcall's `tool_call_id`-correlated patch repair,
  multi-schema routing, and `ToolMessage`-based history replay. Building
  a small sibling tool preserves trustcall's value where it is uniquely
  useful (multi-schema, patch-existing-instances) and removes its
  overhead where it is not.
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
