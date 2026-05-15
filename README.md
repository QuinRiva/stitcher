# stitchcall

> Trust the model to be mostly right. Stitch the gaps.

`stitchcall` is a small Python library for **single-schema** structured
extraction with LLMs. It uses the model's native JSON-mode endpoint for the
initial extract and asks the model for a **JSON Patch (RFC 6902)** when the
output fails Pydantic validation — repairing only the parts that are wrong
instead of regenerating the whole object.

It complements [trustcall](https://github.com/hwchase17/trustcall): trustcall
handles tool-calling, multi-schema, and *multi-instance* patch flows
(`existing={id1: ..., id2: ...}` keyed by `tool_call_id`); `stitchcall`
handles the more common single-schema cases — "give me one Pydantic-validated
object, and please patch it if it's broken" (`ainvoke`) and "apply this
update to one existing instance" (`aupdate`) — with a smaller surface and
(in our testing) a higher success rate at lower wall time on weaker models.

## Install

```bash
pip install stitchcall
```

Requires Python ≥ 3.11.

## Quick start

```python
from pydantic import BaseModel, model_validator
from langchain_google_genai import ChatGoogleGenerativeAI
from stitchcall import Extractor, AggregatedValidationError


class Invoice(BaseModel):
    supplier: str
    line_items: list[str]
    total_cents: int

    @model_validator(mode="after")
    def _coherent(self):
        if not self.line_items:
            raise AggregatedValidationError(
                "invoice has no line items", count=1
            )
        return self


llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
extractor = Extractor(llm, Invoice)

result = await extractor.ainvoke([
    {"role": "system", "content": "Extract invoice data as JSON."},
    {"role": "user",   "content": "<paste the invoice text here>"},
])

print(result.value)        # Invoice instance
print(result.attempts)     # how many LLM calls (initial + patches + re-extracts)
print(result.was_re_extracted)
```

## How the patch loop works

1. **Initial extract** via `llm.with_structured_output(method="json_schema")`
   — the model returns a JSON object conforming (loosely) to your schema.
2. **Validate** via `schema.model_validate(parsed_dict, context=…)` — using
   the user-supplied `validation_context`, so any cross-field validators that
   depend on it run normally.
3. On `ValidationError`:
   - If the cumulative error weight exceeds `max_validation_error_weight`,
     **re-extract from scratch** (the validator says the model produced
     nonsense; patching nonsense is a worse use of tokens than starting
     over).
   - Otherwise, send a **plain `HumanMessage`** containing the previous JSON
     and the validation errors, asking for a JSON Patch back. Apply the
     patch via the `jsonpatch` library, re-validate, repeat.
4. Bounded by `max_attempts`. Returns a `Result(value, attempts,
   was_re_extracted, raw_messages)` on success; raises `RuntimeError` on
   exhaustion.

The repair loop uses no `tool_calls` and no `tool_call_id` correlation —
it's a pure conversation history of `[system, user, AIMessage(prev_json),
HumanMessage(error+patch_request)]`, which sidesteps the JSON-mode
history-compatibility constraints of providers like Gemini.

## When to use this vs. trustcall

| Use case | Tool |
|---|---|
| Single Pydantic schema, single response, JSON-Patch repair | **stitchcall** (`ainvoke`) |
| Update a single existing instance per a user intent | **stitchcall** (`aupdate`) |
| Multiple schemas, model decides which to call | **trustcall** |
| Multi-call extraction (one schema, N invocations) | **trustcall** |
| Patch *multiple* existing instances in one call (`existing=…` keyed by `tool_call_id`) | **trustcall** |
| Tool calling as part of an agent loop (not extraction) | LangChain / LangGraph directly |

If you only ever bind one schema with `tool_choice="<that schema name>"`,
`stitchcall` does the same job with less surface area and avoids the
duplicate-tool-call class of failures that comes with tool-mode binding.

## Updating an existing instance

`aupdate` patches a prior instance per a user intent expressed in messages.
The initial JSON-mode extract is skipped — the existing instance seeds the
patch loop directly:

```python
from langchain_core.messages import SystemMessage, HumanMessage

current_invoice = Invoice(supplier="Acme", line_items=["widget"], total_cents=1500)

result = await extractor.aupdate(
    existing=current_invoice,
    messages=[
        SystemMessage(content="You modify invoice records."),
        HumanMessage(content="Add a line item 'shipping' worth 250 cents."),
    ],
)
print(result.value)        # Invoice with the new line item, validated
print(result.attempts)     # patch turns until validation passed
```

The first turn asks the model for a JSON Patch driven by the user's intent;
subsequent turns (if validation fails) are validator-driven exactly as in
`ainvoke`.

## Design rationale

For the long-form "why JSON Patch and not just retry, why a wrapping
schema and not flat multi-call, why not adapt trustcall directly", see
[`docs/intent.md`](docs/intent.md).

## Status

Alpha. The API is small but may shift before 1.0. Pin a SHA in production:

```toml
# pyproject.toml
[tool.poetry.dependencies]
stitchcall = { git = "https://github.com/QuinRiva/stitchcall.git", rev = "<sha>" }
```

## License

MIT. See [`LICENSE`](LICENSE).
