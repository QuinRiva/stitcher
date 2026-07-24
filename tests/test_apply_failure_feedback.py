"""Behaviour 1 — shape-informed JSON-Patch apply-failure feedback.

On a ``jsonpatch`` apply failure the repair loop feeds the model the current
value at the deepest EXISTING ancestor of the failing pointer, so the retry is
informed (it can see that a scope field is a sentinel string with nothing to
index, or how long a list actually is) rather than re-issuing the same doomed
op. These unit-test the helper suite that builds that feedback.
"""
from __future__ import annotations

import jsonpatch

from stitcher.extractor import (
    _apply_failure_feedback,
    _describe_shape,
    _find_failing_op,
    _nearest_existing,
    _pointer_parts,
    _render_pointer,
)


def _feedback(doc, ops) -> str:
    try:
        jsonpatch.JsonPatch(ops).apply(doc)
        raise AssertionError("expected apply failure")
    except (jsonpatch.JsonPatchException, jsonpatch.JsonPointerException) as exc:
        return str(_apply_failure_feedback(doc, ops, exc))


def test_find_failing_op_pinpoints_culprit_and_state():
    doc = {"scope": "NONE_IN_SCOPE"}
    ops = [
        {"op": "replace", "path": "/scope", "value": []},
        {"op": "add", "path": "/other/-", "value": "x"},
    ]
    state_before, failing_op, applied_before = _find_failing_op(doc, ops)
    assert failing_op["path"] == "/other/-"
    # The state the failing op actually saw reflects the earlier replace.
    assert state_before == {"scope": []}
    assert applied_before is True


def test_feedback_uses_state_after_earlier_ops():
    # The shape must be the state immediately before the failing op (earlier
    # ops already applied), not the original document.
    doc = {"items": ["a", "b", "c"]}
    ops = [
        {"op": "remove", "path": "/items/0"},
        {"op": "replace", "path": "/items/2", "value": "z"},
    ]
    fb = _feedback(doc, ops)
    # After the removal the list is ['b','c'] (indices 0..1); index 2 is gone.
    assert '["b", "c"]' in fb
    assert "valid indices 0..1" in fb
    assert "AFTER your earlier operation" in fb  # provenance note


def test_feedback_move_names_failing_destination():
    # A move whose source resolves but destination fails must surface the
    # destination shape, not the healthy source.
    doc = {"src": [1, 2], "dst": "NONE_IN_SCOPE"}
    ops = [{"op": "move", "from": "/src/0", "path": "/dst/-"}]
    fb = _feedback(doc, ops)
    assert "the destination path" in fb
    assert "NONE_IN_SCOPE" in fb


def test_feedback_preserves_rfc6901_escaping():
    # A key containing '/' must be re-escaped in the surfaced pointer.
    doc = {"a/b": {"x": 1}}
    ops = [{"op": "replace", "path": "/a~1b/y", "value": 2}]
    fb = _feedback(doc, ops)
    assert "/a~1b" in fb


def test_feedback_caps_large_list_shape():
    # The char budget must cover list containers too.
    doc = {"items": [{"k": "x" * 10000} for _ in range(3)]}
    ops = [{"op": "add", "path": "/items/5/x", "value": 1}]
    fb = _feedback(doc, ops)
    assert len(fb) < 3000  # was ~30k before the cap covered lists
    # Truncated content is not mislabelled as a json fence.
    assert '```json\n[{"k": "xxxx' not in fb


def test_feedback_surfaces_sentinel_collapse():
    doc = {"scope": "NONE_IN_SCOPE"}
    ops = [{"op": "add", "path": "/scope/6/roles/-", "value": {}}]
    fb = _feedback(doc, ops)
    assert "could not be applied" in fb
    assert "`/scope`" in fb
    assert "NONE_IN_SCOPE" in fb  # current shape shown so the retry is informed
    # Descent stopped at a scalar → the scalar guidance fires (generically —
    # no schema-specific sentinel vocabulary in a library error message).
    assert "scalar, not a container" in fb


def test_guidance_is_conditional():
    # An out-of-range index on a healthy list: neither the scalar nor the
    # multi-remove advice applies, so neither may appear — irrelevant advice
    # dilutes the actionable signal.
    doc = {"items": [1, 2]}
    ops = [{"op": "replace", "path": "/items/9", "value": 3}]
    fb = _feedback(doc, ops)
    assert "scalar, not a container" not in fb
    assert "highest-index-first" not in fb


def test_multi_remove_guidance_fires_for_same_list():
    doc = {"items": [1, 2, 3]}
    ops = [
        {"op": "remove", "path": "/items/0"},
        {"op": "remove", "path": "/items/5"},
    ]
    fb = _feedback(doc, ops)
    assert "highest-index-first" in fb


def test_empty_list_guidance_fires():
    doc = {"xs": []}
    ops = [{"op": "replace", "path": "/xs/0", "value": 1}]
    fb = _feedback(doc, ops)
    assert "An empty list `[]` has no indices" in fb


def test_describe_shape_truncation_uses_plain_fence():
    big = _describe_shape({"k": "x" * 5000})
    assert big.startswith("```\n") and "```json" not in big
    small = _describe_shape({"k": 1})
    assert small.startswith("```json")


def test_describe_shape_summarises_long_lists():
    assert "5 element(s)" in _describe_shape([1, 2, 3, 4, 5])
    assert "0..4" in _describe_shape([1, 2, 3, 4, 5])
    assert _describe_shape([]) == "an empty list `[]`"


def test_nearest_existing_stops_at_sentinel_string():
    # The key insight: descent must NOT index into the sentinel string; the
    # useful signal is that the field is a scalar, not a list.
    doc = {"scope": "NONE_IN_SCOPE"}
    pointer, value = _nearest_existing(doc, _pointer_parts("/scope/6/roles/-"))
    assert pointer == "/scope"
    assert value == "NONE_IN_SCOPE"


def test_nearest_existing_stops_at_list_bounds():
    doc = {"docs": [{"a": 1}]}
    pointer, value = _nearest_existing(doc, _pointer_parts("/docs/2"))
    assert pointer == "/docs"
    assert value == [{"a": 1}]


def test_nearest_existing_stops_at_append_token():
    doc = {"docs": [1, 2]}
    pointer, value = _nearest_existing(doc, _pointer_parts("/docs/-"))
    assert pointer == "/docs"
    assert value == [1, 2]


def test_pointer_parts_round_trips_escaping():
    # RFC 6901: ~1 -> '/', ~0 -> '~', and back.
    parts = _pointer_parts("/a~1b/c~0d")
    assert parts == ["a/b", "c~d"]
    assert _render_pointer(parts) == "/a~1b/c~0d"
    assert _pointer_parts("") == []
    assert _render_pointer([]) == ""
