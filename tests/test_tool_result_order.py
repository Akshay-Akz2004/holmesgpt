"""Tool results must follow the order of the assistant's tool_calls.

Some chat templates (e.g. Qwen served through vLLM) render tool results without
their tool_call_id, so the model can only match a result to its call by
position. Every path that writes tool results into the history must keep them
in request order, including the paths that resume a paused conversation.
"""

import json
from unittest.mock import MagicMock

import pytest

import holmes.utils.approval_tokens as approval_tokens
from holmes.core.models import FrontendToolResult, ToolApprovalDecision
from holmes.core.tool_calling_llm import ToolCallingLLM, _order_tool_results_by_request


@pytest.fixture(autouse=True)
def stable_signing_key(monkeypatch):
    monkeypatch.setattr(approval_tokens, "SIGNING_KEY", b"\x42" * 32)


def _build_ai() -> ToolCallingLLM:
    ai = ToolCallingLLM(
        tool_executor=MagicMock(),
        max_steps=5,
        llm=MagicMock(),
        tool_results_dir=None,
    )
    ai._invoke_llm_tool_call = MagicMock(
        side_effect=AssertionError("denied tool calls must not run")
    )
    return ai


def _tool_call(tool_call_id: str, **flags) -> dict:
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": tool_call_id})},
        **flags,
    }


def _tool_result(tool_call_id: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "name": "bash", "content": "ok"}


def _tool_result_ids(messages: list) -> list:
    return [m["tool_call_id"] for m in messages if m.get("role") == "tool"]


def test_order_tool_results_by_request_sorts_each_assistant_turn():
    messages = [
        {"role": "user", "content": "check"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("a"), _tool_call("b")]},
        _tool_result("b"),
        _tool_result("unknown"),
        _tool_result("a"),
        {"role": "user", "content": "and now?"},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("c"), _tool_call("d")]},
        _tool_result("d"),
        _tool_result("c"),
    ]

    _order_tool_results_by_request(messages)

    assert _tool_result_ids(messages) == ["a", "b", "unknown", "c", "d"]
    assert messages[5] == {"role": "user", "content": "and now?"}


def test_approval_resume_keeps_request_order():
    # tc_b ran before the pause; tc_a and tc_c waited for approval.
    approval_calls = {
        tool_call_id: _tool_call(
            tool_call_id,
            pending_approval=True,
            approval_token=approval_tokens.mint_token(
                tool_call_id, "bash", json.dumps({"command": tool_call_id})
            ),
        )
        for tool_call_id in ("tc_a", "tc_c")
    }
    messages = [
        {"role": "user", "content": "clean up"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [approval_calls["tc_a"], _tool_call("tc_b"), approval_calls["tc_c"]],
        },
        _tool_result("tc_b"),
    ]
    decisions = [
        ToolApprovalDecision(tool_call_id="tc_a", approved=False),
        ToolApprovalDecision(tool_call_id="tc_c", approved=False),
    ]

    updated, _ = _build_ai()._execute_tool_decisions(messages, decisions)

    assert _tool_result_ids(updated) == ["tc_a", "tc_b", "tc_c"]


def test_orphaned_tool_calls_keep_request_order():
    messages = [
        {"role": "user", "content": "clean up"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("tc_a"), _tool_call("tc_b"), _tool_call("tc_c")],
        },
        _tool_result("tc_b"),
        {"role": "user", "content": "never mind"},
    ]

    updated, _ = _build_ai()._resolve_orphaned_tool_calls(messages)

    assert _tool_result_ids(updated) == ["tc_a", "tc_b", "tc_c"]
    assert updated[-1] == {"role": "user", "content": "never mind"}


def test_frontend_tool_results_keep_request_order():
    messages = [
        {"role": "user", "content": "show me"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                _tool_call("tc_a", pending_frontend=True),
                _tool_call("tc_b"),
                _tool_call("tc_c", pending_frontend=True),
            ],
        },
        _tool_result("tc_b"),
    ]
    results = [
        FrontendToolResult(tool_call_id=tool_call_id, tool_name="bash", result="ok")
        for tool_call_id in ("tc_a", "tc_c")
    ]

    updated, _ = ToolCallingLLM._process_frontend_tool_results(messages, results)

    assert _tool_result_ids(updated) == ["tc_a", "tc_b", "tc_c"]
