"""Tests for app/gradio_app.py — acceptance criterion: module is importable
and the demo object is a gr.ChatInterface instance."""

import builtins
import importlib
from collections.abc import Mapping, Sequence
from typing import Any

import gradio as gr
import pytest


def test_gradio_app_is_importable() -> None:
    """app.gradio_app can be imported without raising any exceptions."""
    import app.gradio_app  # noqa: F401 — import is the test


def test_demo_is_chat_interface_instance() -> None:
    """The demo object is a gr.ChatInterface, confirming the app is constructed."""
    from app.gradio_app import demo

    assert isinstance(demo, gr.ChatInterface)


def test_respond_returns_fallback_when_agent_graph_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """respond returns a user-friendly string when agent.graph cannot be imported."""
    real_import = builtins.__import__

    def _mock_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:  # Any is required — __import__ return type is not narrowable
        if name == "agent.graph":
            raise ImportError("agent.graph not available")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _mock_import)

    # Re-import respond after patching so the lazy import fires fresh.
    module = importlib.import_module("app.gradio_app")
    respond = module.respond

    result = respond("What quests unlock Ice and Glow?", [])
    assert isinstance(result, str)
    assert len(result) > 0


def test_history_to_messages_handles_dict_format() -> None:
    """_history_to_messages normalises dict-format history from Gradio 5+."""
    from app.gradio_app import _history_to_messages

    history: list[Any] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    result = _history_to_messages(history)
    assert result == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]


def test_history_to_messages_handles_list_pair_format() -> None:
    """_history_to_messages normalises list-pair history from Gradio 3/4."""
    from app.gradio_app import _history_to_messages

    history: list[Any] = [["user message", "assistant reply"]]
    result = _history_to_messages(history)
    assert result == [
        {"role": "user", "content": "user message"},
        {"role": "assistant", "content": "assistant reply"},
    ]
