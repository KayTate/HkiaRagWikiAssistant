"""Gradio frontend for the Hello Kitty Island Adventure RAG assistant.

Implements a single-page chat interface backed by the LangGraph agent.
The agent.graph import is deferred inside the respond function so this
module can be imported and tested independently.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so imports work when running
# this file directly (e.g. `python app/gradio_app.py`).
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from typing import Any

import gradio as gr


def _history_to_messages(history: list[Any]) -> list[Any]:
    """Convert Gradio chat history to the message list expected by AgentState.

    Gradio provides history as a list of [user_text, assistant_text] pairs
    (in Gradio 3/4) or as OpenAI-style message dicts (in Gradio 5+). We
    normalise to a flat list of dicts with 'role' and 'content' keys so
    AgentState.messages has a consistent shape regardless of Gradio version.

    Args:
        history: The chat history list supplied by gr.ChatInterface.

    Returns:
        List of message dicts with 'role' ('user' or 'assistant') and
        'content' keys.
    """
    messages: list[dict[str, str]] = []
    for turn in history:
        if isinstance(turn, dict):
            messages.append(
                {
                    "role": str(turn.get("role", "")),
                    "content": str(turn.get("content", "")),
                }
            )
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            user_text, assistant_text = turn
            if user_text:
                messages.append({"role": "user", "content": str(user_text)})
            if assistant_text:
                messages.append({"role": "assistant", "content": str(assistant_text)})
    return messages


def respond(message: str, history: list[Any]) -> str:
    """Handle a single chat turn by invoking the RAG agent graph.

    The agent.graph import is deferred here so this module is importable
    even when Engineer 2's agent graph has not been implemented yet.

    Args:
        message: The user's current message text.
        history: The prior chat turns as provided by gr.ChatInterface.

    Returns:
        The agent's response as a plain string.
    """
    try:
        from agent.graph import compile_graph
        from agent.state import AgentState
    except ImportError:
        return (
            "The agent graph is not available yet. "
            "Please ensure the agent module is installed and try again."
        )

    graph = compile_graph()
    state = AgentState(
        question=message,
        messages=_history_to_messages(history),
    )
    # LangGraph invoke() returns a dict when state is a dataclass.
    raw = graph.invoke(state)  # type: ignore[attr-defined]
    result = AgentState(**raw)
    return result.final_answer


# Gradio 6+ removed the 'theme' kwarg from ChatInterface; theming is applied
# at the Blocks level via gr.Blocks(theme=...) when launching from __main__.
demo = gr.ChatInterface(
    fn=respond,
    title="Hello Kitty Island Adventure Assistant",
    description=(
        "Ask anything about HKIA — quests, characters, crafting, locations, and more."
    ),
    examples=[
        "What quests do I need to complete to unlock Ice and Glow?",
        "What gifts does Keroppi like?",
        "How do I craft a Wooden Block?",
        "How do I get to Icy Peak?",
        "How does the daily gift limit work?",
    ],
)

if __name__ == "__main__":
    demo.launch()
