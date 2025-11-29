"""
utils.py

Small utility module providing a simple file-backed memory abstraction for two memory "forms":
 - "list" : used to store lists like possible diagnoses, short items, etc.
 - "chat" : used to store chat flow as a flat list of alternating user/bot messages:
            [user_msg1, bot_msg1, user_msg2, bot_msg2, ...]

Primary function exported to use:
    change_memory(form: str, type: str, content: Optional[list[str]] = None)

Behavior:
 - type == "append" : content must be list[str]; extends existing memory by those items.
 - type == "update" : content must be list[str] or None; replaces entire memory with content.
                       If content is None or empty list -> treated as reset (clears memory).
 - type == "fetch"  : content ignored; returns the list[str] representing current memory.

Notes:
 - Very small/simple file-based approach (JSON files under ./data). No DB or concurrency.
 - Single-user local use is assumed.

---

Professional usage pattern for Groq LLM inference (endpoint-safe).

This module exposes a single helper function, `llm_invoke()`,
which wraps the Groq client call into a reusable API.

Features supported:
  • Custom system prompt
  • Custom user prompt
  • Model selection (default Llama 3.3 70B)
  • Temperature, max token settings
  • Optional structured outputs via Groq's specification format
    Docs: https://console.groq.com/docs/structured-outputs
  • Model catalog reference:
    https://console.groq.com/docs/models

This function always performs a non-streaming call (stream=False) and
returns the final completion text (or structured JSON as text).
"""

import json
import os
from groq import Groq
from typing import List, Optional, Union, Dict, Any

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

_LIST_FILE = os.path.join(DATA_DIR, "memory_list.json")
_CHAT_FILE = os.path.join(DATA_DIR, "memory_chat.json")


# --- Internal helpers ------------------------------------------------------
def _get_filepath(form: str) -> str:
    if form == "list":
        return _LIST_FILE
    elif form == "chat":
        return _CHAT_FILE
    else:
        raise ValueError("form must be 'list' or 'chat'.")


def _read_memory(form: str) -> List[str]:
    path = _get_filepath(form)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
            if not isinstance(data, list):
                return []
            return [str(x) for x in data]
        except json.JSONDecodeError:
            return []


def _write_memory(form: str, items: List[str]) -> None:
    path = _get_filepath(form)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2, ensure_ascii=False)


# --- Public API ------------------------------------------------------------
def change_memory(form: str, type: str, content: Optional[List[str]] = None) -> Union[List[str], bool]:
    """
    Manage the file-backed memory.

    Args:
        form: "list" or "chat"
        type: "append", "update", or "fetch"
        content: list[str] when required by type ("append" or "update")

    Returns:
        - for "fetch": returns List[str] with current memory
        - for "append" or "update": returns True on success (or raises on error)

    Notes:
        - For "update": if content is None OR content == [], that is treated as a reset (clears memory).
        - For "chat": we expect the stored list to be alternating user/bot messages.
    """
    form = str(form)
    type = str(type).lower()

    if form not in ("list", "chat"):
        raise ValueError("Invalid form. Must be 'list' or 'chat'.")

    if type not in ("append", "update", "fetch"):
        raise ValueError("Invalid type. Must be 'append', 'update', or 'fetch'.")

    if type == "fetch":
        return _read_memory(form)

    # For append/update, ensure content is a list of strings (or None for update->reset)
    if type == "append":
        if not isinstance(content, list):
            raise ValueError("For append, content must be a list of strings.")
        content = [str(x) for x in content]
        existing = _read_memory(form)
        new = existing + content
        _write_memory(form, new)
        return True

    if type == "update":
        # treat None or [] as reset/clear
        if content is None:
            content_items = []
        else:
            if not isinstance(content, list):
                raise ValueError("For update, content must be a list of strings or None (to reset).")
            content_items = [str(x) for x in content]
        _write_memory(form, content_items)
        return True


# Convenience helpers (optional)
def fetch_memory(form: str) -> List[str]:
    return change_memory(form=form, type="fetch")


def append_memory(form: str, items: List[str]) -> bool:
    return change_memory(form=form, type="append", content=items)


def update_memory(form: str, items: Optional[List[str]]) -> bool:
    return change_memory(form=form, type="update", content=items)

# --- Public API for LLM invocation ----------------------------------------------
def invoke_llm(
    system_prompt: str = "",
    user_prompt: str = "",
    model_id: str = "llama-3.3-70b-versatile",
    temperature: float = 0.5,
    max_completion_tokens: int = 1024,
    top_p: float = 1.0,
    structured_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Execute a Groq LLM completion request with optional structured output.
    This function is endpoint-safe: it always uses a non-streaming request.

    Args:
        system_prompt (str): Content for system role.
        user_prompt   (str): Content for user role.
        model_id      (str): Groq model to use. Default: Llama 3.3 70B.
        temperature (float): Sampling temperature.
        max_completion_tokens (int): Maximum tokens to generate.
        top_p       (float): Nucleus sampling cutoff.
        structured_schema (dict | None): JSON schema for structured outputs.
                                          If provided, the model will return
                                          validated JSON according to schema.

    Returns:
        str: The generated model output. If structured output is enabled,
             the returned string will be valid JSON matching the schema.
    """

    if user_prompt.strip() == "":
        raise ValueError("user_prompt must be a non-empty string.")

    client = Groq()

    request_payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "model": model_id,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "top_p": top_p,
        # Always non-streaming for endpoint usage:
        "stream": False,
    }

    # Add structured output instruction if schema provided.
    # See: https://console.groq.com/docs/structured-outputs
    if structured_schema is not None:
        request_payload["response_format"] = {
            "type": "json_schema",
            "json_schema": structured_schema,
        }

    # Perform the request (non-streaming)
    completion = client.chat.completions.create(**request_payload)

    # Return the final message content
    return completion.choices[0].message.content
