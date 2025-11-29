"""
graph_func.py

-------------------------------------------------------------------
AVAILABLE FUNCTIONS (from utils.py):

process_memory(form: str, type: str, content: Optional[list[str]] = None)
      form options:
          "chat" : alternating user / bot messages
                    [user1, bot1, user2, bot2, ...]
          "list" : simple list of strings

      type options:
          "append" : content must be list[str]; extends memory
          "update" : content list[str] or None; replaces whole memory
                      None or [] resets that memory
          "fetch"  : returns current memory list[str]
      content: either list[str] (for append/update) or None (for reset in update type)

Note:
  This project uses ONLY examine_query() as the backend entry point.
  Your logic will be placed inside examine_query() in the marked region.


invoke_llm(
      system_prompt: str,
      user_prompt: str,
      model_id: str = "llama-3.3-70b-versatile",
      temperature: float = 0.5,
      max_completion_tokens: int = 1024,
      top_p: float = 1.0,
      structured_schema: Optional[dict] = None
)

Description:
      A general-purpose wrapper for Groq LLM calls (non-streaming, endpoint-safe).
      Returns the final generated text from the model.

Arguments:
      system_prompt:
          Text that defines the assistant's behavior throughout the conversation.

      user_prompt:
          The user's input or question.

      model_id:
          Any Groq-supported model string.
          Default: "llama-3.3-70b-versatile"
          Full list: https://console.groq.com/docs/models

      temperature:
          Controls randomness. Lower → more deterministic.

      max_completion_tokens:
          Maximum number of tokens the model can generate.

      top_p:
          Nucleus sampling cutoff. 1.0 includes the full distribution.

      structured_schema:
          Optional JSON schema enabling structured outputs.
          If provided, the model returns JSON matching this schema.
          Docs: https://console.groq.com/docs/structured-outputs

Behavior:
      • Always uses stream=False to ensure the endpoint returns a clean final output.
      • Returns completion.choices[0].message.content as str.
      • If structured_schema is supplied, content will be JSON formatted text.

Usage:
      reply = invoke_llm(
          system_prompt="You are a helpful assistant.",
          user_prompt="Explain transformers in simple terms."
      )

      print(reply)

Notes:
      This helper is safe for direct use inside FastAPI endpoints.
      For structured responses, supply `structured_schema={...}`.
      The returned string can be parsed into JSON when schema-based output is used.
-------------------------------------------------------------------
"""

from typing import Tuple
from groq import Groq
from utils import *


def examine_query(query: str, first_query: bool = True) -> Tuple[str, bool]:
    """
    Core placeholder. Replace the inside section with real logic.

    Args:
        query (str): user's raw input text
        first_query (bool): whether this is the first message of a new thread

    Returns:
        (response_str, continue_flag)
    """

    # Basic validation
    if not isinstance(query, str):
        raise ValueError("query must be a string.")
    if not isinstance(first_query, bool):
        raise ValueError("first_query must be a boolean.")
    
    # Initialize reply and flag
    reply_text = ""
    continue_flag = True

    # -------------------------------------------------------
    # WRITE LOGIC HERE:
    #   Use utils.fetch_memory / utils.append_memory / utils.update_memory
    #   to implement the complete query-processing behavior.
    #
    #   Example flow:
    #       flow = utils.fetch_memory("chat")
    #       ... analyze query ...
    #       reply_text = "<your generated output>"
    #       utils.append_memory("chat", [reply_text])
    #       return reply_text, True
    #
    #   This is only an example; the actual logic will differ.
    # -------------------------------------------------------
    #
    # system_prompt = "You are a concise assistant."
    # user_prompt = "Give me a one-sentence explanation of entropy."
    #
    # EXAMPLE LLM CALL
    #
    # reply_text = invoke_llm(
    #     system_prompt=system_prompt,
    #     user_prompt=user_prompt,
    #     model_id="llama-3.3-70b-versatile",   # optional, default is already this
    #     temperature=0.3,
    #     max_completion_tokens=200,
    #     top_p=1.0
    # )

    return reply_text, continue_flag
