import re
from typing import List, Optional


def strip_thinking_tags(text: str) -> str:
    """
    Strip <think>...</think> tags from text for DeepSeek-R1/Qwen thinking mode.
    This ensures the preference model evaluates only the user-visible response.

    Args:
        text (str): Text potentially containing thinking tags.

    Returns:
        str: Text with thinking content removed.
    """
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return stripped.strip()


def strip_scratchpad(text: str) -> str:
    """
    Strip <scratchpad>...</scratchpad> tags from text.
    This ensures the preference model evaluates only the user-visible response.

    Args:
        text (str): Text potentially containing scratchpad tags.

    Returns:
        str: Text with scratchpad content removed.
    """
    stripped = re.sub(r"<scratchpad>.*?</scratchpad>", "", text, flags=re.DOTALL)
    return stripped.strip()


def strip_hidden_reasoning(text: str) -> str:
    """
    Strip all hidden reasoning tags from text (both thinking and scratchpad).
    This ensures the preference model evaluates only the user-visible response.

    Args:
        text (str): Text potentially containing hidden reasoning tags.

    Returns:
        str: Text with all hidden reasoning content removed.
    """
    # Strip thinking tags
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip scratchpad tags
    stripped = re.sub(r"<scratchpad>.*?</scratchpad>", "", stripped, flags=re.DOTALL)
    return stripped.strip()


class AssessorModel:
    def __init__(
        self,
        system_prompt: str,
        prompt_suffix: str,
        allow_to_see_tool_calls: bool,
        perspective: Optional[str] = None,
        valid_tokens: Optional[tuple] = None,
        **kwargs,
    ):
        """
        Initialize the AssessorModel.

        Args:
            system_prompt (str): The system prompt to be used.
            prompt_suffix (str): The suffix to be added to the prompt.
            allow_to_see_tool_calls (bool): Whether to include tool calls in the conversation history.
            perspective (Optional[str]): The perspective to use for feedback (Third_Person or First_Person).
            valid_tokens (Optional[tuple]): A tuple of valid tokens.
            **kwargs: Additional keyword arguments.
        """
        self.system_prompt = system_prompt
        self.prompt_suffix = prompt_suffix
        self.allow_to_see_tool_calls = allow_to_see_tool_calls
        self.valid_tokens = valid_tokens if valid_tokens is not None else ()
        self.perspective = perspective

        # Just so we dont have a million print statements
        kwargs_to_ignore = ["aggregation", "categorical"]
        for kwarg in kwargs_to_ignore:
            if kwarg in kwargs:
                del kwargs[kwarg]

    def prepare_messages(self, state) -> List[dict]:
        """
        Prepare messages for the assessor model based on the conversation history.

        Args:
            state: The current state containing conversation history and format variables.

        Returns:
            List[dict]: A list of prepared messages for the assessor model.

        Raises:
            AssertionError: If the conversation history does not meet certain criteria.
            NotImplementedError: If the First_Person perspective is used.
            ValueError: If an invalid perspective is provided.
        """
        assert any(message["role"] == "environment" for message in state.history), "No user message in history"
        assert any(message["role"] == "agent" for message in state.history), "No agent message in history"
        assert state.history[-1]["role"] in ["agent", "environment_system"], "Last message should be from agent or sys"

        # We dont want to give feedback on the latest system messages. System messages can only happen after an agent message so this is safe.
        conversation_history = (
            state.history[:-1] if state.history[-1]["role"] == "environment_system" else state.history
        )
        if self.allow_to_see_tool_calls:
            filtered_conversation_history = conversation_history
        else:
            filtered_conversation_history = [
                message
                for message in conversation_history
                if message["role"] not in ["tool_call", "tool_response", "environment_system"]
            ]
        if self.perspective == "Third_Person" or self.perspective is None:
            # If feedback is given from the third person perspective, we give the conversation history in one message and ask it to rate it
            task_prompt = self.system_prompt.format_map(state.format_vars).strip()

            context_prompt = "The conversation history is:\n\n"
            role_to_tag = {
                "agent": state.format_vars["agent_name"],
                "environment": "user",
                "environment_system": "system",
                "tool_call": "tool_call",
                "tool_response": "tool_response",
            }
            for message in filtered_conversation_history:
                content = message["content"]
                # Strip hidden reasoning (thinking tags AND scratchpad) from agent messages
                # so preference model only sees user-visible response
                if message["role"] == "agent":
                    content = strip_hidden_reasoning(content)
                context_prompt += wrap_between_tags(content, role_to_tag[message["role"]])
                context_prompt += "\n\n"
            context_prompt += self.prompt_suffix

            messages_out = [
                {"role": "system", "content": task_prompt},
                {"role": "user", "content": context_prompt},
            ]
        elif self.perspective == "First_Person":
            raise NotImplementedError("First person perspective not implemented")
        else:
            raise ValueError("Invalid perspective")

        return messages_out


def wrap_between_tags(text, tag_name):
    """
    Wrap the given text between XML-style tags.

    Args:
        text (str): The text to be wrapped.
        tag_name (str): The name of the tag to use.

    Returns:
        str: The text wrapped between the specified tags.
    """
    return f"<{tag_name}>: {text}</{tag_name}>"
