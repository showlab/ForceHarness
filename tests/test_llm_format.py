from forceharness.agent.llm import LLMClient, ToolCall


def msgs():
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 10
    return [
        {"role": "system", "text": "sys"},
        {"role": "user", "parts": [{"type": "text", "text": "hi"}, {"type": "image", "png": png, "label": "cam"}]},
        {"role": "assistant", "text": "ok", "tool_calls": [ToolCall("c1", "move_to", {"x": 0.1, "y": 0, "z": 0.9})]},
        {"role": "tool", "tool_call_id": "c1", "name": "move_to", "parts": [{"type": "text", "text": "result"}, {"type": "image", "png": png, "label": "plot"}]},
        {"role": "assistant", "text": "", "tool_calls": [ToolCall("c2", "done", {"success": True})]},
        {"role": "tool", "tool_call_id": "c2", "name": "done", "parts": [{"type": "text", "text": "episode ended"}]},
    ]


def test_openai_conversion_moves_tool_images_to_user_message():
    out = LLMClient._to_openai(msgs())
    roles = [m["role"] for m in out]
    assert roles[:3] == ["system", "user", "assistant"]
    i = roles.index("tool")
    assert out[i]["content"].startswith("result") and "attached below" in out[i]["content"]
    assert roles[i + 1] == "user" and out[i + 1]["content"][1]["type"] == "image_url"
    assert out[2]["tool_calls"][0]["function"]["name"] == "move_to"


def test_anthropic_conversion_groups_tool_results():
    system, out = LLMClient._to_anthropic(msgs())
    assert system == "sys"
    assert out[0]["role"] == "user" and out[0]["content"][1]["type"] == "image"
    assert out[1]["content"][1]["type"] == "tool_use"
    assert out[2]["content"][0]["type"] == "tool_result" and out[2]["content"][0]["content"][1]["type"] == "image"


def test_openai_conversion_empty_assistant_reply_has_string_content():
    # a model can reply with an empty message (no text, no tool call); OpenAI rejects content=None with HTTP 400
    out = LLMClient._to_openai([{"role": "system", "text": "s"}, {"role": "user", "parts": [{"type": "text", "text": "u"}]},
                                {"role": "assistant", "text": "", "tool_calls": []}])
    a = out[-1]
    assert a["role"] == "assistant" and isinstance(a["content"], str) and a["content"]
    assert "tool_calls" not in a
