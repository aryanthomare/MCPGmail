from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


@dataclass
class DebugLogger:
    enabled: bool
    log_file: Path | None

    def log(self, event: str, payload: object) -> None:
        if not self.enabled or self.log_file is None:
            return

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "event": event,
            "payload": payload,
        }
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str))
            handle.write("\n")


def build_llm(
    provider: str,
    model: str,
    base_url: str,
    temperature: float,
    claude_model: str = "claude-opus-4-5",
    claude_api_key: str | None = None,
) -> object:
    """Return a LangChain chat model for the requested provider."""
    if provider == "claude":
        return ChatAnthropic(model=claude_model, api_key=claude_api_key, temperature=temperature)
    if provider == "ollama":
        return ChatOllama(model=model, base_url=base_url, temperature=temperature)
    raise ValueError(f"Unknown provider: {provider!r}. Choose 'ollama' or 'claude'.")


def parse_args() -> argparse.Namespace:
    # Keep the CLI focused on one of two execution modes: direct chat or MCP-backed chat.
    parser = argparse.ArgumentParser(
        description="Call a local Ollama model or the Claude API via LangChain and optionally run Gmail MCP tools."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompt",
        help="User prompt to send to the model directly (no MCP tools).",
    )
    input_group.add_argument(
        "--mcp-request",
        help="Request text sent to a LangGraph ReAct agent backed by Gmail MCP tools.",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="Optional system instruction (only used with --prompt).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", os.getenv("OLLAMA_OPENAI_MODEL", "gemma4")),
        help="Model name served by Ollama.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_BASE_URL", os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434")),
        help="Ollama base URL.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-tools",
        type=int,
        default=800,
        help="Maximum tokens in the response.",
    )
    parser.add_argument(
        "--tool-ranking-prompt-file",
        default=os.getenv("TOOL_RANKING_PROMPT_FILE", "tool_ranking_prompt.md"),
        help="Markdown file containing ranking instructions. Supports {{request}} and {{tool_catalog_json}} placeholders.",
    )
    parser.add_argument(
        "--max-agent-loops",
        type=int,
        default=int(os.getenv("MAX_AGENT_LOOPS", "3")),
        help="Maximum number of outer agent loops; each loop runs the agent then asks an LLM completion checker if the task is done.",
    )
    parser.add_argument(
        "--show-llm-io",
        action="store_true",
        help="Print intermediate agent messages for debugging.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log prompts, responses, and section boundaries to a JSONL file.",
    )
    parser.add_argument(
        "--debug-log-file",
        default=os.getenv("MCP_DEBUG_LOG_FILE", "mcp_pipeline_debug.log"),
        help="Path to the debug log file (used with --debug).",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "claude"],
        default=os.getenv("LLM_PROVIDER", "ollama"),
        help="LLM backend to use: 'ollama' (default, local) or 'claude' (Anthropic API).",
    )
    parser.add_argument(
        "--claude-model",
        default=os.getenv("CLAUDE_MODEL", "claude-opus-4-5"),
        help="Anthropic model name (used with --provider claude).",
    )
    parser.add_argument(
        "--claude-api-key",
        default=os.getenv("ANTHROPIC_API_KEY"),
        help="Anthropic API key (used with --provider claude). Defaults to ANTHROPIC_API_KEY env var.",
    )
    return parser.parse_args()




def find_oauth_keys_file() -> Path | None:
    # Support both a local project file and the standard ~/.gmail-mcp location.
    candidates = [
        Path.cwd() / "gcp-oauth.keys.json",
        Path.home() / ".gmail-mcp" / "gcp-oauth.keys.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_credentials_file() -> Path | None:
    # Credentials are stored alongside the OAuth keys or in the user's Gmail MCP folder.
    candidates = [
        Path.cwd() / "credentials.json",
        Path.home() / ".gmail-mcp" / "credentials.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def server_working_directory() -> Path:
    # Use a stable temp folder so the MCP server has a writable working directory.
    temp_dir = Path(os.getenv("TEMP", str(Path.home()))) / "gmail-mcp-client"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def get_tool_schema(tool: object) -> dict[str, object]:
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return {}
    if hasattr(args_schema, "model_json_schema"):
        return args_schema.model_json_schema()  # type: ignore[return-value]
    if hasattr(args_schema, "schema"):
        return args_schema.schema()  # type: ignore[return-value]
    return {}


def extract_json_object(text: str) -> dict[str, object] | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                payload = try_parse_json_object(candidate)
                if payload is not None:
                    return payload

                repaired = repair_missing_closers(candidate)
                if repaired is not None:
                    payload = try_parse_json_object(repaired)
                    if payload is not None:
                        return payload

                return None

    if depth > 0:
        candidate = text[start:]
        repaired = repair_missing_closers(candidate)
        if repaired is not None:
            payload = try_parse_json_object(repaired)
            if payload is not None:
                return payload

    return None


def try_parse_json_object(text: str) -> dict[str, object] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        return payload
    return None


def repair_missing_closers(text: str) -> str | None:
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape = False

    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == '{':
            open_braces += 1
        elif char == '}':
            open_braces -= 1
        elif char == '[':
            open_brackets += 1
        elif char == ']':
            open_brackets -= 1

    if open_braces < 0 or open_brackets < 0:
        return None

    if not open_braces and not open_brackets:
        return None

    if open_brackets > 0 and text.endswith("}"):
        return f"{text[:-1]}{']' * open_brackets}}}"

    closing = "]" * open_brackets + "}" * open_braces
    return f"{text}{closing}"


def build_step_prompt(request_text: str, tool_prompt: str, tool_outputs: str | None = None) -> str:
    if tool_outputs:
        tool_outputs = truncate_for_prompt(tool_outputs)

    sections = [
        "User request:",
        request_text,
        "",
        "Tool catalog prompt:",
        tool_prompt,
    ]
    catalog_json = json.dumps(catalog, indent=2)

    prompt = template.replace("{{request}}", request).replace("{{tool_catalog_json}}", catalog_json)
    if "{{request}}" not in template:
        prompt = f"{prompt}\n\nUser request:\n{request}"
    if "{{tool_catalog_json}}" not in template:
        prompt = f"{prompt}\n\nTool catalog:\n{catalog_json}"

    return "\n\n".join(
        [
            prompt,
            "Return JSON only. No markdown.",
            'Use this exact shape: {"rankedTools":[{"name":"tool_name","reason":"why"}]}',
            "Only include tool names from the provided catalog.",
        ]
    )


def rank_tools_with_prompt(
    tools: list[object],
    request: str,
    model: str,
    base_url: str,
    prompt_file: Path,
    debug_logger: DebugLogger | None = None,
    provider: str = "ollama",
    claude_model: str = "claude-opus-4-5",
    claude_api_key: str | None = None,
) -> list[object] | None:
    if not prompt_file.exists():
        return None

    template = prompt_file.read_text(encoding="utf-8")
    prompt = build_ranking_prompt(template, request, tools)
    llm = build_llm(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=0.0,
        claude_model=claude_model,
        claude_api_key=claude_api_key,
    ).with_structured_output(
        TOOL_RANKING_SCHEMA,
        method="json_schema",
    )
    tool_by_name = {
        getattr(tool, "name", "") or "": tool
        for tool in tools
        if getattr(tool, "name", "")
    }

    if debug_logger:
        debug_logger.log("tool_ranking_prompt_file", str(prompt_file))
        debug_logger.log("tool_ranking_prompt", prompt)

    try:
        response = llm.invoke(
            [
                SystemMessage(content="You are a strict tool-ranking assistant."),
                HumanMessage(content=prompt),
            ]
        )
    except Exception as error:  # noqa: BLE001
        if debug_logger:
            debug_logger.log("tool_ranking_error", str(error))
        return None

    if not isinstance(response, dict):
        return None

    if debug_logger:
        debug_logger.log("tool_ranking_response", response)

def truncate_for_prompt(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text

    head = text[: max_chars // 2]
    tail = text[-(max_chars // 2) :]
    return f"{head}\n\n[...truncated for planning context...]\n\n{tail}"


def find_oauth_keys_file() -> Path | None:
    candidates = [
        Path.cwd() / "gcp-oauth.keys.json",
        Path.home() / ".gmail-mcp" / "gcp-oauth.keys.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

    ranked_tools: list[object] = []
    seen_names: set[str] = set()
    for entry in ranked_entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            continue
        tool = tool_by_name.get(name)
        if tool is None:
            continue
        seen_names.add(name)
        ranked_tools.append(tool)

    return ranked_tools or None


def select_top_tools(
    all_tools: list[object],
    ranked_tools: list[object],
    request_text: str,
    top_tools: int,
) -> list[object]:
    if top_tools <= 0:
        return list(all_tools)

    tool_by_name = {
        getattr(tool, "name", "") or "": tool
        for tool in all_tools
        if getattr(tool, "name", "")
    }
    request_lower = request_text.lower()
    wants_read_email = (
        any(word in request_lower for word in {"read", "view", "summar", "show", "list"})
        and any(word in request_lower for word in {"email", "mail", "inbox"})
    )

    prioritized: list[object] = []
    if wants_read_email:
        for required_name in ["search_emails", "read_email"]:
            tool = tool_by_name.get(required_name)
            if tool is not None:
                prioritized.append(tool)

    for tool in ranked_tools:
        name = getattr(tool, "name", "") or ""
        if not name:
            continue
        if any((getattr(t, "name", "") or "") == name for t in prioritized):
            continue
        prioritized.append(tool)

    return prioritized[:top_tools]


def message_content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    try:
        return json.dumps(content, default=str)
    except TypeError:
        return str(content)


def format_messages_for_completion_check(messages: list[object], max_messages: int = 12) -> str:
    lines: list[str] = []
    for message in messages[-max_messages:]:
        message_type = str(getattr(message, "type", "unknown"))
        content = message_content_to_text(getattr(message, "content", ""))
        lines.append(f"[{message_type}] {content}")
    return "\n".join(lines)


def tool_result_to_text(result: object) -> str:
    content = getattr(result, "content", None)
    if not content:
        return str(result)

    parts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
        elif isinstance(item, dict) and "text" in item:
            parts.append(str(item["text"]))
        else:
            parts.append(str(item))

    return "\n".join(parts)


def tool_result_to_payload(result: object) -> object:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    return str(result)


def clean_email_body(text: str) -> str:
    # Decode entities and remove links first.
    normalized = html.unescape(text)
    normalized = re.sub(r"https?://\S+|www\.\S+", " ", normalized)

    # Drop full script/style blocks, comments, and any HTML tags.
    normalized = re.sub(r"(?is)<(script|style)\b.*?>.*?</\1>", " ", normalized)
    normalized = re.sub(r"(?is)<!--.*?-->", " ", normalized)
    normalized = re.sub(r"(?is)<[^>]+>", " ", normalized)

    # Remove common template artifacts that may remain after tag stripping.
    normalized = re.sub(r"(?i)\b(?:if|endif|mso|gte|acrite-mso-css)\b", " ", normalized)
    normalized = re.sub(
        r"(?i)\b(?:class|id|href|src|style|align|cellpadding|cellspacing|border|width|height|data-[a-z0-9_-]+)\s*=?\s*(?:\"[^\"]*\"|'[^']*')",
        " ",
        normalized,
    )
    normalized = re.sub(
        r"(?i)\b/?(?:div|span|table|tbody|thead|tr|td|th|html|body|head|meta|link|script|style)\b",
        " ",
        normalized,
    )

    # Keep letters, numbers, whitespace, and a small safe punctuation set.
    no_special = re.sub(r"[^A-Za-z0-9\s.,:;!?@\-_'\"()\[\]/]", " ", normalized)
    # Normalize whitespace for easier downstream summarization.
    compact = re.sub(r"\s+", " ", no_special).strip()
    return compact


def build_planning_system_prompt() -> str:
    return (
        "You are a strict JSON planner for Gmail MCP tools. Return valid JSON only with this shape: "
        '{"chosenTools":[{"name":"tool_name","reason":"why","arguments":{}}]}. '
        "Choose at most ONE immediately executable tool per response. "
        "Never include a tool whose arguments depend on a result that has not been produced yet. "
        "If the next needed action depends on a search result ID, choose search_emails first. "
        "If no tool is needed, return an empty chosenTools array. "
        "If you already have enough email data to write a useful summary, return an empty chosenTools array even if more emails may exist. "
        "For summary requests, stop planning once you have enough representative emails to describe the inbox clearly. "
        "Prefer stopping early over collecting extra emails. "
        "Do not request another round just to gather more detail unless the current data is clearly insufficient. "
        "Do not invent IDs. Do not ask clarifying questions."
    )


def build_final_answer_prompt(user_request: str, tool_outputs: str) -> str:
    return (
        "User request:\n"
        f"{user_request}\n\n"
        "Executed tool outputs:\n"
        f"{tool_outputs}\n\n"
        "Write a concise final answer for the user using the tool outputs above. "
        "Keep it under 250 words. Use at most 5 bullet points. "
        "Start with a one-sentence overall summary, then list the most important emails. "
        "Do not repeat the full email bodies or add filler text."
    )

    try:
        response = checker.invoke(
            [
                SystemMessage(content="You are a strict completion checker. Output valid JSON only."),
                HumanMessage(content=prompt),
            ]
        )
    except Exception as error:  # noqa: BLE001
        if debug_logger:
            debug_logger.log("completion_check_error", str(error))
        return False, f"Completion check failed: {error}"

    if not isinstance(response, dict):
        return False, "Completion checker returned an unexpected response type."

def extract_message_ids(text: str) -> list[str]:
    return re.findall(r"\bID:\s*([A-Za-z0-9_-]+)", text)


def summarize_tool_output_for_planning(name: str, cleaned_text: str, raw_text: str) -> str:
    if name == "search_emails":
        message_ids = extract_message_ids(raw_text)
        if not message_ids:
            return "Search completed. No message IDs found in result."

        preview_ids = ", ".join(message_ids[:10])
        suffix = "" if len(message_ids) <= 10 else f" (+{len(message_ids) - 10} more)"
        return (
            f"Search completed. Found {len(message_ids)} message IDs. "
            f"IDs: {preview_ids}{suffix}. "
            f"Use one of these IDs for read_email."
        )

    return truncate_for_prompt(cleaned_text, max_chars=2000)


def resolve_dynamic_arguments(arguments: dict[str, object], last_message_id: str | None) -> dict[str, object]:
    resolved = dict(arguments)
    for key, value in resolved.items():
        if isinstance(value, str) and value == "$LAST_MESSAGE_ID":
            resolved[key] = last_message_id or ""
    return resolved

    if isinstance(completed_value, bool):
        return completed_value, reason
    if isinstance(completed_value, str):
        return completed_value.strip().lower() in {"true", "yes", "1"}, reason

    return False, reason


async def run_direct_chat(
    user_prompt: str,
    system_prompt: str,
    model: str,
    base_url: str,
    temperature: float,
    show_llm_io: bool = False,
    debug_logger: DebugLogger | None = None,
    provider: str = "ollama",
    claude_model: str = "claude-opus-4-5",
    claude_api_key: str | None = None,
) -> None:
    # Direct mode bypasses MCP and sends the prompt straight to the configured LLM.
    llm = build_llm(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
        claude_model=claude_model,
        claude_api_key=claude_api_key,
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    if debug_logger:
        debug_logger.log("direct_chat_request", {"prompt": user_prompt, "system": system_prompt})
    if show_llm_io:
        print("Request messages:\n")
        for msg in messages:
            print(f"[{msg.type}] {msg.content}")
        print("\n--- End request messages ---\n")
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        content = " ".join(str(part) for part in content)
    if debug_logger:
        debug_logger.log("direct_chat_response", content)
    if show_llm_io:
        print("Raw response:\n")
        print(content or "<empty>")
        print("\n--- End raw response ---\n")
    print(content)


async def run_mcp_agent(
    request_text: str,
    model: str,
    base_url: str,
    temperature: float,
    top_tools: int,
    tool_ranking_prompt_file: str,
    max_agent_loops: int,
    show_llm_io: bool = False,
    debug_logger: DebugLogger | None = None,
    provider: str = "ollama",
    claude_model: str = "claude-opus-4-5",
    claude_api_key: str | None = None,
) -> None:
    # MCP mode first verifies credentials, then loads Gmail tools into the agent.
    oauth_keys_file = find_oauth_keys_file()
    if oauth_keys_file is None:
        print(
            "Gmail MCP server cannot start because gcp-oauth.keys.json was not found.",
            file=sys.stderr,
        )
        print(
            "Place the file in the current directory or in ~/.gmail-mcp/, then rerun the script.",
            file=sys.stderr,
        )
        sys.exit(1)

    credentials_file = find_credentials_file()
    if credentials_file is None:
        print("Gmail MCP credentials.json was not found. Run authentication first with:", file=sys.stderr)
        print("  npx @gongrzhe/server-gmail-autoauth-mcp auth", file=sys.stderr)
        sys.exit(1)

    mcp_command = os.getenv("MCP_COMMAND", "npx")
    mcp_args = [
        part
        for part in os.getenv("MCP_ARGS", "-y @gongrzhe/server-gmail-autoauth-mcp").split()
        if part
    ]

    llm = build_llm(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=temperature,
        claude_model=claude_model,
        claude_api_key=claude_api_key,
    )

    outputs: list[str] = []
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            last_message_id: str | None = None
            for step in chosen_tools:
                if not isinstance(step, dict):
                    continue

                name = step.get("name")
                arguments = step.get("arguments", {})
                if not isinstance(name, str) or not name:
                    continue
                if not isinstance(arguments, dict):
                    arguments = {}

                arguments = resolve_dynamic_arguments(arguments, last_message_id)

                # Guard against invalid read_email invocations (empty ID or email address value).
                if name == "read_email":
                    message_id = arguments.get("messageId")
                    if not is_valid_message_id(message_id) and is_valid_message_id(last_message_id):
                        arguments["messageId"] = last_message_id
                        message_id = last_message_id
                    if not is_valid_message_id(message_id):
                        outputs.append(
                            "Tool: read_email\n"
                            f"Arguments: {json.dumps(arguments)}\n"
                            "Skipped: messageId is missing or invalid. "
                            "Call search_emails first and use the returned message ID."
                        )
                        continue

                try:
                    if debug_logger:
                        debug_logger.log(
                            "mcp_tool_request",
                            {
                                "tool": name,
                                "arguments": arguments,
                            },
                        )
                    result = await session.call_tool(name, arguments)
                    result_text = tool_result_to_text(result)
                    cleaned_result_text = clean_email_body(result_text)
                    planning_result_text = summarize_tool_output_for_planning(
                        name,
                        cleaned_result_text,
                        result_text,
                    )
                    result_payload = tool_result_to_payload(result)
                    if debug_logger:
                        debug_logger.log(
                            "mcp_tool_response",
                            {
                                "tool": name,
                                "arguments": arguments,
                                "payload": result_payload,
                                "text": result_text,
                                "cleaned_text": cleaned_result_text,
                                "planning_text": planning_result_text,
                            },
                        )
                    maybe_message_id = extract_message_id(result_text)
                    if maybe_message_id:
                        last_message_id = maybe_message_id
                    outputs.append(
                        f"Tool: {name}\nArguments: {json.dumps(arguments)}\nResult:\n{planning_result_text}"
                    )
                except Exception as error:  # noqa: BLE001
                    if debug_logger:
                        debug_logger.log(
                            "mcp_tool_error",
                            {
                                "tool": name,
                                "arguments": arguments,
                                "error": str(error),
                            },
                        )
                    outputs.append(
                        f"Tool: {name}\nArguments: {json.dumps(arguments)}\nError: {error}"
                    )

    return "\n\n".join(outputs)


def request_planned_step(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
) -> str | None:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    tools = await mcp_client.get_tools()
    all_tools = list(tools)
    selected_tools = list(tools)

    if top_tools > 0:
        prompt_ranked = rank_tools_with_prompt(
            tools=tools,
            request=request_text,
            model=model,
            base_url=base_url,
            prompt_file=Path(tool_ranking_prompt_file).expanduser(),
            debug_logger=debug_logger,
            provider=provider,
            claude_model=claude_model,
            claude_api_key=claude_api_key,
        )
        ranked_tools = prompt_ranked if prompt_ranked is not None else rank_tools(tools, request_text)
        selected_tools = select_top_tools(
            all_tools=all_tools,
            ranked_tools=ranked_tools,
            request_text=request_text,
            top_tools=top_tools,
        )

    tools = selected_tools

def request_planned_step_with_retry(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    debug_logger: DebugLogger | None = None,
    retry_count: int = 2,
) -> str | None:
    current_user_prompt = user_prompt

    for attempt in range(retry_count):
        try:
            plan_content = request_planned_step(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=current_user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as error:  # noqa: BLE001
            if debug_logger:
                debug_logger.log(
                    "planning_attempt_error",
                    {
                        "attempt": attempt + 1,
                        "error": str(error),
                    },
                )
            plan_content = None

        if plan_content:
            plan_json = extract_planning_plan(plan_content)
            if plan_json is not None:
                return plan_content

        if debug_logger:
            debug_logger.log(
                "planning_attempt_invalid",
                {
                    "attempt": attempt + 1,
                    "content": plan_content,
                },
            )

        current_user_prompt = (
            f"{user_prompt}\n\n"
            "Your previous response was invalid, incomplete, or missing JSON. "
            "Return one complete JSON object only with the exact chosenTools shape. "
            "Do not add markdown, prose, or extra fields."
        )

    return None


def extract_planning_plan(plan_content: str | None) -> dict[str, object] | None:
    if not plan_content:
        return None

    if debug_logger:
        debug_logger.log("agent_request", {"request": request_text})

    for loop_index in range(1, max_loops + 1):
        response = await agent.ainvoke({"messages": conversation_messages})
        response_messages = response.get("messages", [])
        if isinstance(response_messages, list) and response_messages:
            conversation_messages = response_messages

        latest_ai_message = None
        if isinstance(response_messages, list):
            for message in reversed(response_messages):
                if isinstance(message, AIMessage):
                    latest_ai_message = message
                    break
        made_tool_calls = bool(getattr(latest_ai_message, "tool_calls", None))

        completed, completion_reason = should_end_agent_loop(
            request_text=request_text,
            messages=conversation_messages,
            model=model,
            base_url=base_url,
            debug_logger=debug_logger,
            provider=provider,
            claude_model=claude_model,
            claude_api_key=claude_api_key,
        )
        if debug_logger:
            debug_logger.log(
                "agent_loop_check",
                {
                    "loop": loop_index,
                    "completed": completed,
                    "reason": completion_reason,
                },
            )

        if completed:
            break

        if not made_tool_calls:
            if debug_logger:
                debug_logger.log(
                    "agent_loop_stopped",
                    {
                        "loop": loop_index,
                        "reason": "No tool calls were produced, so the agent cannot make further progress without new user input.",
                    },
                )
            break

    for message in conversation_messages:
        if debug_logger:
            debug_logger.log(
                "agent_message",
                {"type": message.type, "content": message.content},
            )
            if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                debug_logger.log(
                    "agent_tool_call",
                    {
                        "tool_calls": getattr(message, "tool_calls", []),
                    },
                )
            if getattr(message, "type", "") == "tool":
                debug_logger.log(
                    "tool_return",
                    {
                        "name": getattr(message, "name", ""),
                        "content": message.content,
                    },
                )
        if show_llm_io:
            print(f"[{message.type}] {message.content}\n")
        elif isinstance(message, AIMessage) and message.content and not getattr(message, "tool_calls", None):
            print(message.content)


def extract_chosen_tools(plan_content: str | None) -> dict[str, object] | None:
    plan_json = extract_planning_plan(plan_content)
    if plan_json is None:
        return None

    return plan_json


def main() -> None:
    args = parse_args()
    debug_logger = DebugLogger(
        enabled=args.debug,
        log_file=Path(args.debug_log_file).expanduser() if args.debug else None,
    )
    if debug_logger.enabled and debug_logger.log_file is not None:
        # Record the top-level invocation details before any tool or model work begins.
        debug_logger.log(
            "script_start",
            {
                "request": args.prompt or args.mcp_request or "",
                "model": args.model,
                "temperature": args.temperature,
                "mcp_request": bool(args.mcp_request),
            },
        )

    user_prompt = args.prompt
    if args.mcp_request:
        try:
            log_section(debug_logger, "generate_mcp_prompt", "start", {"request": args.mcp_request})
            user_prompt = build_prompt_from_mcp_request(args.mcp_request, args.top_tools, debug_logger)
            log_section(
                debug_logger,
                "generate_mcp_prompt",
                "end",
                {"prompt": user_prompt},
            )
        except Exception as error:
            if debug_logger.enabled:
                debug_logger.log("script_error", {"stage": "generate_mcp_prompt", "error": str(error)})
            print(f"Failed to build MCP prompt: {error}", file=sys.stderr)
            sys.exit(1)

        if args.show_generated_prompt:
            print("Generated MCP prompt:\n")
            print(user_prompt)
            print("\n--- End generated prompt ---\n")

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )

    planning_system = build_planning_system_prompt() if args.mcp_request else args.system

    if not args.mcp_request:
        try:
            log_section(debug_logger, "single_prompt_llm_call", "start", {"prompt": user_prompt})
            planning_payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": planning_system},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            if args.show_llm_io:
                print("Planning request payload:\n")
                print(json.dumps(planning_payload, indent=2))
                print("\n--- End planning request payload ---\n")

            if debug_logger.enabled:
                debug_logger.log("single_prompt_request", planning_payload)

            plan_response = client.chat.completions.create(
                model=planning_payload["model"],
                messages=planning_payload["messages"],
                temperature=planning_payload["temperature"],
                max_tokens=planning_payload["max_tokens"],
            )
            if debug_logger.enabled:
                debug_logger.log(
                    "single_prompt_response_raw",
                    plan_response.model_dump() if hasattr(plan_response, "model_dump") else str(plan_response),
                )
        except Exception as error:
            if debug_logger.enabled:
                debug_logger.log("script_error", {"stage": "single_prompt_llm_call", "error": str(error)})
            print(f"Request failed: {error}", file=sys.stderr)
            print(
                "Tip: ensure Ollama is running and the model exists, e.g. `ollama run gemma4`.",
                file=sys.stderr,
            )
            sys.exit(1)

        plan_content = plan_response.choices[0].message.content if plan_response.choices else None
        if args.show_llm_io:
            print("Planning raw response:\n")
            print(plan_content or "<empty>")
            print("\n--- End planning raw response ---\n")

        if debug_logger.enabled:
            debug_logger.log("single_prompt_response_content", plan_content)

        if not plan_content:
            print("No content returned by model.")
            return

        print(plan_content)
        return

    tool_outputs: list[str] = []
    current_prompt = build_step_prompt(args.mcp_request, user_prompt)
    max_rounds = 10

    for round_index in range(max_rounds):
        try:
            section_title = f"planning_round_{round_index + 1}"
            log_section(debug_logger, section_title, "start", {"prompt": current_prompt})
            plan_content = request_planned_step_with_retry(
                client=client,
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                debug_logger=debug_logger,
            )
            if debug_logger.enabled:
                debug_logger.log(
                    "planning_round_response_raw",
                    {
                        "title": section_title,
                        "content": plan_content,
                    },
                )
        except Exception as error:
            if debug_logger.enabled:
                debug_logger.log("script_error", {"stage": f"planning_round_{round_index + 1}", "error": str(error)})
            print(f"Request failed: {error}", file=sys.stderr)
            print(
                "Tip: ensure Ollama is running and the model exists, e.g. `ollama run gemma4`.",
                file=sys.stderr,
            )
            sys.exit(1)
        finally:
            log_section(debug_logger, f"planning_round_{round_index + 1}", "end")

        if args.show_llm_io:
            print(f"Planning raw response (round {round_index + 1}):\n")
            print(plan_content or "<empty>")
            print("\n--- End planning raw response ---\n")

        if debug_logger.enabled:
            debug_logger.log(
                "planning_round_response_content",
                {
                    "title": f"planning_round_{round_index + 1}",
                    "content": plan_content,
                },
            )

        if not plan_content:
            print("No content returned by model.")
            return

        plan_json = extract_chosen_tools(plan_content)
        if not plan_json:
            print("Model did not return a valid JSON plan. Raw output:\n")
            print(plan_content)
            return

        chosen_tools = plan_json.get("chosenTools")
        if not isinstance(chosen_tools, list) or not chosen_tools:
            break

        try:
            log_section(debug_logger, f"tool_execution_round_{round_index + 1}", "start", {"plan": plan_json})
            round_output = asyncio.run(execute_mcp_tools_from_plan(plan_json, debug_logger))
            if debug_logger.enabled:
                debug_logger.log(
                    "tool_execution_round_output",
                    {
                        "title": f"tool_execution_round_{round_index + 1}",
                        "output": round_output,
                    },
                )
        except Exception as error:  # noqa: BLE001
            if debug_logger.enabled:
                debug_logger.log("script_error", {"stage": f"tool_execution_round_{round_index + 1}", "error": str(error)})
            print(f"Tool execution failed: {error}", file=sys.stderr)
            return
        finally:
            log_section(debug_logger, f"tool_execution_round_{round_index + 1}", "end")

        tool_outputs.append(round_output)

        current_prompt = build_step_prompt(
            args.mcp_request,
            user_prompt,
            "\n\n".join(tool_outputs),
        )
    else:
        asyncio.run(
            run_mcp_agent(
                request_text=args.mcp_request,
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                top_tools=args.top_tools,
                tool_ranking_prompt_file=args.tool_ranking_prompt_file,
                max_agent_loops=args.max_agent_loops,
                show_llm_io=args.show_llm_io,
                debug_logger=debug_logger,
                provider=args.provider,
                claude_model=args.claude_model,
                claude_api_key=args.claude_api_key,
            )
        )

    if debug_logger.enabled:
        debug_logger.log("script_end", {"status": "ok"})


if __name__ == "__main__":
    main()