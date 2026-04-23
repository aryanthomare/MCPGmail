from __future__ import annotations

import argparse
import asyncio
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


def parse_args() -> argparse.Namespace:
    # Keep the CLI focused on one of two execution modes: direct chat or MCP-backed chat.
    parser = argparse.ArgumentParser(
        description="Call a local Ollama model via LangChain and optionally run Gmail MCP tools."
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
        default=int(os.getenv("TOP_TOOLS", "0")),
        help="How many top-ranked tools to include in the agent context. Use 0 or omit the env var to include all tools.",
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
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        return payload
    return None


def build_mcp_agent_system_prompt(request_text: str) -> str:
    base_prompt = (
        "You are a Gmail assistant with access to MCP tools. "
        "Use search_emails to find candidate messages, then read_email to inspect them. "
        "For date-based requests, use Gmail search syntax with after:YYYY/MM/DD and before:YYYY/MM/DD. "
        "Do not use date: queries. "
        "If the user asks for all emails on a specific day, search the full day range first, then read each result."
    )

    match = re.search(r"\b(\d{1,2}/\d{1,2}/\d{4})\b", request_text)
    if match:
        try:
            request_date = datetime.strptime(match.group(1), "%m/%d/%Y").date()
            next_day = request_date + timedelta(days=1)
            return (
                f"{base_prompt} "
                f"For this request, the likely Gmail query is after:{request_date:%Y/%m/%d} before:{next_day:%Y/%m/%d}."
            )
        except ValueError:
            pass

    match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", request_text)
    if match:
        try:
            request_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
            next_day = request_date + timedelta(days=1)
            return (
                f"{base_prompt} "
                f"For this request, the likely Gmail query is after:{request_date:%Y/%m/%d} before:{next_day:%Y/%m/%d}."
            )
        except ValueError:
            pass

    return base_prompt


COMPLETION_CHECK_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "completed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["completed", "reason"],
    "additionalProperties": False,
}

TOOL_RANKING_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "rankedTools": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["name", "reason"],
                "additionalProperties": True,
            },
        }
    },
    "required": ["rankedTools"],
    "additionalProperties": True,
}


def rank_tools(tools: list[object], request: str) -> list[object]:
    # Keep the ranking lightweight so the main agent can trim tools without a second model call.
    request_lower = request.lower()
    request_tokens = {
        token
        for token in request_lower.replace("/", " ").replace("_", " ").split()
        if token
    }
    read_intent = (
        any(word in request_lower for word in {"read", "view", "summar", "show", "list"})
        and any(word in request_lower for word in {"email", "mail", "inbox"})
    )
    ranked_tools: list[tuple[int, str, object]] = []

    for tool in tools:
        name = getattr(tool, "name", "") or ""
        description = getattr(tool, "description", "") or ""
        schema = getattr(tool, "args_schema", None)
        haystack = {
            token
            for token in f"{name} {description} {json.dumps(schema, default=str)}".lower().replace("/", " ").replace("_", " ").split()
            if token
        }
        matches = sorted(request_tokens.intersection(haystack))
        score = sum(2 if len(token) >= 4 else 1 for token in matches)
        if request_tokens & {"email", "mail", "inbox", "attachment", "label", "draft"}:
            if {"email", "mail", "inbox", "attachment", "label", "draft"} & haystack:
                score += 2

        if read_intent:
            if name in {"search_emails", "read_email"}:
                score += 100
            elif name in {"list_email_labels", "download_attachment"}:
                score += 10
            elif name in {
                "send_email",
                "draft_email",
                "modify_email",
                "delete_email",
                "batch_modify_emails",
                "batch_delete_emails",
                "create_label",
                "update_label",
                "delete_label",
                "get_or_create_label",
                "create_filter",
                "list_filters",
                "get_filter",
                "delete_filter",
                "create_filter_from_template",
            }:
                score -= 20

        ranked_tools.append((score, name, tool))

    return [tool for _, _, tool in sorted(ranked_tools, key=lambda item: (-item[0], item[1]))]


def build_ranking_prompt(template: str, request: str, tools: list[object]) -> str:
    catalog = [
        {
            "name": getattr(tool, "name", "") or "",
            "description": getattr(tool, "description", "") or "",
            "inputSchema": get_tool_schema(tool),
        }
        for tool in tools
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
) -> list[object] | None:
    if not prompt_file.exists():
        return None

    template = prompt_file.read_text(encoding="utf-8")
    prompt = build_ranking_prompt(template, request, tools)
    llm = ChatOllama(model=model, base_url=base_url, temperature=0.0).with_structured_output(
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

    ranked_entries = response.get("rankedTools")
    if not isinstance(ranked_entries, list):
        if debug_logger:
            debug_logger.log("tool_ranking_unparseable", response)
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


def should_end_agent_loop(
    request_text: str,
    messages: list[object],
    model: str,
    base_url: str,
    debug_logger: DebugLogger | None = None,
) -> tuple[bool, str]:
    checker = ChatOllama(model=model, base_url=base_url, temperature=0.0).with_structured_output(
        COMPLETION_CHECK_SCHEMA,
        method="json_schema",
    )
    transcript = format_messages_for_completion_check(messages)
    prompt = "\n\n".join(
        [
            "Decide whether the task is complete based on the conversation so far.",
            "Return a structured decision with completed and reason fields.",
            "Task request:",
            request_text,
            "Conversation transcript:",
            transcript,
        ]
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

    completed_value = response.get("completed")
    reason_value = response.get("reason")
    reason = reason_value if isinstance(reason_value, str) and reason_value else "No reason provided."

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
) -> None:
    # Direct mode bypasses MCP and sends the prompt straight to Ollama.
    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)
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

    llm = ChatOllama(model=model, base_url=base_url, temperature=temperature)

    # The adapter now exposes tools directly; it is no longer an async context manager.
    mcp_client = MultiServerMCPClient(
        {
            "gmail": {
                "command": mcp_command,
                "args": mcp_args,
                "transport": "stdio",
                "env": {
                    **os.environ,
                    "GMAIL_OAUTH_PATH": str(oauth_keys_file),
                    "GMAIL_CREDENTIALS_PATH": str(credentials_file),
                },
                "cwd": str(server_working_directory()),
            }
        }
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
        )
        ranked_tools = prompt_ranked if prompt_ranked is not None else rank_tools(tools, request_text)
        selected_tools = select_top_tools(
            all_tools=all_tools,
            ranked_tools=ranked_tools,
            request_text=request_text,
            top_tools=top_tools,
        )

    tools = selected_tools

    if debug_logger:
        debug_logger.log("mcp_tools_all", [t.name for t in all_tools])
        debug_logger.log("mcp_tools", [t.name for t in tools])

    if show_llm_io:
        print("Discovered MCP tools (all):")
        for tool in all_tools:
            print(f"- {tool.name}: {tool.description or ''}")
        print()
        if top_tools > 0:
            print(f"Selected MCP tools (top {top_tools}):")
            for tool in tools:
                print(f"- {tool.name}: {tool.description or ''}")
            print()

    agent = create_agent(llm, tools)
    conversation_messages: list[object] = [
        SystemMessage(content=build_mcp_agent_system_prompt(request_text)),
        HumanMessage(content=request_text),
    ]
    max_loops = max(1, max_agent_loops)

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

    if args.prompt:
        asyncio.run(
            run_direct_chat(
                user_prompt=args.prompt,
                system_prompt=args.system,
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                show_llm_io=args.show_llm_io,
                debug_logger=debug_logger,
            )
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
            )
        )

    if debug_logger.enabled:
        debug_logger.log("script_end", {"status": "ok"})


if __name__ == "__main__":
    main()