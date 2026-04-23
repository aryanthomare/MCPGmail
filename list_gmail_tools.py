from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage




@dataclass(frozen=True)
class RankedTool:
    name: str
    description: str
    input_schema: dict[str, object]
    score: int
    reason: str


@dataclass
class DebugLogger:
    enabled: bool
    log_file: Path | None

    def log(self, event: str, payload: object) -> None:
        if not self.enabled or self.log_file is None:
            return

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str))
            handle.write("\n")


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in text.lower().replace("/", " ").replace("_", " ").split()
        if token
    }


def find_oauth_keys_file() -> Path | None:
    candidates = [
        Path.cwd() / "gcp-oauth.keys.json",
        Path.home() / ".gmail-mcp" / "gcp-oauth.keys.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def find_credentials_file() -> Path | None:
    candidates = [
        Path.cwd() / "credentials.json",
        Path.home() / ".gmail-mcp" / "credentials.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def server_working_directory() -> Path:
    temp_dir = Path(os.getenv("TEMP", str(Path.home()))) / "gmail-mcp-client"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


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


def get_tool_schema(tool: object) -> dict[str, object]:
    """Extract the input schema from a LangChain tool (args_schema) or fall back to {}."""
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return {}
    if hasattr(args_schema, "model_json_schema"):
        return args_schema.model_json_schema()  # type: ignore[return-value]
    if hasattr(args_schema, "schema"):
        return args_schema.schema()  # type: ignore[return-value]
    return {}


def build_ranking_prompt(request: str, tools: list[object]) -> str:
    tool_catalog = [
        {
            "name": getattr(tool, "name", "") or "",
            "description": getattr(tool, "description", "") or "",
        }
        for tool in tools
    ]

    return "\n\n".join(
        [
            "You are a strict tool-ranking assistant for a Gmail MCP client.",
            "For EACH tool below, decide if it is relevant to the user request.",
            "Write ONE sentence explaining your decision (why relevant or why not).",
            "Return JSON only. No markdown, no explanation, no backticks.",
            'Use this exact shape: {"rankedTools":[{"name":"tool_name","reason":"why"}],"skipped":["tool_name: reason","tool_name: reason"]}',
            "Rules:",
            "- Evaluate ALL tools in the catalog.",
            "- rankedTools: tools that ARE relevant, ordered from most to least important.",
            "- skipped: tools that are NOT relevant, with one-sentence reasons.",
            "- Do not invent tool names.",
            "- Use tool name and description only.",
            "Available tools:",
            json.dumps(tool_catalog, indent=2),
            "User request:",
            request,
        ]
    )


def rank_tools_with_local_model(
    tools: list[object],
    request: str,
    debug_logger: DebugLogger | None = None,
) -> list[RankedTool]:
    if not tools:
        return []

    base_url = os.getenv(
        "LOCAL_LLM_BASE_URL",
        os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    model = os.getenv(
        "LOCAL_LLM_MODEL",
        os.getenv("OLLAMA_MODEL", "gemma4"),
    )

    llm = ChatOllama(model=model, base_url=base_url, temperature=0.0, format="json")
    prompt = build_ranking_prompt(request, tools)
    if debug_logger:
        debug_logger.log(
            "local_model_prompt",
            {
                "model": model,
                "base_url": base_url,
                "prompt": prompt,
            },
        )
    tool_by_name = {
        getattr(tool, "name", "") or "": tool
        for tool in tools
        if getattr(tool, "name", "")
    }
    last_error: str | None = None
    last_content: str | None = None

    def request_ranking(prompt_text: str, prefix: str) -> list[object] | None:
        nonlocal last_error
        nonlocal last_content
        try:
            messages = [
                SystemMessage(content="You are a tool-ranking assistant. Return valid JSON only with the requested shape."),
                HumanMessage(content=prompt_text),
            ]
            response = llm.invoke(messages)
        except Exception as error:  # noqa: BLE001
            last_error = str(error)
            if debug_logger:
                debug_logger.log(f"{prefix}_error", str(error))
            return None

        content = response.content if hasattr(response, "content") else None
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        last_content = content
        if debug_logger:
            debug_logger.log(f"{prefix}_response_content", content)
        if not content:
            return None

        plan = extract_json_object(content)
        if not plan:
            return None

        ranked_entries = plan.get("rankedTools")
        if not isinstance(ranked_entries, list):
            return None

        return ranked_entries

    ranked_entries = request_ranking(prompt, "local_model")

    # Parse and log skipped tools from the JSON response if available
    skipped_tools = []
    if last_content:
        response_json = extract_json_object(last_content)
        if response_json and isinstance(response_json.get("skipped"), list):
            skipped_tools = response_json.get("skipped", [])
            if debug_logger and skipped_tools:
                debug_logger.log("model_skipped_tools", skipped_tools)

    if not ranked_entries:
        if debug_logger:
            debug_logger.log(
                "local_model_no_tools",
                {
                    "message": "Local model returned no ranked tools.",
                    "last_error": last_error,
                    "last_content": last_content,
                    "skipped_reasoning": skipped_tools,
                },
            )
        return []

    ranked_tools: list[RankedTool] = []
    seen_names: set[str] = set()
    total_tools = len(tools)

    for index, entry in enumerate(ranked_entries):
        if not isinstance(entry, dict):
            continue

        name = entry.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            continue

        tool = tool_by_name.get(name)
        if tool is None:
            continue

        seen_names.add(name)
        description = getattr(tool, "description", "") or ""
        input_schema = get_tool_schema(tool)
        reason = entry.get("reason")
        ranked_tools.append(
            RankedTool(
                name=name,
                description=description,
                input_schema=input_schema,
                score=max(total_tools - index, 1),
                reason=reason if isinstance(reason, str) and reason else "Selected by local model ranking.",
            )
        )

    if not ranked_tools:
        if debug_logger:
            debug_logger.log(
                "local_model_no_tools",
                {
                    "message": "Local model returned ranked tools, but none matched known tool names.",
                    "ranked_entries": ranked_entries,
                },
            )
        return []

    return ranked_tools


def rank_tools(tools: list[object], request: str) -> list[RankedTool]:
    request_tokens = tokenize(request)
    ranked_tools: list[RankedTool] = []

    for tool in tools:
        name = getattr(tool, "name", "") or ""
        description = getattr(tool, "description", "") or ""
        input_schema = get_tool_schema(tool)
        haystack = tokenize(f"{name} {description} {json.dumps(input_schema, default=str)}")

        matches = sorted(request_tokens.intersection(haystack))
        score = sum(2 if len(token) >= 4 else 1 for token in matches)
        if request_tokens & {"email", "mail", "inbox", "attachment", "label", "draft"}:
            if {"email", "mail", "inbox", "attachment", "label", "draft"} & haystack:
                score += 2

        ranked_tools.append(
            RankedTool(
                name=name,
                description=description,
                input_schema=input_schema,
                score=score,
                reason=(
                    f"Matched: {', '.join(matches)}"
                    if matches
                    else "No direct lexical match; kept as a fallback."
                ),
            )
        )

    return sorted(ranked_tools, key=lambda item: (-item.score, item.name))


def build_prompt(request: str, tools: list[object]) -> str:
    tool_catalog = [
        {
            "name": getattr(tool, "name", "") or "",
            "description": getattr(tool, "description", "") or "",
            "inputSchema": get_tool_schema(tool),
        }
        for tool in tools
    ]

    return "\n\n".join(
        [
            "You are a tool-planning assistant for a Gmail MCP client.",
            "Select the smallest set of tools needed to satisfy the user request.",
            "Return JSON only. No markdown, no explanation, no backticks.",
            'Use this exact shape: {"chosenTools":[{"name":"tool_name","reason":"why","arguments":{}}]}',
            "Rules:",
            "- Only use tools from the provided catalog.",
            "- If no tool is needed, return an empty chosenTools array.",
            "- If you choose a tool, include only arguments that are supported by its schema.",
            "Relevant tools:",
            json.dumps(tool_catalog, indent=2),
            "User request:",
            request,
        ]
    )


async def fetch_tools(
    command: str,
    args: list[str],
    oauth_keys_file: Path,
    credentials_file: Path,
) -> list[object]:
    async with MultiServerMCPClient(
        {
            "gmail": {
                "command": command,
                "args": args,
                "transport": "stdio",
                "env": {
                    **os.environ,
                    "GMAIL_OAUTH_PATH": str(oauth_keys_file),
                    "GMAIL_CREDENTIALS_PATH": str(credentials_file),
                },
                "cwd": str(server_working_directory()),
            }
        }
    ) as client:
        return list(await client.get_tools())


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print Gmail MCP tools and rank the best matches for a request."
    )
    parser.add_argument(
        "request",
        nargs="*",
        help="Optional request text used to rank the tools.",
    )
    parser.add_argument(
        "--command",
        default=os.getenv("MCP_COMMAND", "npx"),
        help="Command used to start the MCP server.",
    )
    parser.add_argument(
        "--args",
        default=os.getenv("MCP_ARGS", "-y @gongrzhe/server-gmail-autoauth-mcp"),
        help="Arguments used to start the MCP server.",
    )
    parser.add_argument(
        "--auth",
        action="store_true",
        help="Run the Gmail MCP auth command instead of listing tools.",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Print the generated prompt and exit after local-model tool ranking.",
    )
    parser.add_argument(
        "--top-tools",
        type=int,
        default=int(os.getenv("TOP_TOOLS", "0")),
        help="How many top-ranked tools to include in the generated prompt context. Use 0 or omit the env var to include all tools.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log prompts and responses to a debug log file.",
    )
    parser.add_argument(
        "--debug-log-file",
        default=os.getenv("MCP_DEBUG_LOG_FILE", "mcp_tool_debug.log"),
        help="Path to the debug log file (used with --debug).",
    )
    args = parser.parse_args()

    debug_logger = DebugLogger(
        enabled=args.debug,
        log_file=Path(args.debug_log_file).expanduser() if args.debug else None,
    )
    if debug_logger.enabled and debug_logger.log_file is not None:
        print(f"Debug logging enabled: {debug_logger.log_file}")
        debug_logger.log(
            "script_start",
            {
                "request": " ".join(args.request),
                "top_tools": args.top_tools,
                "command": args.command,
                "args": args.args,
            },
        )

    request_text = " ".join(args.request).strip()
    server_args = [part for part in args.args.split() if part]

    oauth_keys_file = find_oauth_keys_file()
    if oauth_keys_file is None:
        print(
            "Gmail MCP server cannot start because gcp-oauth.keys.json was not found."
        )
        print(
            "Place the file in the current directory or in ~/.gmail-mcp/, then rerun the script."
        )
        return

    print(f"Using OAuth keys file: {oauth_keys_file}")

    credentials_file = find_credentials_file()
    if credentials_file is None:
        print(
            "Gmail MCP credentials.json was not found. Run authentication first with:"
        )
        print("  npx @gongrzhe/server-gmail-autoauth-mcp auth")
        print(
            "After the browser flow completes, rerun this script to list tools over MCP."
        )
        return

    print(f"Using credentials file: {credentials_file}")

    if args.auth:
        auth_command = [args.command, *server_args, "auth"]
        print("Running Gmail MCP authentication:")
        print(" ".join(auth_command))
        completed = subprocess.run(
            auth_command,
            check=False,
            env={
                **os.environ,
                "GMAIL_OAUTH_PATH": str(oauth_keys_file),
                "GMAIL_CREDENTIALS_PATH": str(credentials_file),
            },
            cwd=server_working_directory(),
        )
        sys.exit(completed.returncode)

    if not request_text:
        print('Usage: python list_gmail_tools.py [--prompt-only] "your request here"')
        return

    tools = await fetch_tools(args.command, server_args, oauth_keys_file, credentials_file)
    if debug_logger.enabled:
        debug_logger.log(
            "mcp_list_tools_response",
            [
                {
                    "name": getattr(tool, "name", "") or "",
                    "description": getattr(tool, "description", "") or "",
                }
                for tool in tools
            ],
        )
        debug_logger.log(
            "mcp_list_tools_response_raw",
            {
                "tools": [
                    {
                        "name": getattr(tool, "name", "") or "",
                        "description": getattr(tool, "description", "") or "",
                        "inputSchema": get_tool_schema(tool),
                    }
                    for tool in tools
                ],
            },
        )

    try:
        ranked = rank_tools_with_local_model(tools, request_text, debug_logger)
    except RuntimeError as error:
        if debug_logger.enabled:
            debug_logger.log("ranking_error", str(error))
            debug_logger.log("script_end", {"error": str(error)})
        print(str(error), file=sys.stderr)
        sys.exit(1)
    if debug_logger.enabled:
        debug_logger.log(
            "ranked_tools",
            [
                {
                    "name": item.name,
                    "score": item.score,
                    "reason": item.reason,
                }
                for item in ranked
            ],
        )
    top_tools = None if args.top_tools <= 0 else args.top_tools
    tool_by_name = {
        getattr(tool, "name", "") or "": tool
        for tool in tools
        if getattr(tool, "name", "")
    }
    prompt_tools = [
        tool_by_name[ranked_tool.name]
        for ranked_tool in (ranked if top_tools is None else ranked[:top_tools])
        if ranked_tool.name in tool_by_name
    ]

    prompt = build_prompt(request_text, prompt_tools)
    if debug_logger.enabled:
        debug_logger.log("planner_prompt", prompt)

    ranked_display = ranked if top_tools is None else ranked[:top_tools]

    print("Discovered MCP tools:")
    for tool in tools:
        name = getattr(tool, "name", "") or ""
        description = getattr(tool, "description", "") or ""
        print(f"- {name}{f': {description}' if description else ''}")

    print()
    print(f"Tool ranking for request: {request_text}")
    for ranked_tool in ranked_display:
        print(f"- {ranked_tool.name} [score {ranked_tool.score}] {ranked_tool.reason}")

    print()
    if top_tools is None:
        print(f"Prompt tool context size: {len(prompt_tools)} tool(s) (all tools)")
    else:
        print(f"Prompt tool context size: {len(prompt_tools)} tool(s) (top-tools={top_tools})")

    print()
    print("Ollama prompt:")
    print(prompt)

    if args.prompt_only:
        if debug_logger.enabled:
            debug_logger.log("script_end", {"prompt_only": True})
        return

    if debug_logger.enabled:
        debug_logger.log("script_end", {"prompt_only": False})


if __name__ == "__main__":
    asyncio.run(main())