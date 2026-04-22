from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent


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


async def run_direct_chat(
    user_prompt: str,
    system_prompt: str,
    model: str,
    base_url: str,
    temperature: float,
    show_llm_io: bool = False,
    debug_logger: DebugLogger | None = None,
) -> None:
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
    show_llm_io: bool = False,
    debug_logger: DebugLogger | None = None,
) -> None:
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

    async with MultiServerMCPClient(
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
    ) as mcp_client:
        tools = mcp_client.get_tools()

        if debug_logger:
            debug_logger.log("mcp_tools", [t.name for t in tools])

        if show_llm_io:
            print("Discovered MCP tools:")
            for tool in tools:
                print(f"- {tool.name}: {tool.description or ''}")
            print()

        agent = create_react_agent(llm, tools)

        if debug_logger:
            debug_logger.log("agent_request", {"request": request_text})

        response = await agent.ainvoke({"messages": [HumanMessage(content=request_text)]})

        for message in response["messages"]:
            if debug_logger:
                debug_logger.log(
                    "agent_message",
                    {"type": message.type, "content": message.content},
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
                show_llm_io=args.show_llm_io,
                debug_logger=debug_logger,
            )
        )

    if debug_logger.enabled:
        debug_logger.log("script_end", {"status": "ok"})


if __name__ == "__main__":
    main()