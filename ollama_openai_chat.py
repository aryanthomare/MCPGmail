from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client
from openai import OpenAI


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


def log_section(debug_logger: DebugLogger | None, title: str, phase: str, payload: object | None = None) -> None:
    if debug_logger is None:
        return

    debug_logger.log(
        f"section_{phase}",
        {
            "title": title,
            "payload": payload,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call a local Ollama model through the OpenAI-compatible API and print the response."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompt",
        help="User prompt to send to the model.",
    )
    input_group.add_argument(
        "--mcp-request",
        help="Request text used to generate an MCP planning prompt via list_gmail_tools.py, then send it to Ollama.",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="Optional system instruction.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_OPENAI_MODEL", "gemma4"),
        help="Model name served by Ollama.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1"),
        help="OpenAI-compatible base URL for Ollama.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OLLAMA_OPENAI_API_KEY", "ollama"),
        help="API key value required by the OpenAI client (placeholder is fine for local Ollama).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Maximum tokens in the response.",
    )
    parser.add_argument(
        "--show-generated-prompt",
        action="store_true",
        help="Print the generated MCP prompt before sending it to Ollama (only used with --mcp-request).",
    )
    parser.add_argument(
        "--show-llm-io",
        action="store_true",
        help="Print request payloads and raw model responses for debugging.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="For --mcp-request, print the model plan only and skip MCP tool execution.",
    )
    parser.add_argument(
        "--top-tools",
        type=int,
        default=int(os.getenv("TOP_TOOLS", "6")),
        help="How many top-ranked tools to include in generated MCP prompt context.",
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


def extract_generated_prompt(stdout: str) -> str:
    marker = "Ollama prompt:\n"
    index = stdout.find(marker)
    if index == -1:
        raise ValueError("Could not find 'Ollama prompt:' in list_gmail_tools.py output.")

    prompt = stdout[index + len(marker) :].strip()
    if not prompt:
        raise ValueError("Generated MCP prompt was empty.")

    return prompt


def build_prompt_from_mcp_request(
    request_text: str,
    top_tools: int,
    debug_logger: DebugLogger | None = None,
) -> str:
    script_path = Path(__file__).with_name("list_gmail_tools.py")
    command = [
        sys.executable,
        str(script_path),
        "--prompt-only",
        "--top-tools",
        str(max(1, top_tools)),
        request_text,
    ]

    env = None
    if debug_logger and debug_logger.enabled and debug_logger.log_file is not None:
        command.insert(2, "--debug")
        command.extend(["--debug-log-file", str(debug_logger.log_file)])
        env = {
            **os.environ,
            "MCP_DEBUG_LOG_FILE": str(debug_logger.log_file),
        }

    if debug_logger:
        debug_logger.log(
            "subprocess_start",
            {
                "title": "generate_mcp_prompt",
                "command": command,
            },
        )

    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    if debug_logger:
        debug_logger.log(
            "subprocess_stdout",
            {
                "title": "generate_mcp_prompt",
                "stdout": completed.stdout,
            },
        )
        debug_logger.log(
            "subprocess_stderr",
            {
                "title": "generate_mcp_prompt",
                "stderr": completed.stderr,
            },
        )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "No stderr captured."
        raise RuntimeError(f"Prompt generation failed: {stderr}")

    return extract_generated_prompt(completed.stdout)


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
    if tool_outputs:
        sections.extend(
            [
                "",
                "Tool outputs so far:",
                tool_outputs,
            ]
        )

    return "\n".join(sections)


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


def extract_message_id(text: str) -> str | None:
    match = re.search(r"\bID:\s*([A-Za-z0-9_-]+)", text)
    if match:
        return match.group(1)
    return None


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


def is_valid_message_id(value: object) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if not candidate:
        return False
    if "@" in candidate:
        return False
    return True


async def execute_mcp_tools_from_plan(
    plan: dict[str, object],
    debug_logger: DebugLogger | None = None,
) -> str:
    chosen_tools = plan.get("chosenTools")
    if not isinstance(chosen_tools, list) or not chosen_tools:
        return "No tools selected by the model plan."

    oauth_keys_file = find_oauth_keys_file()
    credentials_file = find_credentials_file()
    if oauth_keys_file is None or credentials_file is None:
        raise RuntimeError(
            "Missing OAuth files. Ensure gcp-oauth.keys.json and credentials.json are configured."
        )

    mcp_command = os.getenv("MCP_COMMAND", "npx")
    mcp_args = [
        part
        for part in os.getenv("MCP_ARGS", "-y @gongrzhe/server-gmail-autoauth-mcp").split()
        if part
    ]

    server_params = StdioServerParameters(
        command=mcp_command,
        args=mcp_args,
        env={
            **os.environ,
            "GMAIL_OAUTH_PATH": str(oauth_keys_file),
            "GMAIL_CREDENTIALS_PATH": str(credentials_file),
        },
        cwd=server_working_directory(),
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
    return response.choices[0].message.content if response.choices else None


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

    plan_json = extract_json_object(plan_content)
    if not plan_json:
        return None

    chosen_tools = plan_json.get("chosenTools")
    if not isinstance(chosen_tools, list):
        return None

    return plan_json


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
        debug_logger.log(
            "script_start",
            {
                "request": args.prompt or args.mcp_request or "",
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_tools": args.top_tools,
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
                system_prompt=planning_system,
                user_prompt=current_prompt,
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

    if not tool_outputs:
        print("No tools selected by the model plan.")
        return

    final_prompt = build_final_answer_prompt(args.mcp_request, "\n\n".join(tool_outputs))

    if args.show_llm_io:
        print("Executed tool outputs:\n")
        print(tool_outputs)
        print("\n--- End executed tool outputs ---\n")

    try:
        log_section(debug_logger, "final_summary", "start", {"prompt": final_prompt})
        final_payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a Gmail assistant. Use tool results as source of truth.",
                },
                {"role": "user", "content": final_prompt},
            ],
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        if args.show_llm_io:
            print("Final request payload:\n")
            print(json.dumps(final_payload, indent=2))
            print("\n--- End final request payload ---\n")

        if debug_logger.enabled:
            debug_logger.log("final_summary_request", final_payload)

        final_response = client.chat.completions.create(
            model=final_payload["model"],
            messages=final_payload["messages"],
            temperature=final_payload["temperature"],
            max_tokens=final_payload["max_tokens"],
        )
        if debug_logger.enabled:
            debug_logger.log(
                "final_summary_response_raw",
                final_response.model_dump() if hasattr(final_response, "model_dump") else str(final_response),
            )
    except Exception as error:
        if debug_logger.enabled:
            debug_logger.log("script_error", {"stage": "final_summary", "error": str(error)})
        print(f"Final summarization failed: {error}", file=sys.stderr)
        print("Tool outputs were:\n")
        print(tool_outputs)
        return
    finally:
        log_section(debug_logger, "final_summary", "end")

    final_content = final_response.choices[0].message.content if final_response.choices else None
    if args.show_llm_io:
        print("Final raw response:\n")
        print(final_content or "<empty>")
        print("\n--- End final raw response ---\n")

    if debug_logger.enabled:
        debug_logger.log("final_summary_response_content", final_content)
        debug_logger.log("script_end", {"status": "ok"})

    if not final_content:
        print("No final content returned by model. Tool outputs were:\n")
        print(tool_outputs)
        return

    print(final_content)


if __name__ == "__main__":
    main()