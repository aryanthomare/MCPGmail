# Gmail MCP tool planner

This project connects to `@gongrzhe/server-gmail-autoauth-mcp`, lists the available MCP tools, and ranks which tools are most relevant for a user request.

The Python entrypoint asks a local model to rank the discovered tools, then prints the selected tool context and generated prompt in the terminal.

## Install

```bash
npm install
```

## Run

```bash
npm run analyze -- "find unread emails from GitHub with attachments"
```

Python version:

```bash
python list_gmail_tools.py "find unread emails from GitHub with attachments"
```

Reduce prompt context to top-ranked tools:

```bash
python list_gmail_tools.py --top-tools 4 "find unread emails from GitHub with attachments"
```

Ollama via OpenAI API format:

```bash
python ollama_openai_chat.py --prompt "Summarize unread emails from GitHub"
```

## Local LLM mode

Set these environment variables to let the planner ask a local model for a tool plan:

- `LOCAL_LLM_BASE_URL` - defaults to `http://localhost:11434/v1`
- `LOCAL_LLM_MODEL` - defaults to `llama3.1`
- `LOCAL_LLM_API_KEY` - optional, only needed if your local endpoint requires one

Python script environment overrides:

- `MCP_COMMAND` - defaults to `npx`
- `MCP_ARGS` - defaults to `-y @gongrzhe/server-gmail-autoauth-mcp`

Example:

```bash
$env:LOCAL_LLM_BASE_URL = "http://localhost:11434/v1"
$env:LOCAL_LLM_MODEL = "llama3.1"
npm run analyze -- "draft an email to my manager"
```

To print a prompt you can paste into Ollama directly:

```bash
npm run analyze -- --prompt-only "draft an email to my manager"
```

That prints the generated prompt after the local model has already ranked the tools. The prompt still contains the discovered tool catalog and the request.

## Notes

- The Gmail MCP server requires Google OAuth credentials to be set up before tool calls will succeed.
- The code only plans tool usage by default. It does not execute Gmail actions unless you extend it to call `client.callTool(...)` for selected tools.

## Ollama OpenAI script

The script `ollama_openai_chat.py` uses the OpenAI Python SDK against a local Ollama server.

Defaults:

- `--base-url` -> `http://localhost:11434/v1`
- `--model` -> `llama3.1`
- `--api-key` -> `ollama` (placeholder for local usage)

You can also set environment variables:

- `OLLAMA_OPENAI_BASE_URL`
- `OLLAMA_OPENAI_MODEL`
- `OLLAMA_OPENAI_API_KEY`

Example:

```bash
python ollama_openai_chat.py --model llama3.1 --prompt "Plan Gmail MCP tools for: draft an email to my manager"
```

One command that generates the prompt from MCP tools and sends it to Ollama:

```bash
python ollama_openai_chat.py --mcp-request "draft an email to my manager"
```

Use fewer tools in planner context:

```bash
python ollama_openai_chat.py --mcp-request "draft an email to my manager" --top-tools 4
```

If you also want to inspect the generated prompt:

```bash
python ollama_openai_chat.py --mcp-request "draft an email to my manager" --show-generated-prompt
```
