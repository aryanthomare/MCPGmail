# Gmail MCP LangChain Agent

This project connects to `@gongrzhe/server-gmail-autoauth-mcp`, loads the available Gmail MCP tools via LangChain MCP adapters, and runs a LangGraph ReAct agent backed by a local Ollama model to answer Gmail-related requests.

Both Python and TypeScript entry points are provided.

## Install

### Python

```bash
pip install -r requirements.txt
```

### Node.js

```bash
npm install
```

## Usage

### Python — Gmail agent (LangChain + LangGraph + Ollama)

```bash
python ollama_openai_chat.py --mcp-request "find unread emails from GitHub with attachments"
```

Limit the agent to the top N tools:

```bash
python ollama_openai_chat.py --mcp-request "find unread emails from GitHub with attachments" --top-tools 3
```

Provide ranking instructions from a Markdown file:

```bash
python ollama_openai_chat.py --mcp-request "find unread emails from GitHub with attachments" --top-tools 3 --tool-ranking-prompt-file tool_ranking_prompt.md
```

Enable a bounded outer agent loop (stops early when completion checker says task is done):

```bash
python ollama_openai_chat.py --mcp-request "find unread emails from GitHub with attachments" --max-agent-loops 3
```

The Markdown prompt supports placeholders:

- `{{request}}`
- `{{tool_catalog_json}}`

Direct prompt (no MCP tools):

```bash
python ollama_openai_chat.py --prompt "Summarize unread emails from GitHub"
```

Show intermediate agent messages:

```bash
python ollama_openai_chat.py --mcp-request "draft an email to my manager" --show-llm-io
```

Enable debug log:

```bash
python ollama_openai_chat.py --mcp-request "draft an email to my manager" --debug
```

### Python — List and rank tools

```bash
python list_gmail_tools.py "find unread emails from GitHub with attachments"
```

Limit prompt context to top-ranked tools:

```bash
python list_gmail_tools.py --top-tools 4 "find unread emails from GitHub with attachments"
```

If you omit `--top-tools`, all ranked tools are included in the prompt context.

### Node.js / TypeScript — Gmail agent

```bash
npm run analyze -- "find unread emails from GitHub with attachments"
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama base URL |
| `MCP_COMMAND` | `npx` | Command to start the MCP server |
| `MCP_ARGS` | `-y @gongrzhe/server-gmail-autoauth-mcp` | Arguments for the MCP server command |

Example:

```bash
export OLLAMA_MODEL=gemma4
export OLLAMA_BASE_URL=http://localhost:11434
python ollama_openai_chat.py --mcp-request "draft an email to my manager"
```

## Architecture

- **LangChain MCP adapters** (`langchain-mcp-adapters` / `@langchain/mcp-adapters`) connect to the Gmail MCP server and expose its tools as LangChain `BaseTool` instances.
- **ChatOllama** (`langchain-ollama` / `@langchain/ollama`) provides the local Ollama model as a LangChain chat model.
- **LangGraph ReAct agent** (`langgraph` / `@langchain/langgraph`) runs the tool-use loop: the model decides which tools to call, the tools are executed, and results are fed back until the request is fulfilled.

## Notes

- The Gmail MCP server requires Google OAuth credentials to be set up before tool calls will succeed.
- Run `npx @gongrzhe/server-gmail-autoauth-mcp auth` once to complete the browser OAuth flow and store credentials.
- Use `python list_gmail_tools.py --auth` to run authentication via the Python script.

Example: 
python ollama_openai_chat.py --mcp-request "read all emails from 4/22/2026" --debug --show-llm-io --top-tools 4 --tool-ranking-prompt-file tool_ranking_prompt.md --model gemma4