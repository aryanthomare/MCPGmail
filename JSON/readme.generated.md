# Gmail MCP Tool Manager

A multi‑language toolset that powers a Gmail‑MCP server, provides a Python helper for listing and ranking Gmail tools, and offers TypeScript modules for prompt generation. The repository also ships with automated tests for changelog extraction and command execution.

---

## 1. Project Overview

Gmail MCP Tool Manager enables developers to interact with Gmail tools through a unified interface. It exposes a Node.js/TypeScript HTTP server that talks to the Gmail API, a Python CLI helper that discovers and prioritizes available tools, and a set of TypeScript utilities for crafting prompts and commands. Automated tests validate changelog parsing and command execution logic, ensuring reliability across updates.

---

## 2. Features

- **Gmail‑MCP server** – RESTful API (Node.js/TypeScript) that authenticates with Gmail, fetches messages, and executes tool‑specific commands.
- **Python helper** – CLI for listing Gmail tools, ranking them by relevance, and exporting tool metadata in JSON.
- **Prompt building utilities** – TypeScript modules that help construct well‑formed prompts for Gmail MCP commands.
- **Automated testing** – Jest (for TypeScript) and PyTest (for Python) test suites for changelog extraction, command execution, and API endpoints.
- **Strict build configuration** – `tsconfig.json` targets ES2022 with strict type checking to catch bugs early.

---

## 3. Tech Stack

| Layer | Language / Runtime | Key Libraries |
|-------|--------------------|----------------|
| Server | TypeScript (Node.js) | Express, googleapis, jest |
| CLI Helper | Python | google-api-python-client, requests, pytest |
| Prompt Utilities | TypeScript | ts-node, lodash |
| Build / Testing | Node.js, Python | npm scripts, ts-node, Jest, PyTest |

---

## 4. Project Structure

```
Gmail-MCP-Tool-Manager/
├── server/                    # Node/TS server source
│   ├── src/
│   │   ├── index.ts          # Application entry point
│   │   ├── routes.ts         # Express routes
│   │   └── gmailClient.ts    # Gmail API wrapper
│   └── tsconfig.json
├── cli/                       # Python helper
│   ├── tool_manager.py        # CLI logic
│   └── requirements.txt
├── prompts/                   # Prompt building modules
│   ├── promptBuilder.ts
│   └── types.ts
├── tests/
│   ├── server.test.ts
│   └── cli.test.py
├── package.json
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## 5. Setup

### Prerequisites

- **Node.js** v20+  
- **Python** 3.10+  
- **Google Cloud Project** with Gmail API enabled  
  - Create OAuth2 credentials (client ID & secret)  
  - Set `redirect_uri` to `http://localhost:3000/oauth2callback`

### Clone the repo

```bash
git clone https://github.com/your-org/gmail-mcp-tool-manager.git
cd gmail-mcp-tool-manager
```

### Install Node dependencies

```bash
npm ci
```

### Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r cli/requirements.txt
```

---

## 6. Usage

### 6.1 Start the Gmail MCP Server

```bash
npm run start
# Server listens on http://localhost:3000
```

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/tools` | List available Gmail tools |
| POST | `/api/run` | Execute a specific tool command |

Example request:

```bash
curl -X POST http://localhost:3000/api/run \
  -H "Content-Type: application/json" \
  -d '{"tool":"compose","params":{"to":"alice@example.com","subject":"Hello"}}'
```

### 6.2 Python CLI Helper

```bash
python cli/tool_manager.py list
```

Ranking:

```bash
python cli/tool_manager.py rank
```

Export JSON metadata:

```bash
python cli/tool_manager.py export --output tools.json
```

### 6.3 Prompt Utilities

```ts
import { buildComposePrompt } from '../prompts/promptBuilder';

const prompt = buildComposePrompt({
  to: 'bob@example.com',
  subject: 'Report',
  body: 'Please see attached.'
});
console.log(prompt);
```

---

## 7. Notes

- **Authentication** – The first time you start the server, it will prompt for a Gmail OAuth2 token. Copy the code from the browser to the terminal.  
- **Testing** – Run the full test suite with:

  ```bash
  npm test          # TypeScript tests
  pytest tests/     # Python tests
  ```

- **Deployment** – For production, set `NODE_ENV=production` and consider Dockerizing the Node server.  
- **Extending Tools** – Add a new tool by creating a new route in `server/src/routes.ts` and implementing its logic in `gmailClient.ts`.  
- **Type Safety** – All TypeScript code compiles with `tsc --noEmit`. Ensure the compiler passes before committing changes.

Feel free to open issues or submit pull requests to improve functionality or add new tools.
