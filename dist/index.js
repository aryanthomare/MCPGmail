import { Client } from '@modelcontextprotocol/sdk/client';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio';
const MCP_COMMAND = process.env.MCP_COMMAND ?? 'npx';
const MCP_ARGS = (process.env.MCP_ARGS ?? '@gongrzhe/server-gmail-autoauth-mcp')
    .split(' ')
    .filter(Boolean);
const LOCAL_LLM_BASE_URL = process.env.LOCAL_LLM_BASE_URL ?? 'http://localhost:11434/v1';
const LOCAL_LLM_MODEL = process.env.LOCAL_LLM_MODEL ?? 'llama3.1';
const LOCAL_LLM_API_KEY = process.env.LOCAL_LLM_API_KEY ?? '';
function tokenize(text) {
    return text
        .toLowerCase()
        .replace(/[^a-z0-9_\s-]+/g, ' ')
        .split(/\s+/)
        .map((token) => token.trim())
        .filter(Boolean);
}
function schemaText(schema) {
    if (!schema) {
        return '';
    }
    return JSON.stringify(schema).toLowerCase();
}
function rankTools(tools, prompt) {
    const promptTokens = new Set(tokenize(prompt));
    return tools
        .map((tool) => {
        const haystack = tokenize([tool.name, tool.description ?? '', schemaText(tool.inputSchema)].join(' '));
        let score = 0;
        const matches = [];
        for (const token of promptTokens) {
            if (haystack.includes(token)) {
                score += token.length >= 4 ? 2 : 1;
                matches.push(token);
            }
        }
        const emailFocusedWords = ['email', 'mail', 'inbox', 'label', 'attachment', 'draft'];
        for (const keyword of emailFocusedWords) {
            if (promptTokens.has(keyword) && haystack.includes(keyword)) {
                score += 2;
            }
        }
        const reason = matches.length
            ? `Matched on: ${Array.from(new Set(matches)).join(', ')}`
            : 'No direct lexical match; kept as a fallback tool.';
        return { ...tool, score, reason };
    })
        .sort((left, right) => right.score - left.score || left.name.localeCompare(right.name));
}
function buildPlanPrompt(request, tools) {
    const toolCatalog = tools.map((tool) => ({
        name: tool.name,
        description: tool.description ?? '',
        inputSchema: tool.inputSchema ?? {},
    }));
    return [
        'You are a tool-planning assistant for a Gmail MCP client.',
        'Select the smallest set of tools needed to satisfy the user request.',
        'Return JSON only. No markdown, no explanation, no backticks.',
        'Use this exact shape:',
        '{"chosenTools":[{"name":"tool_name","reason":"why","arguments":{}}]}',
        'Rules:',
        '- Only use tools from the provided catalog.',
        '- If no tool is needed, return an empty chosenTools array.',
        '- If you choose a tool, include only arguments that are supported by its schema.',
        'Available tools:',
        JSON.stringify(toolCatalog, null, 2),
        'User request:',
        request,
    ].join('\n\n');
}
function parseCliArgs() {
    const rawArgs = process.argv.slice(2);
    const promptOnly = rawArgs.includes('--prompt-only');
    const request = rawArgs.filter((arg) => arg !== '--prompt-only').join(' ').trim();
    return { request, promptOnly };
}
async function analyzeWithLocalLlm(request, tools) {
    const body = {
        model: LOCAL_LLM_MODEL,
        messages: [
            {
                role: 'system',
                content: 'You are a strict JSON tool planner. Never invent tools. Use only the provided tool catalog.',
            },
            {
                role: 'user',
                content: buildPlanPrompt(request, tools),
            },
        ],
        temperature: 0,
    };
    const response = await fetch(`${LOCAL_LLM_BASE_URL.replace(/\/$/, '')}/chat/completions`, {
        method: 'POST',
        headers: {
            'content-type': 'application/json',
            ...(LOCAL_LLM_API_KEY ? { authorization: `Bearer ${LOCAL_LLM_API_KEY}` } : {}),
        },
        body: JSON.stringify(body),
    });
    if (!response.ok) {
        throw new Error(`Local LLM request failed: ${response.status} ${response.statusText}`);
    }
    const payload = (await response.json());
    const content = payload.choices?.[0]?.message?.content?.trim();
    if (!content) {
        return null;
    }
    const jsonStart = content.indexOf('{');
    const jsonEnd = content.lastIndexOf('}');
    if (jsonStart === -1 || jsonEnd === -1 || jsonEnd <= jsonStart) {
        return null;
    }
    const parsed = JSON.parse(content.slice(jsonStart, jsonEnd + 1));
    if (!parsed || !Array.isArray(parsed.chosenTools)) {
        return null;
    }
    return parsed;
}
async function connectToGmailMcpServer() {
    const transport = new StdioClientTransport({
        command: MCP_COMMAND,
        args: ['-y', ...MCP_ARGS],
    });
    const client = new Client({ name: 'gmail-tool-planner', version: '1.0.0' }, {
        capabilities: {},
    });
    await client.connect(transport);
    return client;
}
async function main() {
    const { request, promptOnly } = parseCliArgs();
    if (!request) {
        console.error('Usage: npm run analyze -- [--prompt-only] "your request here"');
        process.exitCode = 1;
        return;
    }
    const client = await connectToGmailMcpServer();
    try {
        const { tools } = await client.listTools();
        const rankedTools = rankTools(tools, request);
        const prompt = buildPlanPrompt(request, tools);
        console.log('\nDiscovered MCP tools:');
        for (const tool of tools) {
            console.log(`- ${tool.name}${tool.description ? `: ${tool.description}` : ''}`);
        }
        console.log('\nHeuristic analysis:');
        for (const tool of rankedTools.slice(0, 5)) {
            console.log(`- ${tool.name} [score ${tool.score}] ${tool.reason}`);
        }
        if (promptOnly) {
            console.log('\nOllama prompt:');
            console.log(prompt);
            return;
        }
        try {
            const plan = await analyzeWithLocalLlm(request, tools);
            if (plan) {
                console.log('\nLocal LLM plan:');
                for (const item of plan.chosenTools) {
                    console.log(`- ${item.name}: ${item.reason}`);
                    if (item.arguments && Object.keys(item.arguments).length > 0) {
                        console.log(`  arguments: ${JSON.stringify(item.arguments)}`);
                    }
                }
            }
        }
        catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            console.log(`\nLocal LLM planning skipped: ${message}`);
            console.log('\nOllama prompt:');
            console.log(prompt);
        }
    }
    finally {
        await client.close();
    }
}
void main().catch((error) => {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    console.error(message);
    process.exitCode = 1;
});
