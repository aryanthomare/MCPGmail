import { ChatOllama } from '@langchain/ollama';
import { MultiServerMCPClient } from '@langchain/mcp-adapters';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { HumanMessage } from '@langchain/core/messages';
const MCP_COMMAND = process.env.MCP_COMMAND ?? 'npx';
const MCP_ARGS = (process.env.MCP_ARGS ?? '-y @gongrzhe/server-gmail-autoauth-mcp')
    .split(' ')
    .filter(Boolean);
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL ?? 'llama3.1';
function parseCliArgs() {
    const rawArgs = process.argv.slice(2).filter((arg) => arg !== '--');
    const request = rawArgs.join(' ').trim();
    return { request };
}
async function main() {
    const { request } = parseCliArgs();
    if (!request) {
        console.error('Usage: npm run analyze -- "your request here"');
        process.exitCode = 1;
        return;
    }
    const mcpClient = new MultiServerMCPClient({
        gmail: {
            transport: 'stdio',
            command: MCP_COMMAND,
            args: ['-y', ...MCP_ARGS.filter((a) => a !== '-y')],
        },
    });
    try {
        const tools = await mcpClient.getTools();
        console.log('\nDiscovered MCP tools:');
        for (const tool of tools) {
            console.log(`- ${tool.name}${tool.description ? `: ${tool.description}` : ''}`);
        }
        const llm = new ChatOllama({
            model: OLLAMA_MODEL,
            baseUrl: OLLAMA_BASE_URL,
            temperature: 0,
        });
        const agent = createReactAgent({ llm, tools });
        console.log(`\nRunning agent for request: ${request}\n`);
        const response = await agent.invoke({
            messages: [new HumanMessage(request)],
        });
        for (const message of response.messages) {
            const type = message._getType();
            const content = typeof message.content === 'string'
                ? message.content
                : JSON.stringify(message.content);
            if (type === 'ai' && content && !('tool_calls' in message && message.tool_calls?.length)) {
                console.log(content);
            }
        }
    }
    finally {
        await mcpClient.close();
    }
}
void main().catch((error) => {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    console.error(message);
    process.exitCode = 1;
});
