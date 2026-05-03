import express, { Request, Response } from "express";
import { randomUUID } from "node:crypto";
import { Ollama, type Message, type Tool, type WebSearchResult } from "ollama";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { isInitializeRequest } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";

const PORT = Number(process.env.PORT ?? 3000);
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL ?? "http://ollama:11434";
const DEFAULT_OLLAMA_MODEL = process.env.OLLAMA_MODEL ?? "llama3.2";
const OLLAMA_API_KEY = process.env.OLLAMA_API_KEY ?? "";

const ollamaClient = new Ollama({
  host: OLLAMA_BASE_URL,
  ...(OLLAMA_API_KEY
    ? { headers: { Authorization: `Bearer ${OLLAMA_API_KEY}` } }
    : {}),
});

const OLLAMA_TOOLS: Tool[] = [
  {
    type: "function",
    function: {
      name: "list_ollama_models",
      description: "List models available in the Ollama container",
      parameters: { type: "object", properties: {}, required: [] },
    },
  },
  {
    type: "function",
    function: {
      name: "web_search",
      description: "Search the web for up-to-date information",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "The search query" },
          max_results: { type: "number", description: "Maximum results to return (default 5, max 10)" },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "web_fetch",
      description: "Fetch the content of a web page by URL",
      parameters: {
        type: "object",
        properties: {
          url: { type: "string", description: "The URL to fetch" },
        },
        required: ["url"],
      },
    },
  },
];

async function executeLocalTool(
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  switch (name) {
    case "list_ollama_models": {
      const models = await listOllamaModels();
      return models.length
        ? `Available Ollama models:\n${models.map((m) => `- ${m}`).join("\n")}`
        : "No Ollama models found. Run `ollama pull llama3.2` in the Ollama container.";
    }
    case "web_search": {
      const query = String(args.query ?? "");
      const maxResults = args.max_results ? Number(args.max_results) : undefined;
      const searchResult = await ollamaClient.webSearch({ query, ...(maxResults ? { maxResults } : {}) });
      if (!searchResult.results?.length) {
        return "No results found.";
      }
      return searchResult.results
        .map((r: WebSearchResult, i: number) => `[${i + 1}] ${r.content}`)
        .join("\n\n");
    }
    case "web_fetch": {
      const url = String(args.url ?? "");
      const fetchResult = await ollamaClient.webFetch({ url });
      return `**${fetchResult.title}**\n${fetchResult.url}\n\n${fetchResult.content}`;
    }
    default:
      return `Tool "${name}" is not directly executable by this server.`;
  }
}

const MAX_TOOL_STEPS = 10;

async function callOllamaChat(options: {
  model: string;
  messages: Message[];
  tools?: Tool[];
}): Promise<string> {
  const messages: Message[] = [...options.messages];

  for (let step = 0; step < MAX_TOOL_STEPS; step++) {
    const response = await ollamaClient.chat({
      model: options.model,
      messages,
      tools: options.tools,
      stream: false,
    });

    const msg = response.message;

    if (!msg.tool_calls || msg.tool_calls.length === 0) {
      return msg.content;
    }

    messages.push({ role: "assistant", content: msg.content ?? "", tool_calls: msg.tool_calls });

    for (const toolCall of msg.tool_calls) {
      const toolName = toolCall.function.name;
      const toolArgs = (toolCall.function.arguments ?? {}) as Record<string, unknown>;
      let toolResult: string;

      try {
        toolResult = await executeLocalTool(toolName, toolArgs);
      } catch (err) {
        toolResult = `Error executing tool "${toolName}": ${
          err instanceof Error ? err.message : String(err)
        }`;
      }

      messages.push({ role: "tool", content: toolResult });
    }
  }

  return "Reached maximum tool call steps without a final response.";
}

async function listOllamaModels(): Promise<string[]> {
  const response = await ollamaClient.list();
  return response.models.map((m) => m.name);
}

function createMcpServer(): McpServer {
  const server = new McpServer({
    name: "ollama-http",
    version: "1.0.0",
  });

  server.registerTool(
    "ask_ollama",
    {
      description: "Ask an Ollama model running in another Docker container",
      inputSchema: {
        query: z.string().optional().describe("The prompt to send to Ollama"),
        prompt: z.string().optional().describe("Alias for query"),
        model: z
          .string()
          .optional()
          .describe(
            `The Ollama model to use. Defaults to ${DEFAULT_OLLAMA_MODEL}`,
          ),
        system: z
          .string()
          .optional()
          .describe("Optional system instruction for the model"),
      },
    },
    async ({ query, prompt, model, system }) => {
      const text = query ?? prompt ?? "";
      const selectedModel = model ?? DEFAULT_OLLAMA_MODEL;

      // Heartbeat: return alive response without calling Ollama
      if (!text.trim()) {
        return {
          content: [
            {
              type: "text",
              text: "Ollama MCP server is alive and ready.",
            },
          ],
        };
      }

      try {
        const messages: Message[] = [];

        if (system) {
          messages.push({
            role: "system",
            content: system,
          });
        }

        messages.push({
          role: "user",
          content: text,
        });

        const reply = await callOllamaChat({
          model: selectedModel,
          messages,
          tools: OLLAMA_TOOLS,
        });

        return {
          content: [
            {
              type: "text",
              text: reply,
            },
          ],
        };
      } catch (error) {
        console.error("Error calling Ollama:", error);

        return {
          content: [
            {
              type: "text",
              text: `Failed to call Ollama model "${selectedModel}" at ${OLLAMA_BASE_URL}: ${
                error instanceof Error ? error.message : String(error)
              }`,
            },
          ],
          isError: true,
        };
      }
    },
  );

  server.registerTool(
    "list_ollama_models",
    {
      description: "List models available in the Ollama container",
      inputSchema: {},
    },
    async () => {
      try {
        const models = await listOllamaModels();

        if (!models.length) {
          return {
            content: [
              {
                type: "text",
                text: "No Ollama models found. Run `ollama pull llama3.2` in the Ollama container.",
              },
            ],
          };
        }

        return {
          content: [
            {
              type: "text",
              text: `Available Ollama models:\n\n${models
                .map((name) => `- ${name}`)
                .join("\n")}`,
            },
          ],
        };
      } catch (error) {
        console.error("Error listing Ollama models:", error);

        return {
          content: [
            {
              type: "text",
              text: `Failed to list Ollama models at ${OLLAMA_BASE_URL}: ${
                error instanceof Error ? error.message : String(error)
              }`,
            },
          ],
          isError: true,
        };
      }
    },
  );

  server.registerTool(
    "web_search",
    {
      description: "Search the web for up-to-date information",
      inputSchema: {
        query: z.string().describe("The search query"),
        max_results: z
          .number()
          .optional()
          .describe("Maximum results to return (default 5, max 10)"),
      },
    },
    async ({ query, max_results }) => {
      try {
        const searchResult = await ollamaClient.webSearch({
          query,
          ...(max_results ? { maxResults: max_results } : {}),
        });

        if (!searchResult.results?.length) {
          return {
            content: [{ type: "text", text: "No results found." }],
          };
        }

        return {
          content: [
            {
              type: "text",
              text: searchResult.results
                .map((r: WebSearchResult, i: number) => `[${i + 1}] ${r.content}`)
                .join("\n\n"),
            },
          ],
        };
      } catch (error) {
        console.error("Error calling web_search:", error);
        return {
          content: [
            {
              type: "text",
              text: `Web search failed: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
          isError: true,
        };
      }
    },
  );

  server.registerTool(
    "web_fetch",
    {
      description: "Fetch the content of a web page by URL",
      inputSchema: {
        url: z.string().describe("The URL to fetch"),
      },
    },
    async ({ url }) => {
      try {
        const fetchResult = await ollamaClient.webFetch({ url });

        return {
          content: [
            {
              type: "text",
              text: `**${fetchResult.title}**\n${fetchResult.url}\n\n${fetchResult.content}`,
            },
          ],
        };
      } catch (error) {
        console.error("Error calling web_fetch:", error);
        return {
          content: [
            {
              type: "text",
              text: `Web fetch failed: ${error instanceof Error ? error.message : String(error)}`,
            },
          ],
          isError: true,
        };
      }
    },
  );

  return server;
}

const app = express();
app.use(express.json());

const transports: Record<string, StreamableHTTPServerTransport> = {};

app.post("/mcp", async (req: Request, res: Response) => {
  const sessionId = req.headers["mcp-session-id"] as string | undefined;

  let transport: StreamableHTTPServerTransport;

  if (sessionId && transports[sessionId]) {
    transport = transports[sessionId];
  } else if (!sessionId && isInitializeRequest(req.body)) {
    transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: () => randomUUID(),
      onsessioninitialized: (newSessionId) => {
        transports[newSessionId] = transport;
      },
    });

    transport.onclose = () => {
      if (transport.sessionId) {
        delete transports[transport.sessionId];
      }
    };

    const server = createMcpServer();
    await server.connect(transport);
  } else {
    res.status(400).json({
      jsonrpc: "2.0",
      error: {
        code: -32000,
        message: "Bad Request: No valid session ID provided",
      },
      id: null,
    });
    return;
  }

  await transport.handleRequest(req, res, req.body);
});

app.get("/mcp", async (req: Request, res: Response) => {
  const sessionId = req.headers["mcp-session-id"] as string | undefined;

  if (!sessionId || !transports[sessionId]) {
    res.status(400).send("Invalid or missing MCP session ID");
    return;
  }

  const transport = transports[sessionId];
  await transport.handleRequest(req, res);
});

app.delete("/mcp", async (req: Request, res: Response) => {
  const sessionId = req.headers["mcp-session-id"] as string | undefined;

  if (!sessionId || !transports[sessionId]) {
    res.status(400).send("Invalid or missing MCP session ID");
    return;
  }

  const transport = transports[sessionId];
  await transport.handleRequest(req, res);
});

app.get("/health", (_req: Request, res: Response) => {
  res.json({
    ok: true,
    ollamaBaseUrl: OLLAMA_BASE_URL,
    defaultModel: DEFAULT_OLLAMA_MODEL,
    cloudMode: !!OLLAMA_API_KEY,
  });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Ollama HTTP MCP server listening on port ${PORT}`);
  console.log(`Ollama base URL: ${OLLAMA_BASE_URL}`);
  console.log(`Default Ollama model: ${DEFAULT_OLLAMA_MODEL}`);
  console.log(`Cloud mode: ${OLLAMA_API_KEY ? "enabled (API key set)" : "disabled (local)"}`);
});
