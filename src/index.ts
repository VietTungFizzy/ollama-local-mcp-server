import express, { Request, Response } from "express";
import { randomUUID } from "node:crypto";
import { Ollama } from "ollama";
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

type OllamaMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

async function callOllamaChat(options: {
  model: string;
  messages: OllamaMessage[];
}): Promise<string> {
  const response = await ollamaClient.chat({
    model: options.model,
    messages: options.messages,
    stream: false,
  });

  return response.message.content;
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

      console.log(`[debug] ask_ollama called`);
      console.log(`[debug]   host      : ${OLLAMA_BASE_URL}`);
      console.log(`[debug]   model     : ${selectedModel}`);
      console.log(`[debug]   text      : ${JSON.stringify(text.slice(0, 300))}`);
      console.log(`[debug]   system    : ${JSON.stringify(system?.slice(0, 100) ?? null)}`);
      console.log(`[debug]   cloud mode: ${!!OLLAMA_API_KEY}`);

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
        const messages: OllamaMessage[] = [];

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
        });

        console.log(`[debug] Ollama reply (${reply.length} chars): ${JSON.stringify(reply.slice(0, 300))}${reply.length > 300 ? "…" : ""}`);

        return {
          content: [
            {
              type: "text",
              text: reply,
            },
          ],
        };
      } catch (error) {
        console.error("[debug] Error calling Ollama:");
        console.error(`[debug]   host : ${OLLAMA_BASE_URL}`);
        console.error(`[debug]   model: ${selectedModel}`);
        if (error instanceof Error) {
          console.error(`[debug]   name : ${error.name}`);
          console.error(`[debug]   msg  : ${error.message}`);
          if ("cause" in error) console.error(`[debug]   cause: ${String((error as NodeJS.ErrnoException).cause)}`);
        } else {
          console.error(`[debug]   raw  : ${String(error)}`);
        }

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
