/**
 * RLM-based session compaction.
 *
 * Implements the Recursive Language Models paradigm (Zhang, Kraska, Khattab —
 * arXiv 2512.24601) for session compaction, following the architectural
 * patterns from deepfates/cantrip:
 *
 *   1. Strict context isolation — REPL output is metadata-only in the driving
 *      model's conversation history. Only explicit print() output is shown.
 *   2. Forced sandbox — the model MUST use tool_use (python_repl / submit_answer).
 *      It cannot answer without going through the REPL.
 *   3. stdin-based IPC — llm_query() uses Python's input() which blocks while
 *      the TypeScript host resolves the LLM call asynchronously via the Jupyter
 *      stdin channel. No file polling.
 *   4. Recursive sub-agents — llm_query() spawns a nested RLM agent at depth+1
 *      with its own kernel, context, and tools. Falls back to plain LLM at
 *      max depth.
 *   5. Resource cleanup — try/finally for kernel disposal, token tracking across
 *      all recursion levels.
 *
 * Architecture:
 *   - Conversation text → IPython kernel `context` variable (programmatic space)
 *   - Model calls python_repl tool → kernel executes → metadata-only result
 *   - llm_query() in kernel → input() blocks → host resolves via inputProvider
 *   - Model calls submit_answer tool → compaction complete
 */
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { AssistantMessage, Message, Model, Tool } from "@oh-my-pi/pi-ai";
import { completeSimple } from "@oh-my-pi/pi-ai";
import { Type, type Static } from "@sinclair/typebox";
import { logger } from "@oh-my-pi/pi-utils";
import { renderPromptTemplate } from "../../config/prompt-templates";
import { executePython, disposeKernelSessionById } from "../../ipy/executor";
import { convertToLlm } from "../../session/messages";
import { serializeConversation } from "./utils";
import rlmCompactionSystemPrompt from "../../prompts/compaction/rlm-compaction-system.md" with { type: "text" };

// ============================================================================
// Types
// ============================================================================

export interface RlmCompactionOptions {
	/** Model that drives the RLM loop (writes REPL code, calls submit_answer) */
	model: Model;
	/** API key for the driving model */
	apiKey: string;
	/** Optional cheaper model for llm_query() sub-calls from the REPL */
	leafModel?: Model;
	/** API key for the leaf model */
	leafApiKey?: string;
	/** Maximum iterations of the tool-use loop */
	maxIterations: number;
	/** Abort signal */
	signal?: AbortSignal;
	/** Previous compaction summary (for iterative updates) */
	previousSummary?: string;
	/** Custom instructions from the user (e.g., /compact focus on auth) */
	customInstructions?: string;
	/** Working directory for the Python kernel */
	cwd?: string;
	/** Token budget for driving model responses */
	reserveTokens: number;
	/** Current recursion depth (default: 0) */
	depth?: number;
	/** Maximum recursion depth (default: 2) */
	maxDepth?: number;
}

/** Accumulated token usage across the driving model and all sub-LLM calls. */
export interface RlmUsage {
	drivingInputTokens: number;
	drivingOutputTokens: number;
	subLlmInputTokens: number;
	subLlmOutputTokens: number;
	totalCost: number;
}

export interface RlmCompactionResult {
	/** The final compaction summary produced by the model */
	summary: string;
	/** Number of iterations the RLM loop ran */
	iterations: number;
	/** Number of llm_query() sub-calls made from the REPL */
	subLlmCalls: number;
	/** Number of code blocks executed */
	codeBlocksExecuted: number;
	/** Accumulated token usage */
	usage: RlmUsage;
}

// ============================================================================
// Constants
// ============================================================================

/** Maximum characters of print() stdout to feed back to the model per tool call */
const MAX_PRINT_OUTPUT_CHARS = 2_000;

/** Timeout for each Python cell execution (ms). Generous to allow for llm_query latency. */
const CELL_TIMEOUT_MS = 120_000;

// ============================================================================
// Tool Schemas
// ============================================================================

const pythonReplSchema = Type.Object({
	code: Type.String({ description: "Python code to execute in the REPL environment" }),
});

const submitAnswerSchema = Type.Object({
	summary: Type.String({ description: "The final compaction summary text" }),
	variable: Type.Optional(Type.String({
		description: "Name of a Python variable containing the summary (alternative to summary field)",
	})),
});

type PythonReplParams = Static<typeof pythonReplSchema>;
type SubmitAnswerParams = Static<typeof submitAnswerSchema>;

const RLM_TOOLS: Tool[] = [
	{
		name: "python_repl",
		description:
			"Execute Python code in the REPL environment. " +
			"The REPL has a pre-loaded `context` variable (string) containing the full conversation. " +
			"Use print() to see specific data — only printed output is shown. " +
			"Use llm_query(prompt) to delegate semantic analysis to a sub-LLM.",
		parameters: pythonReplSchema,
	},
	{
		name: "submit_answer",
		description:
			"Submit the final compaction summary. You MUST explore the context via python_repl " +
			"before calling this. Provide the summary text directly, or set `variable` to the name " +
			"of a Python variable you built up across REPL calls.",
		parameters: submitAnswerSchema,
	},
];

// ============================================================================
// System Prompt
// ============================================================================

const RLM_SYSTEM_PROMPT = renderPromptTemplate(rlmCompactionSystemPrompt);

// ============================================================================
// Metadata Formatting
// ============================================================================

/**
 * Format REPL output as metadata-only (like cantrip's formatRlmMetadata).
 * Keeps data out of the driving model's token window.
 */
export function formatReplMetadata(output: string): string {
	const trimmed = output.trim();
	if (!trimmed) return "[no output]";
	const lines = trimmed.split("\n");
	const preview = trimmed.slice(0, 200).replace(/\n/g, "\\n");
	return `[${trimmed.length} chars, ${lines.length} lines] "${preview}${trimmed.length > 200 ? "..." : ""}"`;
}

/**
 * Extract text from an AssistantMessage's content blocks.
 */
function extractText(response: AssistantMessage): string {
	return response.content
		.filter((c): c is { type: "text"; text: string } => c.type === "text")
		.map(c => c.text)
		.join("\n");
}

/**
 * Accumulate usage from a completeSimple response.
 */
function accumulateUsage(usage: RlmUsage, response: AssistantMessage, target: "driving" | "sub"): void {
	const u = response.usage;
	if (!u) return;
	const input = u.input + u.cacheRead + u.cacheWrite;
	const output = u.output;
	const cost = u.cost?.total ?? 0;
	if (target === "driving") {
		usage.drivingInputTokens += input;
		usage.drivingOutputTokens += output;
	} else {
		usage.subLlmInputTokens += input;
		usage.subLlmOutputTokens += output;
	}
	usage.totalCost += cost;
}

// ============================================================================
// REPL Setup
// ============================================================================

/**
 * Build Python setup code that:
 * 1. Loads the serialized conversation into a `context` variable
 * 2. Defines llm_query() using input() for stdin-based IPC — Python blocks on
 *    input() while the TypeScript host resolves the LLM call via inputProvider.
 */
export function buildSetupCode(contextText: string): string {
	// Escape for Python triple-quoted string
	const escaped = contextText
		.replace(/\\/g, "\\\\")
		.replace(/"""/g, '\\"\\"\\"');

	return `
import json

# Load conversation context
context = """${escaped}"""

# Metadata
context_length = len(context)
context_lines = context.count("\\n") + 1

def llm_query(prompt, sub_context=None, model=None):
    """Query a sub-LLM. The prompt can reference the conversation context.
    If sub_context is provided, the child agent explores that instead.
    Returns the model's text response as a string.
    Use this to analyze chunks of context that are too large to print."""
    request = json.dumps({
        "type": "llm_query",
        "prompt": str(prompt),
        "context": sub_context,
        "model": model,
    })
    response_json = input(request)
    response = json.loads(response_json)
    if "error" in response:
        raise RuntimeError(f"llm_query failed: {response['error']}")
    return response["response"]

def llm_query_batched(prompts, sub_contexts=None, model=None):
    """Query a sub-LLM with multiple prompts. Returns a list of responses.
    Optionally pass sub_contexts (list of strings) to give each child its own data."""
    results = []
    for i, p in enumerate(prompts):
        ctx = sub_contexts[i] if sub_contexts and i < len(sub_contexts) else None
        results.append(llm_query(p, sub_context=ctx, model=model))
    return results

def SHOW_VARS():
    """Print all user-defined variables and their types/sizes."""
    for name, val in sorted(globals().items()):
        if name.startswith("_") or callable(val) or name in ("json", "In", "Out"):
            continue
        t = type(val).__name__
        if isinstance(val, str):
            print(f"  {name}: str ({len(val)} chars)")
        elif isinstance(val, (list, tuple, dict, set)):
            print(f"  {name}: {t} ({len(val)} items)")
        else:
            print(f"  {name}: {t}")

print(f"Context loaded: {context_length} characters, {context_lines} lines")
`.trim();
}

// ============================================================================
// Main RLM Loop
// ============================================================================

/**
 * Run RLM-based compaction on a set of messages.
 *
 * The model uses two tools:
 *   - `python_repl`: Execute Python code in a REPL with `context` variable
 *   - `submit_answer`: Provide the final compaction summary
 *
 * The REPL provides `llm_query(prompt, sub_context?)` which spawns recursive
 * child RLM agents (depth-limited, falling back to plain LLM at max depth).
 */
export async function rlmCompact(
	messagesToSummarize: AgentMessage[],
	recentMessages: AgentMessage[],
	options: RlmCompactionOptions,
): Promise<RlmCompactionResult> {
	const {
		model,
		apiKey,
		leafModel,
		leafApiKey,
		maxIterations,
		signal,
		previousSummary,
		customInstructions,
		cwd,
		reserveTokens,
	} = options;
	const currentDepth = options.depth ?? 0;
	const maxRecursionDepth = options.maxDepth ?? 2;

	// Serialize conversation
	const llmMessages = convertToLlm([...messagesToSummarize]);
	const conversationText = serializeConversation(llmMessages);

	// Also serialize recent messages for task context
	const recentLlmMessages = convertToLlm(recentMessages);
	const recentText = serializeConversation(recentLlmMessages);

	// Kernel session for this agent level
	const sessionId = `rlm-${Date.now()}-${Math.random().toString(36).slice(2)}`;
	const kernelSessionId = `rlm-compaction-d${currentDepth}-${sessionId}`;

	// Usage tracker
	const usage: RlmUsage = {
		drivingInputTokens: 0,
		drivingOutputTokens: 0,
		subLlmInputTokens: 0,
		subLlmOutputTokens: 0,
		totalCost: 0,
	};

	let iterations = 0;
	let totalSubCalls = 0;
	let totalCodeBlocks = 0;

	try {
		// Set up the REPL environment
		const setupCode = buildSetupCode(conversationText);

		logger.debug("RLM compaction: setting up REPL", {
			depth: currentDepth,
			contextChars: conversationText.length,
			contextLines: conversationText.split("\n").length,
		});

		const setupResult = await executePython(setupCode, {
			cwd,
			timeoutMs: CELL_TIMEOUT_MS,
			signal,
			sessionId: kernelSessionId,
			kernelMode: "session",
		});

		if (setupResult.exitCode !== 0) {
			throw new Error(`RLM REPL setup failed: ${setupResult.output}`);
		}

		// Sub-LLM model resolution
		const subQueryModel = leafModel || model;
		const subQueryApiKey = leafApiKey || apiKey;

		// inputProvider: handles llm_query() calls from Python's input()
		const inputProvider = async (prompt: string): Promise<string> => {
			let request: { type: string; prompt: string; context?: string; model?: string };
			try {
				request = JSON.parse(prompt);
			} catch {
				return JSON.stringify({ error: "Invalid request JSON" });
			}

			if (request.type !== "llm_query") {
				return JSON.stringify({ error: `Unknown request type: ${request.type}` });
			}

			totalSubCalls++;

			if (currentDepth >= maxRecursionDepth) {
				// At max depth — fall back to plain LLM call with truncated context
				logger.debug("RLM llm_query at max depth, using plain LLM", {
					depth: currentDepth,
					promptLength: request.prompt.length,
				});
				const truncatedPrompt = request.prompt.length > 10_000
					? request.prompt.slice(0, 10_000) + "\n... [truncated]"
					: request.prompt;
				try {
					const response = await completeSimple(
						subQueryModel,
						{
							messages: [{
								role: "user",
								content: [{ type: "text", text: truncatedPrompt }],
								timestamp: Date.now(),
							}],
						},
						{ apiKey: subQueryApiKey, signal, maxTokens: 8192 },
					);
					accumulateUsage(usage, response, "sub");
					return JSON.stringify({ response: extractText(response) });
				} catch (error) {
					return JSON.stringify({ error: error instanceof Error ? error.message : String(error) });
				}
			}

			// Recursive: spawn a nested RLM agent
			logger.debug("RLM llm_query spawning child agent", {
				depth: currentDepth + 1,
				promptLength: request.prompt.length,
				hasSubContext: !!request.context,
			});

			try {
				// Build sub-messages from the provided context (or parent context)
				const subContextText = request.context ?? conversationText;
				const subMessages: AgentMessage[] = [{
					role: "user",
					content: [{ type: "text", text: subContextText }],
					timestamp: Date.now(),
				}];

				const childResult = await rlmCompact(subMessages, [], {
					model: subQueryModel,
					apiKey: subQueryApiKey,
					leafModel,
					leafApiKey,
					maxIterations: Math.min(maxIterations, 8),
					signal,
					customInstructions: request.prompt,
					cwd,
					reserveTokens,
					depth: currentDepth + 1,
					maxDepth: maxRecursionDepth,
				});

				// Aggregate child usage
				totalSubCalls += childResult.subLlmCalls;
				totalCodeBlocks += childResult.codeBlocksExecuted;
				usage.subLlmInputTokens += childResult.usage.drivingInputTokens + childResult.usage.subLlmInputTokens;
				usage.subLlmOutputTokens += childResult.usage.drivingOutputTokens + childResult.usage.subLlmOutputTokens;
				usage.totalCost += childResult.usage.totalCost;

				return JSON.stringify({ response: childResult.summary });
			} catch (error) {
				return JSON.stringify({ error: error instanceof Error ? error.message : String(error) });
			}
		};

		// Build the driving model's conversation
		const maxTokens = Math.floor(0.8 * reserveTokens);
		const messageHistory: Message[] = [];

		// Build the initial task description
		let taskDescription = "Summarize a coding session conversation for context compaction.";
		if (customInstructions) {
			taskDescription += ` Focus: ${customInstructions}`;
		}
		if (previousSummary) {
			taskDescription += `\n\nThere is a previous compaction summary that should be updated/merged:\n${previousSummary}`;
		}
		if (recentText) {
			taskDescription += `\n\nThe user's most recent messages (kept in context, not needing summarization) are:\n${recentText.slice(0, 2000)}`;
		}

		const contextMeta =
			`Context: a string with ${conversationText.length} characters and ${conversationText.split("\n").length} lines. ` +
			`Loaded in the REPL as the \`context\` variable. Explore it using the python_repl tool.`;

		// Initial user message with task + metadata
		messageHistory.push({
			role: "user",
			content: `${contextMeta}\n\nTask: ${taskDescription}`,
			timestamp: Date.now(),
		} as Message);

		// ── Tool-use loop ──
		for (let i = 0; i < maxIterations; i++) {
			if (signal?.aborted) {
				throw new Error("RLM compaction aborted");
			}

			iterations++;

			// Call the driving model with tools
			const response = await completeSimple(
				model,
				{
					systemPrompt: RLM_SYSTEM_PROMPT,
					messages: messageHistory,
					tools: RLM_TOOLS,
				},
				{ maxTokens, signal, apiKey, reasoning: "high" },
			);

			accumulateUsage(usage, response, "driving");

			if (response.stopReason === "error") {
				throw new Error(`RLM driving model failed: ${response.errorMessage || "Unknown error"}`);
			}

			// Add assistant message to history
			messageHistory.push(response as Message);

			// If the model didn't make tool calls, prompt it to use tools
			if (response.stopReason !== "toolUse") {
				messageHistory.push({
					role: "user",
					content: "You must use either the python_repl tool to explore the context, or the submit_answer tool to provide the final summary. Please use a tool now.",
					timestamp: Date.now(),
				} as Message);
				continue;
			}

			// Process tool calls
			const toolCalls = response.content.filter(
				(c): c is { type: "toolCall"; id: string; name: string; arguments: Record<string, unknown> } =>
					c.type === "toolCall",
			);

			for (const toolCall of toolCalls) {
				if (signal?.aborted) {
					throw new Error("RLM compaction aborted");
				}

				if (toolCall.name === "submit_answer") {
					const args = toolCall.arguments as SubmitAnswerParams;
					let summary = args.summary;

					// If variable is specified, retrieve from kernel
					if (args.variable) {
						const retrieveResult = await executePython(`print(${args.variable})`, {
							cwd,
							timeoutMs: CELL_TIMEOUT_MS,
							signal,
							sessionId: kernelSessionId,
							kernelMode: "session",
						});
						if (retrieveResult.exitCode === 0 && retrieveResult.output.trim()) {
							summary = retrieveResult.output.trim();
						}
					}

					logger.debug("RLM compaction: submit_answer received", {
						depth: currentDepth,
						iteration: i + 1,
						summaryLength: summary.length,
					});

					return {
						summary,
						iterations,
						subLlmCalls: totalSubCalls,
						codeBlocksExecuted: totalCodeBlocks,
						usage,
					};
				}

				if (toolCall.name === "python_repl") {
					const args = toolCall.arguments as PythonReplParams;
					totalCodeBlocks++;

					// Execute code in the persistent kernel with inputProvider for llm_query IPC
					const execResult = await executePython(args.code, {
						cwd,
						timeoutMs: CELL_TIMEOUT_MS,
						signal,
						sessionId: kernelSessionId,
						kernelMode: "session",
						inputProvider,
					});

					// Build tool result: metadata-only for the overall output,
					// but include truncated print() output for the model's strategic peeks.
					const rawOutput = execResult.output.trim();
					let toolResultText: string;

					if (!rawOutput) {
						toolResultText = "[no output]";
					} else if (rawOutput.length <= MAX_PRINT_OUTPUT_CHARS) {
						// Short enough to show in full (the model's strategic print() calls)
						toolResultText = rawOutput;
					} else {
						// Too long — show metadata + truncated preview
						toolResultText =
							`${formatReplMetadata(rawOutput)}\n\n` +
							`Printed output (first ${MAX_PRINT_OUTPUT_CHARS} chars):\n` +
							rawOutput.slice(0, MAX_PRINT_OUTPUT_CHARS) +
							`\n... [${rawOutput.length - MAX_PRINT_OUTPUT_CHARS} more chars truncated]`;
					}

					if (execResult.exitCode !== 0) {
						toolResultText += `\n\n[Exit code: ${execResult.exitCode}]`;
					}

					// Push tool result to message history
					messageHistory.push({
						role: "toolResult",
						toolCallId: toolCall.id,
						toolName: "python_repl",
						content: [{ type: "text", text: toolResultText }],
						isError: execResult.exitCode !== 0,
						timestamp: Date.now(),
					} as Message);
				}
			}
		}

		// Max iterations reached — force submit_answer
		logger.debug("RLM compaction: max iterations reached, requesting submit_answer", {
			depth: currentDepth,
			iterations,
			subLlmCalls: totalSubCalls,
		});

		messageHistory.push({
			role: "user",
			content: "You have reached the maximum number of iterations. Call submit_answer now with whatever summary you have assembled.",
			timestamp: Date.now(),
		} as Message);

		const finalResponse = await completeSimple(
			model,
			{
				systemPrompt: RLM_SYSTEM_PROMPT,
				messages: messageHistory,
				tools: RLM_TOOLS,
			},
			{ maxTokens, signal, apiKey, reasoning: "high" },
		);

		accumulateUsage(usage, finalResponse, "driving");

		// Check if the model used submit_answer
		const finalToolCalls = finalResponse.content.filter(
			(c): c is { type: "toolCall"; id: string; name: string; arguments: Record<string, unknown> } =>
				c.type === "toolCall",
		);

		for (const tc of finalToolCalls) {
			if (tc.name === "submit_answer") {
				const args = tc.arguments as SubmitAnswerParams;
				let summary = args.summary;
				if (args.variable) {
					const retrieveResult = await executePython(`print(${args.variable})`, {
						cwd,
						timeoutMs: CELL_TIMEOUT_MS,
						signal,
						sessionId: kernelSessionId,
						kernelMode: "session",
					});
					if (retrieveResult.exitCode === 0 && retrieveResult.output.trim()) {
						summary = retrieveResult.output.trim();
					}
				}
				return {
					summary,
					iterations: iterations + 1,
					subLlmCalls: totalSubCalls,
					codeBlocksExecuted: totalCodeBlocks,
					usage,
				};
			}
		}

		// Fallback: extract text from the response
		const fallbackText = extractText(finalResponse);
		return {
			summary: fallbackText || "Compaction failed: model did not submit an answer.",
			iterations: iterations + 1,
			subLlmCalls: totalSubCalls,
			codeBlocksExecuted: totalCodeBlocks,
			usage,
		};
	} finally {
		// Always dispose the kernel session to free resources
		try {
			await disposeKernelSessionById(kernelSessionId);
		} catch (err) {
			logger.debug("RLM compaction: failed to dispose kernel session", {
				sessionId: kernelSessionId,
				error: err instanceof Error ? err.message : String(err),
			});
		}
	}
}
