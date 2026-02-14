/**
 * RLM-based session compaction.
 *
 * Implements the Recursive Language Models paradigm (Zhang, Kraska, Khattab —
 * arXiv 2512.24601) for session compaction. Instead of feeding the full
 * conversation into a single LLM summarization call, the conversation is
 * loaded into oh-my-pi's IPython kernel as a `context` variable. The model
 * then writes Python code to programmatically examine, search, chunk, and
 * delegate analysis to sub-LLM calls — discovering its own summarization
 * strategy rather than following a hard-coded algorithm.
 *
 * Architecture:
 *   - Conversation text → temp file → IPython kernel `context` variable
 *   - `llm_query(prompt)` in kernel → routes to completeSimple() with smol model
 *   - Iterative loop: model generates ```repl code → kernel executes → output fed back
 *   - Model calls FINAL(summary) when done
 */
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { Model, Message } from "@oh-my-pi/pi-ai";
import { completeSimple } from "@oh-my-pi/pi-ai";
import { logger } from "@oh-my-pi/pi-utils";
import { renderPromptTemplate } from "../../config/prompt-templates";
import { executePython } from "../../ipy/executor";
import { convertToLlm } from "../../session/messages";
import { serializeConversation } from "./utils";
import rlmCompactionSystemPrompt from "../../prompts/compaction/rlm-compaction-system.md" with { type: "text" };

// ============================================================================
// Types
// ============================================================================

export interface RlmCompactionOptions {
	/** Model that drives the RLM loop (writes REPL code, calls FINAL) */
	model: Model;
	/** API key for the driving model */
	apiKey: string;
	/** Optional cheaper model for llm_query() sub-calls from the REPL */
	leafModel?: Model;
	/** API key for the leaf model */
	leafApiKey?: string;
	/** Maximum iterations of the generate-code → execute → observe loop */
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
}

// ============================================================================
// Constants
// ============================================================================

/** Maximum characters of REPL output to feed back to the model per iteration */
const MAX_OUTPUT_CHARS = 20_000;

/** Timeout for each Python cell execution (seconds) */
const CELL_TIMEOUT_SEC = 30;

/** Regex to find ```repl code blocks in model output.
 *  Uses a tempered greedy token to avoid matching across block boundaries. */
const REPL_CODE_BLOCK_RE = /```repl\s*\n((?:(?!```)[\s\S])*?)\n```/g;

/** Regex to find FINAL(answer) in model output */
const FINAL_RE = /^\s*FINAL\(([\s\S]*)\)\s*$/m;

/** Regex to find FINAL_VAR(varname) in model output */
const FINAL_VAR_RE = /^\s*FINAL_VAR\((\w+)\)\s*$/m;

// ============================================================================
// System Prompt
// ============================================================================

const RLM_SYSTEM_PROMPT = renderPromptTemplate(rlmCompactionSystemPrompt);

// ============================================================================
// Code Parsing
// ============================================================================

interface CodeBlock {
	code: string;
}

export function findCodeBlocks(text: string): CodeBlock[] {
	const blocks: CodeBlock[] = [];
	let match: RegExpExecArray | null;
	const re = new RegExp(REPL_CODE_BLOCK_RE.source, "g");
	while ((match = re.exec(text)) !== null) {
		const code = match[1].trim();
		if (code) {
			blocks.push({ code });
		}
	}
	return blocks;
}

export function findFinalAnswer(text: string): string | null {
	// Check for FINAL(answer)
	const finalMatch = FINAL_RE.exec(text);
	if (finalMatch) {
		return finalMatch[1].trim();
	}
	return null;
}

export function findFinalVar(text: string): string | null {
	const match = FINAL_VAR_RE.exec(text);
	if (match) {
		return match[1].trim();
	}
	return null;
}

// ============================================================================
// REPL Setup
// ============================================================================

/**
 * Build Python setup code that:
 * 1. Loads the serialized conversation into a `context` variable
 * 2. Defines llm_query() that communicates back to the TypeScript process
 *    via a temp file protocol (since we can't inject TCP sockets into IPython)
 */
export function buildSetupCode(contextText: string, requestFile: string, responseFile: string): string {
	// Escape for Python triple-quoted string
	const escaped = contextText
		.replace(/\\/g, "\\\\")
		.replace(/"""/g, '\\"\\"\\"');

	return `
import json, time, os

# Load conversation context
context = """${escaped}"""

# Metadata
context_length = len(context)
context_lines = context.count("\\n") + 1

# Request/response files for llm_query communication
_RLM_REQUEST_FILE = ${JSON.stringify(requestFile)}
_RLM_RESPONSE_FILE = ${JSON.stringify(responseFile)}

def llm_query(prompt, model=None):
    """Query a sub-LLM. The prompt can be up to ~500K characters.
    Returns the model's text response as a string.
    Use this to analyze chunks of context that are too large to print."""
    request = json.dumps({"prompt": str(prompt), "model": model})
    with open(_RLM_REQUEST_FILE, "w") as f:
        f.write(request)
    # Signal ready
    with open(_RLM_REQUEST_FILE + ".ready", "w") as f:
        f.write("1")
    # Wait for response (poll)
    for _ in range(600):  # 60 second timeout
        if os.path.exists(_RLM_RESPONSE_FILE + ".ready"):
            break
        time.sleep(0.1)
    else:
        raise TimeoutError("llm_query timed out waiting for response")
    with open(_RLM_RESPONSE_FILE, "r") as f:
        response = json.loads(f.read())
    # Cleanup
    try:
        os.remove(_RLM_RESPONSE_FILE + ".ready")
    except OSError:
        pass
    if "error" in response:
        raise RuntimeError(f"llm_query failed: {response['error']}")
    return response["response"]

def llm_query_batched(prompts, model=None):
    """Query a sub-LLM with multiple prompts. Returns a list of responses."""
    return [llm_query(p, model) for p in prompts]

def SHOW_VARS():
    """Print all user-defined variables and their types/sizes."""
    for name, val in sorted(globals().items()):
        if name.startswith("_") or callable(val) or name in ("json", "time", "os", "In", "Out"):
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
// llm_query Handler
// ============================================================================

/**
 * Start a background polling loop that handles llm_query requests from the
 * Python kernel. Returns a handle to stop the loop and await the total count.
 *
 * This MUST run concurrently with executePython — the Python side blocks
 * inside llm_query() waiting for a response file, so the TypeScript handler
 * needs to detect and serve requests while the kernel is still executing.
 */
function startLlmQueryHandler(
	requestFile: string,
	responseFile: string,
	model: Model,
	apiKey: string,
	signal?: AbortSignal,
): { stop: () => void; result: Promise<number> } {
	let stopped = false;
	const stop = () => { stopped = true; };

	const result = (async () => {
		let callCount = 0;
		const readyFile = requestFile + ".ready";
		const fs = await import("node:fs/promises");

		while (!stopped && !signal?.aborted) {
			// Check if there's a pending request
			let hasPending = false;
			try {
				await fs.stat(readyFile);
				hasPending = true;
			} catch {
				// No request pending — wait before polling again
				await new Promise(r => setTimeout(r, 50));
				continue;
			}

			if (!hasPending) continue;

			// Read request
			const requestText = await Bun.file(requestFile).text();
			const request = JSON.parse(requestText) as { prompt: string; model?: string };

			// Clean up ready signal
			try { await fs.unlink(readyFile); } catch {}

			callCount++;
			logger.debug("RLM llm_query call", { callIndex: callCount, promptLength: request.prompt.length });

			try {
				const response = await completeSimple(
					model,
					{
						messages: [{
							role: "user",
							content: [{ type: "text", text: request.prompt }],
							timestamp: Date.now(),
						}],
					},
					{
						apiKey,
						signal,
						maxTokens: 8192,
					},
				);

				const text = response.content
					.filter((c): c is { type: "text"; text: string } => c.type === "text")
					.map(c => c.text)
					.join("\n");

				await Bun.write(responseFile, JSON.stringify({ response: text }));
			} catch (error) {
				const msg = error instanceof Error ? error.message : String(error);
				await Bun.write(responseFile, JSON.stringify({ error: msg }));
			}

			// Signal response ready
			await Bun.write(responseFile + ".ready", "1");
		}

		return callCount;
	})();

	return { stop, result };
}

// ============================================================================
// Main RLM Loop
// ============================================================================

/**
 * Run RLM-based compaction on a set of messages.
 *
 * The model is given access to a Python REPL with:
 *   - `context`: the serialized conversation text
 *   - `llm_query(prompt)`: delegates to a sub-LLM
 *   - `SHOW_VARS()`: lists variables in scope
 *
 * The model iteratively writes code to explore, chunk, analyze, and
 * summarize the conversation, then calls FINAL(summary) when done.
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

	// Serialize conversation
	const allMessages = [...messagesToSummarize];
	const llmMessages = convertToLlm(allMessages);
	const conversationText = serializeConversation(llmMessages);

	// Also serialize recent messages for task context
	const recentLlmMessages = convertToLlm(recentMessages);
	const recentText = serializeConversation(recentLlmMessages);

	// Create temp files for llm_query IPC
	const tmpDir = await import("node:os").then(os => os.tmpdir());
	const sessionId = `rlm-${Date.now()}-${Math.random().toString(36).slice(2)}`;
	const requestFile = `${tmpDir}/${sessionId}-request.json`;
	const responseFile = `${tmpDir}/${sessionId}-response.json`;

	// Set up the REPL environment
	const setupCode = buildSetupCode(conversationText, requestFile, responseFile);

	logger.debug("RLM compaction: setting up REPL", {
		contextChars: conversationText.length,
		contextLines: conversationText.split("\n").length,
		recentChars: recentText.length,
	});

	// Execute setup in a persistent session kernel so variables survive across iterations
	const kernelSessionId = `rlm-compaction-${sessionId}`;
	const setupResult = await executePython(setupCode, {
		cwd,
		timeoutMs: CELL_TIMEOUT_SEC * 1000,
		signal,
		sessionId: kernelSessionId,
		kernelMode: "session",
	});

	if (setupResult.exitCode !== 0) {
		throw new Error(`RLM REPL setup failed: ${setupResult.output}`);
	}

	// Build the RLM conversation for the driving model
	const maxTokens = Math.floor(0.8 * reserveTokens);
	// Message history for the driving model's conversation.
	// Uses a looser type since we build assistant messages without full pi-ai fields;
	// cast to Message[] at completeSimple() call sites.
	const messageHistory: Array<{
		role: "user" | "assistant";
		content: string | Array<{ type: string; text: string }>;
		timestamp: number;
	}> = [];

	// Build the initial user prompt
	let taskDescription = "Summarize a coding session conversation for context compaction.";
	if (customInstructions) {
		taskDescription += ` Focus: ${customInstructions}`;
	}

	const contextMeta = `Your context is a str with ${conversationText.length} total characters and ${conversationText.split("\n").length} lines.`;
	if (previousSummary) {
		taskDescription += `\n\nThere is a previous compaction summary that should be updated/merged:\n${previousSummary}`;
	}
	if (recentText) {
		taskDescription += `\n\nThe user's most recent messages (kept in context, not needing summarization) are:\n${recentText.slice(0, 2000)}`;
	}

	// Seed with context metadata (like RLM does)
	messageHistory.push({
		role: "assistant",
		content: [{ type: "text", text: contextMeta }],
		timestamp: Date.now(),
	});

	let iterations = 0;
	let totalSubCalls = 0;
	let totalCodeBlocks = 0;

	// Use a dedicated kernel session for all iterations
	// (setup already created variables; subsequent executePython calls reuse the session)

	for (let i = 0; i < maxIterations; i++) {
		if (signal?.aborted) {
			throw new Error("RLM compaction aborted");
		}

		iterations++;

		// Build user prompt for this iteration
		let iterPrompt: string;
		if (i === 0) {
			iterPrompt = `You have not interacted with the REPL environment or seen your context yet. Your next action should be to explore the context and figure out how to produce a comprehensive summary.

Think step-by-step about what to do using the REPL environment (which contains the \`context\` variable) and the \`llm_query()\` function to produce your summary.

Task: ${taskDescription}

Continue using the REPL environment and write code in \`\`\`repl\`\`\` tags. Your next action:`;
		} else {
			iterPrompt = `The above is your previous interactions with the REPL environment. Think step-by-step about what to do next.

Continue using the REPL environment to work toward a comprehensive summary. When you have enough information, call FINAL(your_summary) with the complete summary text.

Your next action:`;
		}

		messageHistory.push({
			role: "user",
			content: [{ type: "text", text: iterPrompt }],
			timestamp: Date.now(),
		});

		// Call the driving model
		const response = await completeSimple(
			model,
			{
				systemPrompt: RLM_SYSTEM_PROMPT,
				messages: messageHistory as Message[],
			},
			{ maxTokens, signal, apiKey, reasoning: "high" },
		);

		if (response.stopReason === "error") {
			throw new Error(`RLM driving model failed: ${response.errorMessage || "Unknown error"}`);
		}

		const responseText = response.content
			.filter((c): c is { type: "text"; text: string } => c.type === "text")
			.map(c => c.text)
			.join("\n");

		// Check for FINAL answer before executing code
		const finalAnswer = findFinalAnswer(responseText);
		if (finalAnswer) {
			logger.debug("RLM compaction: FINAL answer received", {
				iteration: i + 1,
				summaryLength: finalAnswer.length,
			});
			return {
				summary: finalAnswer,
				iterations,
				subLlmCalls: totalSubCalls,
				codeBlocksExecuted: totalCodeBlocks,
			};
		}

		// Extract and execute code blocks
		const codeBlocks = findCodeBlocks(responseText);

		if (codeBlocks.length === 0) {
			// Model didn't write code and didn't call FINAL — add response and continue
			messageHistory.push({
				role: "assistant",
				content: [{ type: "text", text: responseText }],
				timestamp: Date.now(),
			});
			continue;
		}

		// Add assistant response to history
		messageHistory.push({
			role: "assistant",
			content: [{ type: "text", text: responseText }],
			timestamp: Date.now(),
		});

		// Execute each code block
		for (const block of codeBlocks) {
			if (signal?.aborted) {
				throw new Error("RLM compaction aborted");
			}

			totalCodeBlocks++;

			// Start the llm_query handler BEFORE executing code.
			// Python's llm_query() blocks waiting for a response file, so
			// the handler must run concurrently to serve those requests.
			const subQueryModel = leafModel || model;
			const subQueryApiKey = leafApiKey || apiKey;
			const handler = startLlmQueryHandler(
				requestFile,
				responseFile,
				subQueryModel,
				subQueryApiKey,
				signal,
			);

			// Execute code in the persistent kernel (may call llm_query)
			const execResult = await executePython(block.code, {
				cwd,
				timeoutMs: CELL_TIMEOUT_SEC * 1000,
				signal,
				sessionId: kernelSessionId,
				kernelMode: "session",
			});

			// Python execution finished — stop the handler and collect stats
			handler.stop();
			const subCalls = await handler.result;
			totalSubCalls += subCalls;

			// Truncate output for the model
			let output = execResult.output.trim();
			if (output.length > MAX_OUTPUT_CHARS) {
				output = output.slice(0, MAX_OUTPUT_CHARS) + `\n... [truncated, ${output.length - MAX_OUTPUT_CHARS} more chars]`;
			}

			// Check for FINAL_VAR in the code output or model response
			const finalVarName = findFinalVar(responseText);
			if (finalVarName) {
				// Retrieve the variable value
				const retrieveResult = await executePython(`print(${finalVarName})`, {
					cwd,
					timeoutMs: CELL_TIMEOUT_SEC * 1000,
					signal,
					sessionId: kernelSessionId,
					kernelMode: "session",
				});
				if (retrieveResult.exitCode === 0 && retrieveResult.output.trim()) {
					return {
						summary: retrieveResult.output.trim(),
						iterations,
						subLlmCalls: totalSubCalls,
						codeBlocksExecuted: totalCodeBlocks,
					};
				}
			}

			// Feed output back to the model
			const statusSuffix = execResult.exitCode !== 0
				? `\n\n[Exit code: ${execResult.exitCode}]`
				: "";

			messageHistory.push({
				role: "user",
				content: [{
					type: "text",
					text: `Code executed:\n\`\`\`python\n${block.code}\n\`\`\`\n\nREPL output:\n${output || "(no output)"}${statusSuffix}`,
				}],
				timestamp: Date.now(),
			});
		}
	}

	// Max iterations reached — ask for a final answer
	logger.debug("RLM compaction: max iterations reached, requesting final answer", {
		iterations,
		subLlmCalls: totalSubCalls,
	});

	messageHistory.push({
		role: "user",
		content: [{
			type: "text",
			text: "You have reached the maximum number of iterations. Please provide a final comprehensive summary now using FINAL(your_summary).",
		}],
		timestamp: Date.now(),
	});

	const finalResponse = await completeSimple(
		model,
		{
			systemPrompt: RLM_SYSTEM_PROMPT,
			messages: messageHistory as Message[],
		},
		{ maxTokens, signal, apiKey, reasoning: "high" },
	);

	const finalText = finalResponse.content
		.filter((c): c is { type: "text"; text: string } => c.type === "text")
		.map(c => c.text)
		.join("\n");

	const finalAnswer = findFinalAnswer(finalText);

	// Cleanup temp files
	try {
		const fs = await import("node:fs/promises");
		await fs.unlink(requestFile).catch(() => {});
		await fs.unlink(responseFile).catch(() => {});
		await fs.unlink(requestFile + ".ready").catch(() => {});
		await fs.unlink(responseFile + ".ready").catch(() => {});
	} catch {}

	return {
		summary: finalAnswer || finalText,
		iterations: iterations + 1,
		subLlmCalls: totalSubCalls,
		codeBlocksExecuted: totalCodeBlocks,
	};
}
