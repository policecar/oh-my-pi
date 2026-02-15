/**
 * RLM-based session compaction.
 *
 * Implements the Recursive Language Models paradigm (Zhang, Kraska, Khattab —
 * arXiv 2512.24601) for session compaction, following the architectural
 * patterns from deepfates/cantrip:
 *
 *   1. Strict context isolation — REPL output is ALWAYS metadata-only in the
 *      driving model's conversation history. The model sees a char-count +
 *      150-char preview, never raw data. (cantrip: formatRlmMetadata)
 *   2. Forced sandbox exit — submit_answer() is a host function INSIDE the
 *      sandbox, NOT a separate tool. The model MUST call submit_answer() from
 *      within python_repl code. There is only ONE tool (python_repl).
 *      (cantrip: SIGNAL_FINAL pattern)
 *   3. stdin-based IPC — llm_query() and submit_answer() use Python's input()
 *      which blocks while the TypeScript host resolves asynchronously via the
 *      Jupyter stdin channel. (cantrip: Asyncify; not available in Bun, so we
 *      use Jupyter's input_request/input_reply instead.)
 *   4. Recursive sub-agents — llm_query() spawns a nested RLM agent at depth+1
 *      with its own kernel, context, and tools. Falls back to plain LLM at
 *      max depth. (cantrip: onLlmQuery recursion)
 *   5. Resource cleanup — try/finally for kernel disposal, token tracking across
 *      all recursion levels, structured progress events.
 *      (cantrip: child.sandbox.dispose(), usage tracking)
 *
 * Key difference from cantrip: llm_batch is sequential (not parallel) because
 * Python's input() serializes all IPC through a single stdin channel. Cantrip
 * uses Promise.all with concurrency 8 via QuickJS Asyncify, which is not
 * available in Bun (oven-sh/bun#20878).
 *
 * Architecture:
 *   - Conversation text → IPython kernel `context` variable (programmatic space)
 *   - Model calls python_repl tool → kernel executes → metadata-only result
 *   - submit_answer() in kernel → input() → host captures result → _SubmitSignal
 *   - llm_query() in kernel → input() → host resolves via inputProvider
 *   - Model keeps calling python_repl until submit_answer() terminates
 */
import type { AgentMessage } from "@oh-my-pi/pi-agent-core";
import type { AssistantMessage, Message, Model, Tool } from "@oh-my-pi/pi-ai";
import { completeSimple } from "@oh-my-pi/pi-ai";
import { Type, type Static } from "@sinclair/typebox";
import { logger } from "@oh-my-pi/pi-utils";
import { executePython, disposeKernelSessionById } from "../../ipy/executor";
import { convertToLlm } from "../../session/messages";
import { serializeConversation } from "./utils";

// ============================================================================
// Types
// ============================================================================

export interface RlmCompactionOptions {
	/** Model that drives the RLM loop (writes REPL code) */
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
	/** Progress callback for sub-agent activity (like cantrip's RlmProgressCallback) */
	onProgress?: RlmProgressCallback;
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

/** Structured progress events for sub-agent activity (mirrors cantrip's RlmProgressEvent). */
export type RlmProgressEvent =
	| { type: "sub_agent_start"; depth: number; query: string }
	| { type: "sub_agent_end"; depth: number }
	| { type: "batch_start"; depth: number; count: number }
	| { type: "batch_item"; depth: number; index: number; total: number; query: string }
	| { type: "batch_end"; depth: number };

export type RlmProgressCallback = (event: RlmProgressEvent) => void;

// ============================================================================
// Constants
// ============================================================================

/** Timeout for each Python cell execution (ms). Generous to allow for llm_query latency. */
const CELL_TIMEOUT_MS = 120_000;

// ============================================================================
// Tool Schema — single tool, like cantrip's js_rlm
// ============================================================================

const pythonReplSchema = Type.Object({
	code: Type.String({ description: "Python code to execute in the REPL environment" }),
});

type PythonReplParams = Static<typeof pythonReplSchema>;

const RLM_TOOLS: Tool[] = [
	{
		name: "python_repl",
		description:
			"Execute Python code in the persistent REPL. Results are returned as metadata. " +
			"You MUST use submit_answer() to return your final result.",
		parameters: pythonReplSchema,
	},
];

// ============================================================================
// Progress
// ============================================================================

/** Default progress callback: logs to debug in the tree format used by cantrip's console. */
function defaultProgress(depth: number): RlmProgressCallback {
	const indent = "  ".repeat(depth);
	return (event) => {
		switch (event.type) {
			case "sub_agent_start": {
				const preview = event.query.slice(0, 50) + (event.query.length > 50 ? "..." : "");
				logger.debug(`${indent}├─ [depth:${event.depth}] "${preview}"`);
				break;
			}
			case "sub_agent_end":
				logger.debug(`${indent}└─ [depth:${event.depth}] done`);
				break;
			case "batch_start":
				logger.debug(`${indent}├─ [depth:${event.depth}] llm_batch(${event.count} tasks)`);
				break;
			case "batch_item": {
				const preview = event.query.slice(0, 30) + (event.query.length > 30 ? "..." : "");
				logger.debug(`${indent}│  ├─ [${event.index + 1}/${event.total}] "${preview}"`);
				break;
			}
			case "batch_end":
				logger.debug(`${indent}└─ [depth:${event.depth}] batch complete`);
				break;
		}
	};
}

// ============================================================================
// Metadata Formatting — matches cantrip's formatRlmMetadata
// ============================================================================

/**
 * Format REPL output as metadata-only (cantrip's formatRlmMetadata).
 * Prevents the driving model's prompt history from being flooded with data.
 */
export function formatReplMetadata(output: string): string {
	if (!output || output === "undefined") return "[Result: undefined]";
	const length = output.length;
	const preview = output.slice(0, 150).replace(/\n/g, " ");
	return `[Result: ${length} chars] "${preview}${length > 150 ? "..." : ""}"`;
}

// ============================================================================
// Helpers
// ============================================================================

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
// Dynamic System Prompt — modeled on cantrip's getRlmSystemPrompt()
// ============================================================================

/**
 * Generate the RLM system prompt dynamically based on context metadata and
 * recursion depth. Modeled on cantrip's getRlmSystemPrompt().
 */
export function getRlmSystemPrompt(options: {
	contextLength: number;
	contextLines: number;
	contextPreview: string;
	hasRecursion: boolean;
}): string {
	const { contextLength, contextLines, contextPreview, hasRecursion } = options;

	const subLlmIntro = hasRecursion
		? "You can access, transform, and analyze this context interactively in a Python sandbox that can recursively query sub-LLMs. Use sub-LLMs when semantic understanding is needed; prefer code for structured, deterministic tasks."
		: "You can access, transform, and analyze this context interactively in a Python sandbox. The sandbox provides `llm_query` for semantic analysis of small snippets and code for data processing — choose the right approach for the task.";

	const subLlmNote = hasRecursion
		? "Spawns a recursive sub-agent with its own REPL and context. Useful for semantic analysis and ambiguous reasoning. For structured data and exact computations, prefer code."
		: "Useful for semantic analysis of small snippets. Note: context passed to sub-LLMs is truncated to ~10K chars, so prefer code for large-scale data processing.";

	const strategySection = hasRecursion
		? `### STRATEGY
First probe the context to understand its structure and size. Then choose the right approach:
- **Code-solvable tasks** (counting, filtering, searching, regex): Use Python directly. This is fast and exact.
- **Semantic tasks** (classification, summarization, understanding meaning): Use \`llm_query\`/\`llm_batch\` on individual items or small chunks.
- **Mixed tasks**: Combine both — use code to extract/chunk data, then \`llm_query\` to analyze each chunk.

If the context is large and unstructured, chunk it and delegate only the semantic pieces. For structured data, code is usually sufficient.`
		: `### STRATEGY
First probe the context to understand its structure and size. Then choose the right approach:
- **Code-solvable tasks** (counting, filtering, searching, regex): Use Python directly. This is fast and exact.
- **Semantic tasks** (classification, summarization, understanding meaning): Use \`llm_query\`/\`llm_batch\` on individual items or small chunks.
- **Mixed tasks**: Combine both — use code to extract/chunk data, then \`llm_query\` to analyze each chunk.

Analyze your input data before choosing a strategy. For structured data, code is usually sufficient. For unstructured text requiring comprehension, delegate to sub-LLMs.`;

	const preview = contextPreview.slice(0, 200).replace(/\n/g, " ");

	return `You are a session compaction agent tasked with summarizing a coding conversation. ${subLlmIntro} You will be queried iteratively until you provide a final answer via submit_answer().

### DATA ENVIRONMENT
A global variable \`context\` contains the full conversation:
- **Type**: string
- **Length**: ${contextLength} characters (${contextLines} lines)
- **Preview**: "${preview}..."

You MUST use the \`python_repl\` tool to explore this variable. You cannot see the data otherwise.
Make sure you look through the context sufficiently before answering.

### SANDBOX PHYSICS (IPython)
1. **PERSISTENCE**: Variables persist between \`python_repl\` tool calls.
2. **METADATA-ONLY**: Tool results show only a char-count and 150-char preview. Use \`llm_query()\` to analyze large data, or \`print()\` for small strategic peeks.
3. **BLOCKING**: All host functions (llm_query, llm_batch, submit_answer) are synchronous and blocking.

### HOST FUNCTIONS
- \`llm_query(prompt, sub_context=None)\`: Query a sub-LLM. Returns a string. ${subLlmNote}
- \`llm_batch(tasks)\`: Sequential delegation. Takes a list of \`{"query": ..., "context": ...}\` dicts (max 50). Returns a list of strings.
- \`submit_answer(result)\`: Terminates the task and returns \`result\` to the caller. This is the ONLY way to finish.
- \`print(...)\`: Prints output (visible as metadata preview in tool results).
- \`SHOW_VARS()\`: Lists all user-defined variables with types and sizes.

${strategySection}

### EXAMPLES

#### Probing context structure
\`\`\`python
print(f"Length: {len(context)}, Lines: {context.count(chr(10))}")
print(context[:500])
print("---")
print(context[-500:])
\`\`\`

#### Chunking and delegating to sub-LLMs
\`\`\`python
chunk_size = len(context) // 5
summaries = []
for i in range(5):
    chunk = context[i*chunk_size:(i+1)*chunk_size]
    summary = llm_query("Summarize this section of a coding conversation.", sub_context=chunk)
    summaries.append(summary)
    print(f"Chunk {i}: {summary[:100]}")
\`\`\`

#### Using llm_batch for multiple queries
\`\`\`python
import re
sections = re.split(r"\\n={3,}\\n", context)
print(f"Found {len(sections)} sections")
tasks = [{"query": "Extract key decisions and file changes.", "context": s} for s in sections]
results = llm_batch(tasks)
\`\`\`

#### Building up a summary iteratively
\`\`\`python
lines = context.split("\\n")
buffer = []
for i in range(0, len(lines), 100):
    chunk = "\\n".join(lines[i:i+100])
    info = llm_query("Extract relevant information for a session summary.", sub_context=chunk)
    buffer.append(info)
    print(f"Section {i//100}: {info[:80]}")

final = llm_query("Synthesize these section summaries into a comprehensive session summary:\\n" + "\\n---\\n".join(buffer))
submit_answer(final)
\`\`\`

### OUTPUT FORMAT

Your final summary (passed to submit_answer) must follow this structure:

\`\`\`
## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
- [Requirements, preferences mentioned]

## Progress

### Done
- [x] [Completed tasks with specific file paths and details]

### In Progress
- [ ] [Current incomplete work]

### Blocked
- [Issues preventing progress]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Data, pending questions, references needed to continue]

## Additional Notes
[Anything important not covered above]
\`\`\`

Preserve exact file paths, function names, error messages, and command outputs.

### RULES
- Do NOT print the entire \`context\` — it will be truncated and waste iterations.
- You MUST use \`python_repl\` to explore the context before calling \`submit_answer()\`.
- Tool results show only metadata. Use \`print()\` strategically and \`llm_query()\` for analysis.
- Be thorough: scan the full conversation, don't just summarize the beginning.
- Keep the summary concise but complete — it replaces the original conversation.

Think step by step, plan, and execute immediately — do not just say "I will do this". Use the sandbox and sub-LLMs when appropriate. Submit your final summary via \`submit_answer()\`.`;
}

// ============================================================================
// REPL Setup
// ============================================================================

/**
 * Build Python setup code that:
 * 1. Loads the serialized conversation into a `context` variable
 * 2. Defines submit_answer() as a host function (via stdin IPC + _SubmitSignal)
 *    — like cantrip's SIGNAL_FINAL pattern
 * 3. Defines llm_query() and llm_batch() using input() for stdin-based IPC
 * 4. Defines SHOW_VARS() helper
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

# --------------------------------------------------------------------------
# submit_answer: cantrip's SIGNAL_FINAL pattern adapted for Python
# --------------------------------------------------------------------------

class _SubmitSignal(BaseException):
    """Internal signal to terminate execution after submit_answer."""
    pass

def submit_answer(result):
    """Submit the final answer. Terminates execution.
    This is the ONLY way to finish the task."""
    if not isinstance(result, str):
        result = json.dumps(result, indent=2)
    request = json.dumps({"type": "submit_answer", "result": result})
    input(request)
    raise _SubmitSignal()

# --------------------------------------------------------------------------
# llm_query / llm_batch: recursive sub-LLM delegation via stdin IPC
# --------------------------------------------------------------------------

def llm_query(prompt, sub_context=None, model=None):
    """Query a sub-LLM. Returns the model's text response as a string.
    If sub_context is provided, the child agent explores that instead.
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

def llm_batch(tasks):
    """Process multiple queries sequentially. Max 50 tasks.
    Each task is a dict with 'query' and optional 'context', or a string.
    Returns a list of responses.
    Note: sequential because Python's input() serializes IPC (unlike cantrip's
    parallel Promise.all — not available without WASM Asyncify)."""
    if not isinstance(tasks, list):
        raise TypeError("llm_batch requires a list of tasks")
    if len(tasks) > 50:
        raise ValueError(f"llm_batch: too many tasks ({len(tasks)} > 50). Split into smaller batches.")
    results = []
    for task in tasks:
        if isinstance(task, str):
            results.append(llm_query(task))
        elif isinstance(task, dict):
            q = task.get("query") or task.get("input") or task.get("task", "")
            c = task.get("context") or task.get("sub_context")
            results.append(llm_query(q, sub_context=c))
        else:
            raise TypeError(f"Each task must be a string or dict, got {type(task)}")
    return results

# Backward-compatibility alias
llm_query_batched = lambda prompts, sub_contexts=None, model=None: [
    llm_query(p, sub_context=sub_contexts[i] if sub_contexts and i < len(sub_contexts) else None, model=model)
    for i, p in enumerate(prompts)
]

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
 * The model has a single tool:
 *   - `python_repl`: Execute Python code in a REPL with `context` variable
 *
 * The REPL provides host functions (like cantrip's registerRlmFunctions):
 *   - `submit_answer(result)`: Terminates via SIGNAL_FINAL pattern
 *   - `llm_query(prompt, sub_context?)`: Recursive sub-agent delegation
 *   - `llm_batch(tasks)`: Sequential batch delegation (max 50)
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
	const progress = options.onProgress ?? defaultProgress(currentDepth);

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

	// Captured submit_answer result — set by inputProvider when submit_answer()
	// is called from within the REPL (cantrip's SIGNAL_FINAL pattern).
	let pendingSubmitAnswer: string | undefined;

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

		// inputProvider: handles submit_answer() and llm_query() calls from
		// Python's input(). Like cantrip's registerRlmFunctions + onLlmQuery.
		const inputProvider = async (prompt: string): Promise<string> => {
			let request: { type: string; prompt?: string; context?: string; model?: string; result?: string };
			try {
				request = JSON.parse(prompt);
			} catch {
				return JSON.stringify({ error: "Invalid request JSON" });
			}

			// submit_answer: capture result and acknowledge (SIGNAL_FINAL pattern)
			if (request.type === "submit_answer") {
				if (pendingSubmitAnswer === undefined) {
					pendingSubmitAnswer = request.result ?? "";
				}
				return JSON.stringify({ ack: true });
			}

			if (request.type !== "llm_query") {
				return JSON.stringify({ error: `Unknown request type: ${request.type}` });
			}

			totalSubCalls++;
			const childDepth = currentDepth + 1;
			const queryText = request.prompt ?? "";
			progress({ type: "sub_agent_start", depth: childDepth, query: queryText });

			let result: string;

			if (currentDepth >= maxRecursionDepth) {
				// At max depth — fall back to plain LLM call with truncated context
				// (matches cantrip: "if depth >= maxDepth, fall back to plain LLM
				// completion with truncated context (10,000 char limit)")
				logger.debug("RLM llm_query at max depth, using plain LLM", {
					depth: currentDepth,
					promptLength: queryText.length,
				});
				const truncatedPrompt = queryText.length > 10_000
					? queryText.slice(0, 10_000) + "\n... [truncated]"
					: queryText;
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
					result = extractText(response);
				} catch (error) {
					progress({ type: "sub_agent_end", depth: childDepth });
					return JSON.stringify({ error: error instanceof Error ? error.message : String(error) });
				}
			} else {
				// Recursive: spawn a nested RLM agent (like cantrip's child RLM)
				logger.debug("RLM llm_query spawning child agent", {
					depth: childDepth,
					promptLength: queryText.length,
					hasSubContext: !!request.context,
				});

				try {
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
						customInstructions: queryText,
						cwd,
						reserveTokens,
						depth: childDepth,
						maxDepth: maxRecursionDepth,
						onProgress: progress,
					});

					// Aggregate child usage
					totalSubCalls += childResult.subLlmCalls;
					totalCodeBlocks += childResult.codeBlocksExecuted;
					usage.subLlmInputTokens += childResult.usage.drivingInputTokens + childResult.usage.subLlmInputTokens;
					usage.subLlmOutputTokens += childResult.usage.drivingOutputTokens + childResult.usage.subLlmOutputTokens;
					usage.totalCost += childResult.usage.totalCost;

					result = childResult.summary;
				} catch (error) {
					progress({ type: "sub_agent_end", depth: childDepth });
					return JSON.stringify({ error: error instanceof Error ? error.message : String(error) });
				}
			}

			progress({ type: "sub_agent_end", depth: childDepth });
			return JSON.stringify({ response: result });
		};

		// Generate dynamic system prompt (like cantrip's getRlmSystemPrompt)
		const contextPreview = conversationText.slice(0, 200);
		const contextLines = conversationText.split("\n").length;
		const systemPrompt = getRlmSystemPrompt({
			contextLength: conversationText.length,
			contextLines,
			contextPreview,
			hasRecursion: currentDepth < maxRecursionDepth,
		});

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

		// Initial user message with task
		messageHistory.push({
			role: "user",
			content: `Task: ${taskDescription}`,
			timestamp: Date.now(),
		} as Message);

		// ── Tool-use loop (single tool: python_repl) ──
		for (let i = 0; i < maxIterations; i++) {
			if (signal?.aborted) {
				throw new Error("RLM compaction aborted");
			}

			iterations++;

			// Call the driving model with tools (no extended thinking — cantrip doesn't use it)
			const response = await completeSimple(
				model,
				{
					systemPrompt,
					messages: messageHistory,
					tools: RLM_TOOLS,
				},
				{ maxTokens, signal, apiKey },
			);

			accumulateUsage(usage, response, "driving");

			if (response.stopReason === "error") {
				throw new Error(`RLM driving model failed: ${response.errorMessage || "Unknown error"}`);
			}

			// Add assistant message to history
			messageHistory.push(response as Message);

			// If the model didn't make tool calls, prompt it to use the tool
			if (response.stopReason !== "toolUse") {
				messageHistory.push({
					role: "user",
					content: "You must use the python_repl tool to explore the context and call submit_answer() from within the REPL when done. Please use the python_repl tool now.",
					timestamp: Date.now(),
				} as Message);
				continue;
			}

			// Process tool calls (only python_repl)
			const toolCalls = response.content.filter(
				(c): c is { type: "toolCall"; id: string; name: string; arguments: Record<string, unknown> } =>
					c.type === "toolCall",
			);

			for (const toolCall of toolCalls) {
				if (signal?.aborted) {
					throw new Error("RLM compaction aborted");
				}

				if (toolCall.name !== "python_repl") {
					// Unknown tool — send error result
					messageHistory.push({
						role: "toolResult",
						toolCallId: toolCall.id,
						toolName: toolCall.name,
						content: [{ type: "text", text: `Unknown tool "${toolCall.name}". Use python_repl and call submit_answer() from within the REPL.` }],
						isError: true,
						timestamp: Date.now(),
					} as Message);
					continue;
				}

				const args = toolCall.arguments as PythonReplParams;
				totalCodeBlocks++;

				// Execute code in the persistent kernel with inputProvider
				const execResult = await executePython(args.code, {
					cwd,
					timeoutMs: CELL_TIMEOUT_MS,
					signal,
					sessionId: kernelSessionId,
					kernelMode: "session",
					inputProvider,
				});

				// Check if submit_answer was called from within the REPL
				// (cantrip's SIGNAL_FINAL pattern — execution terminates, we have the result)
				if (pendingSubmitAnswer !== undefined) {
					const summary = pendingSubmitAnswer;
					logger.debug("RLM compaction: submit_answer received via REPL", {
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

				// Strict metadata-only tool result (like cantrip's formatRlmMetadata)
				let toolResultText = formatReplMetadata(execResult.output);
				if (execResult.exitCode !== 0) {
					toolResultText += `\n[Exit code: ${execResult.exitCode}]`;
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

		// Max iterations reached — ask the model to call submit_answer() in REPL
		logger.debug("RLM compaction: max iterations reached", {
			depth: currentDepth,
			iterations,
			subLlmCalls: totalSubCalls,
		});

		messageHistory.push({
			role: "user",
			content: "You have reached the maximum number of iterations. Call submit_answer() now from the python_repl with whatever summary you have assembled.",
			timestamp: Date.now(),
		} as Message);

		const finalResponse = await completeSimple(
			model,
			{
				systemPrompt,
				messages: messageHistory,
				tools: RLM_TOOLS,
			},
			{ maxTokens, signal, apiKey },
		);

		accumulateUsage(usage, finalResponse, "driving");
		messageHistory.push(finalResponse as Message);

		// Process the final response's tool calls
		const finalToolCalls = finalResponse.content.filter(
			(c): c is { type: "toolCall"; id: string; name: string; arguments: Record<string, unknown> } =>
				c.type === "toolCall",
		);

		for (const toolCall of finalToolCalls) {
			if (toolCall.name === "python_repl") {
				const args = toolCall.arguments as PythonReplParams;
				totalCodeBlocks++;
				await executePython(args.code, {
					cwd,
					timeoutMs: CELL_TIMEOUT_MS,
					signal,
					sessionId: kernelSessionId,
					kernelMode: "session",
					inputProvider,
				});
				if (pendingSubmitAnswer !== undefined) {
					return {
						summary: pendingSubmitAnswer,
						iterations: iterations + 1,
						subLlmCalls: totalSubCalls,
						codeBlocksExecuted: totalCodeBlocks,
						usage,
					};
				}
			}
		}

		// Fallback: extract text from the response (shouldn't happen with well-behaved models)
		const fallbackText = extractText(finalResponse);
		return {
			summary: fallbackText || "Compaction failed: model did not call submit_answer().",
			iterations: iterations + 1,
			subLlmCalls: totalSubCalls,
			codeBlocksExecuted: totalCodeBlocks,
			usage,
		};
	} finally {
		// Always dispose the kernel session to free resources
		// (cantrip: child.sandbox.dispose())
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
