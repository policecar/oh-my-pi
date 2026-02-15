# Plan: Improve RLM Compaction — Five Cantrip-Inspired Improvements

## Overview

Five improvements to make our RLM compaction match the architectural rigor of cantrip's RLM implementation. Each is an independent workstream but they interact — (1) and (2) reshape the driving loop, (3) replaces the IPC mechanism that (4) relies on, and (5) wraps everything in proper lifecycle management.

---

## 1. Strict Context Isolation — Metadata-Only Tool Results

**Problem:** We feed up to 20,000 chars of raw REPL output per code block into `messageHistory`. Over several iterations this bloats the driving model's token window with exactly the data RLM is supposed to keep *outside* it.

**What cantrip does:** `js_rlm` returns only `[Result: N chars] "first 150 chars..."` to the conversation history. The model must use `console.log()` strategically or delegate to `llm_query()` for semantic analysis.

### Changes

**`rlm-compaction.ts`** — After `executePython` returns, format the result as metadata-only:

```typescript
// Current (bad — leaks data into token window):
let output = execResult.output.trim();
if (output.length > MAX_OUTPUT_CHARS) {
    output = output.slice(0, MAX_OUTPUT_CHARS) + `\n... [truncated]`;
}
messageHistory.push({ role: "user", content: [{ type: "text",
    text: `REPL output:\n${output}` }] });

// New (metadata-only):
const outputMeta = formatReplMetadata(execResult.output);
messageHistory.push({ role: "user", content: [{ type: "text",
    text: `REPL result: ${outputMeta}` }] });
```

Add a `formatReplMetadata` function (matching cantrip's `formatRlmMetadata`):

```typescript
function formatReplMetadata(output: string): string {
    const trimmed = output.trim();
    if (!trimmed || trimmed === "undefined") return "[no output]";
    const lines = trimmed.split("\n");
    const preview = trimmed.slice(0, 200).replace(/\n/g, "\\n");
    return `[${trimmed.length} chars, ${lines.length} lines] "${preview}${trimmed.length > 200 ? "..." : ""}"`;
}
```

**Exception: `print()` output.** The model needs *some* way to see specific data. We differentiate:
- Captured `stdout` from explicit `print()` calls: shown in full (up to a smaller cap, e.g. 2,000 chars). This is the model's strategic "peek" mechanism.
- Overall execution result / return value: metadata-only.

But `executePython` combines stdout and return value into a single `output` string. The IPython kernel distinguishes them via message types (`stream` vs `execute_result`). So we need to modify our usage to capture them separately.

**`kernel.ts` / `executor.ts` interaction:** The kernel already distinguishes `stream` (stdout) from `execute_result` (return value) in its message routing (lines 715-772). The `OutputSink` receives both via `onChunk`. To separate them, we need to:
- Add a `streamOutput` field to `KernelExecuteResult` (or extend `PythonResult`) to separate stdout from the execution result.
- OR: use the existing `displayOutputs` array, which captures `display_data` / `execute_result` messages, while stdout goes through `onChunk`.

**Approach:** Add a `stdoutOnly: string` field to `PythonResult` by tracking `stream` messages separately from `execute_result` in `executeWithKernel`. Feed `stdoutOnly` (the model's deliberate prints, capped at 2K chars) back to `messageHistory`, and use metadata-only for the overall output.

**`rlm-compaction-system.md`** — Update the system prompt:
- Explain that REPL results are metadata-only.
- Emphasize: "Use `print()` to see specific data. Only printed output appears in your conversation. The execution result is summarized as metadata."
- Add examples showing the peek-via-print pattern.

### Files Modified
- `packages/coding-agent/src/session/compaction/rlm-compaction.ts` — `formatReplMetadata()`, change output handling in the code block execution loop
- `packages/coding-agent/src/ipy/executor.ts` — Add `stdoutOnly` field to `PythonResult`, track `stream` messages separately in `executeWithKernel`
- `packages/coding-agent/src/ipy/kernel.ts` — Expose stdout vs execute_result distinction in `KernelExecuteResult`
- `packages/coding-agent/src/prompts/compaction/rlm-compaction-system.md` — Update strategy docs

---

## 2. Single-Tool Forced Sandbox Exit

**Problem:** The model emits ````repl` code blocks as text, and calls `FINAL()` as text. Nothing forces it to actually use the sandbox — it can write `FINAL(generic summary)` on iteration 1 without exploring.

**What cantrip does:** Gives the agent ONLY `js_rlm` as a tool (via the tool_use API). The only way to finish is `submit_answer()` from within the sandbox, thrown as `SIGNAL_FINAL:` → `TaskComplete`. The model literally cannot produce a final answer without going through the sandbox.

### Changes

**Replace text-based code parsing with proper tool_use.** Instead of extracting ````repl` blocks from the model's text output, define two tools and pass them via `context.tools` to `completeSimple`:

**Tool 1: `python_repl`** — Execute Python code in the RLM kernel.
```typescript
const rlmPythonReplSchema = Type.Object({
    code: Type.String({ description: "Python code to execute in the REPL" }),
});
```

**Tool 2: `submit_answer`** — Provide the final compaction summary.
```typescript
const rlmSubmitAnswerSchema = Type.Object({
    summary: Type.String({ description: "The final compaction summary" }),
});
```

**`rlm-compaction.ts`** — Restructure the main loop:

```typescript
// Define tools
const tools = [
    { name: "python_repl", description: "Execute Python code in the REPL...", parameters: rlmPythonReplSchema },
    { name: "submit_answer", description: "Submit the final summary...", parameters: rlmSubmitAnswerSchema },
];

// In the loop, pass tools to completeSimple:
const response = await completeSimple(model, {
    systemPrompt: RLM_SYSTEM_PROMPT,
    messages: messageHistory as Message[],
    tools,  // <-- forces tool_use
}, { maxTokens, signal, apiKey, reasoning: "high" });

// Handle tool calls instead of parsing text:
for (const block of response.content) {
    if (block.type === "toolCall") {
        if (block.name === "submit_answer") {
            return { summary: block.arguments.summary, ... };
        }
        if (block.name === "python_repl") {
            const execResult = await executePython(block.arguments.code, ...);
            // Push tool result back to messageHistory
            messageHistory.push({ role: "toolResult", toolCallId: block.id, ... });
        }
    }
}
```

**Key behavioral change:** When `stopReason === "toolUse"`, the model MUST be calling one of our tools. When `stopReason === "stop"`, the model is trying to respond without a tool — we should prompt it to use `submit_answer` or `python_repl`.

**`rlm-compaction-system.md`** — Rewrite to describe tools instead of ````repl` syntax:
- "You have two tools: `python_repl` and `submit_answer`."
- "You MUST explore the context via `python_repl` before calling `submit_answer`."
- Remove all ````repl` block documentation.

**Remove:** `findCodeBlocks()`, `findFinalAnswer()`, `findFinalVar()`, `REPL_CODE_BLOCK_RE`, `FINAL_RE`, `FINAL_VAR_RE`. These are no longer needed when using proper tool_use.

**Keep `FINAL_VAR` as a tool variant?** No. `submit_answer` takes the summary string directly. If the model wants to build it in a variable, it can do `submit_answer(summary=my_var)` — but since tool args are JSON, the model must `print()` the variable first and then paste it. Alternatively, add a `variable_name` parameter to `submit_answer` that we retrieve from the kernel. Decision: add an optional `variable` field to the schema.

### Files Modified
- `packages/coding-agent/src/session/compaction/rlm-compaction.ts` — Replace text parsing with tool_use loop, define tool schemas, handle tool calls and tool results
- `packages/coding-agent/src/prompts/compaction/rlm-compaction-system.md` — Rewrite for tool-based interaction
- `packages/coding-agent/test/rlm-compaction.test.ts` — Replace text-parsing tests with tool_use tests

### Dependencies
- Needs `@sinclair/typebox` for `Type.Object` / `Type.String` (already a dependency)
- Needs to import `Tool` type from `@oh-my-pi/pi-ai`

---

## 3. Replace File-Based IPC with stdin Channel

**Problem:** `llm_query()` uses file-based polling (Python writes JSON to a temp file, TypeScript polls every 50ms). This adds 50-100ms latency per roundtrip, requires temp file cleanup, and is fragile.

**Why not Pyodide/WASM?** Bun doesn't support JSPI (the modern replacement for Asyncify), and Pyodide doesn't even detect Bun as a valid runtime. MicroPython lacks too much stdlib. The WASM path is blocked for now.

**What we can do:** The IPython kernel already handles `input_request` / `input_reply` (kernel.ts lines 774-796). Python's `input()` is **blocking** — it suspends execution until the kernel sends an `input_reply`. This is exactly the synchronous IPC primitive we need.

Currently, the kernel sends an empty `input_reply` immediately. We extend it to route `llm_query` requests.

### Changes

**`kernel.ts`** — Add an `inputProvider` callback to `KernelExecuteOptions`:

```typescript
export interface KernelExecuteOptions {
    // ... existing fields ...
    /** Async callback to provide input when the kernel requests it via input().
     *  Receives the prompt string, returns the response string. */
    inputProvider?: (prompt: string) => Promise<string>;
}
```

In the `input_request` handler (line 774), use the provider:

```typescript
case "input_request": {
    const prompt = (response.content as { prompt?: string }).prompt ?? "";
    if (options?.inputProvider) {
        // Route to provider — blocks Python until we reply
        const value = await options.inputProvider(prompt);
        this.#sendMessage({
            channel: "stdin",
            header: { /* ... input_reply ... */ },
            parent_header: response.header,
            content: { value },
        });
    } else {
        stdinRequested = true;
        // ... existing empty-reply behavior ...
    }
    break;
}
```

**`executor.ts`** — Thread the `inputProvider` through `PythonExecutorOptions` → `executeWithKernel` → `kernel.execute`:

```typescript
export interface PythonExecutorOptions {
    // ... existing fields ...
    inputProvider?: (prompt: string) => Promise<string>;
}
```

In `executeWithKernel`, pass it to `kernel.execute`:
```typescript
const result = await kernel.execute(code, {
    signal: options?.signal,
    timeoutMs: options?.timeoutMs,
    onChunk: text => sink.push(text),
    onDisplay: output => void displayOutputs.push(output),
    inputProvider: options?.inputProvider,   // NEW
    allowStdin: !!options?.inputProvider,    // Enable stdin when we have a provider
});
```

**`rlm-compaction.ts`** — Replace file-based IPC with stdin IPC:

The setup code changes `llm_query()` to use `input()`:

```python
def llm_query(prompt, model=None):
    """Query a sub-LLM. Blocks until response is received."""
    request = json.dumps({"type": "llm_query", "prompt": str(prompt), "model": model})
    response_json = input(request)  # Blocks — TypeScript provides the reply
    response = json.loads(response_json)
    if "error" in response:
        raise RuntimeError(f"llm_query failed: {response['error']}")
    return response["response"]
```

The TypeScript side passes an `inputProvider` to `executePython`:

```typescript
const execResult = await executePython(block.code, {
    cwd,
    timeoutMs: CELL_TIMEOUT_SEC * 1000,
    signal,
    sessionId: kernelSessionId,
    kernelMode: "session",
    inputProvider: async (prompt: string) => {
        // Parse the request from the prompt string
        const request = JSON.parse(prompt);
        if (request.type === "llm_query") {
            totalSubCalls++;
            const response = await completeSimple(subQueryModel, ...);
            return JSON.stringify({ response: extractText(response) });
        }
        return JSON.stringify({ error: "Unknown request type" });
    },
});
```

**Remove:** `startLlmQueryHandler()`, `requestFile` / `responseFile` temp file logic, all `.ready` signal file handling. The entire concurrent polling infrastructure goes away.

**Remove from setup code:** `_RLM_REQUEST_FILE`, `_RLM_RESPONSE_FILE`, `time.sleep`, `os.path.exists` polling, `os.remove` cleanup.

### Files Modified
- `packages/coding-agent/src/ipy/kernel.ts` — Add `inputProvider` to `KernelExecuteOptions`, route `input_request` through it
- `packages/coding-agent/src/ipy/executor.ts` — Thread `inputProvider` through `PythonExecutorOptions` and `executeWithKernel`
- `packages/coding-agent/src/session/compaction/rlm-compaction.ts` — Replace file IPC with `inputProvider`, simplify `buildSetupCode`, remove `startLlmQueryHandler`

### Risks
- **`input()` prompt string has a size limit?** Need to verify. The Jupyter protocol sends the prompt as a string in the `input_request` message — should handle large JSON. But if the model passes a very large prompt to `llm_query()`, the JSON-encoded request could be huge. Mitigation: the prompt is *from* the model, not the context itself; it should be reasonable.
- **Concurrent `input()` calls:** If the code calls `llm_query()` in a loop or in `llm_query_batched()`, each call serializes through the same stdin channel. This is fine — they're sequential from Python's perspective.
- **`stdinRequested` flag:** Currently when `stdinRequested` is true, `executeWithKernel` returns with exitCode 1 (line 492-500). We need to suppress this when we have an `inputProvider` (the provider handles stdin, so `stdinRequested` should remain false).

---

## 4. True Recursive RLM Agents

**Problem:** Our `llm_query()` calls `completeSimple()` — a flat, single-shot LLM completion with no tools, no sandbox, no REPL. The sub-LLM can't write code, can't explore, can't delegate further. It's just a plain prompt-in, text-out call.

**What cantrip does:** `llm_query(query, subContext)` spawns a **complete nested RLM agent** at `depth + 1`, with its own sandbox, its own context, and its own tools. Depth-limited (default 2) with fallback to plain LLM at max depth.

### Changes

**`rlm-compaction.ts`** — Add a `depth` and `maxDepth` parameter to `RlmCompactionOptions`:

```typescript
export interface RlmCompactionOptions {
    // ... existing fields ...
    /** Current recursion depth (default: 0) */
    depth?: number;
    /** Maximum recursion depth (default: 2) */
    maxDepth?: number;
}
```

**Change `inputProvider` to spawn recursive agents:**

```typescript
inputProvider: async (prompt: string) => {
    const request = JSON.parse(prompt);
    if (request.type !== "llm_query") {
        return JSON.stringify({ error: "Unknown request type" });
    }
    totalSubCalls++;

    const currentDepth = options.depth ?? 0;
    const maxRecursionDepth = options.maxDepth ?? 2;

    if (currentDepth >= maxRecursionDepth) {
        // At max depth — fall back to plain LLM call with truncated context
        const truncatedPrompt = request.prompt.length > 10_000
            ? request.prompt.slice(0, 10_000) + "\n... [truncated]"
            : request.prompt;
        const response = await completeSimple(subQueryModel, {
            messages: [{ role: "user", content: [{ type: "text", text: truncatedPrompt }], timestamp: Date.now() }],
        }, { apiKey: subQueryApiKey, signal, maxTokens: 8192 });
        return JSON.stringify({ response: extractText(response) });
    }

    // Recursive: spawn a nested RLM agent
    const subContext = request.context ?? conversationText;
    // Create synthetic messages from the sub-context
    const subMessages: AgentMessage[] = [{
        role: "user",
        content: [{ type: "text", text: subContext }],
        timestamp: Date.now(),
    }];

    const childResult = await rlmCompact(subMessages, [], {
        model: subQueryModel,
        apiKey: subQueryApiKey,
        leafModel,
        leafApiKey,
        maxIterations: Math.min(maxIterations, 8),  // Limit child iterations
        signal,
        customInstructions: request.prompt,
        cwd,
        reserveTokens,
        depth: currentDepth + 1,
        maxDepth: maxRecursionDepth,
    });

    totalSubCalls += childResult.subLlmCalls;
    totalCodeBlocks += childResult.codeBlocksExecuted;
    return JSON.stringify({ response: childResult.summary });
}
```

**Update `buildSetupCode`** — Add `llm_query(prompt, sub_context=None)` parameter:

```python
def llm_query(prompt, sub_context=None, model=None):
    """Query a sub-LLM with optional sub-context.
    If sub_context is provided, the child agent explores it instead of the parent context."""
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
```

**Update `llm_query_batched`** to pass sub_context:
```python
def llm_query_batched(prompts, sub_contexts=None, model=None):
    results = []
    for i, p in enumerate(prompts):
        ctx = sub_contexts[i] if sub_contexts and i < len(sub_contexts) else None
        results.append(llm_query(p, sub_context=ctx, model=model))
    return results
```

**Kernel session isolation:** Each recursive level needs its own kernel session ID so variable state doesn't collide:
```typescript
const kernelSessionId = `rlm-compaction-d${currentDepth}-${sessionId}`;
```

**`rlm-compaction-system.md`** — Update to document recursive delegation:
- "Use `llm_query(prompt)` for semantic analysis of data too large to print."
- "Use `llm_query(prompt, sub_context=chunk)` to delegate analysis of a specific chunk to a child agent that can write its own code to explore it."
- "Child agents have their own REPL and can write code, search, and delegate further (up to depth limit)."

### Files Modified
- `packages/coding-agent/src/session/compaction/rlm-compaction.ts` — Add `depth`/`maxDepth` to options, recursive `rlmCompact` call in `inputProvider`, kernel session isolation
- `packages/coding-agent/src/prompts/compaction/rlm-compaction-system.md` — Document recursive delegation
- `packages/coding-agent/src/session/compaction/compaction.ts` — Thread `maxDepth` from settings if we add a setting for it

### Risks
- **Token cost explosion:** Recursive agents multiply LLM calls. A depth-2 agent with 3 `llm_query` calls that each do 3 more = 9+ sub-LLM calls. Mitigation: limit child iterations (e.g., max 8 at depth 1, max 4 at depth 2), and cap total sub-calls across all levels.
- **Kernel session limit:** Each depth level creates a new kernel session. With `MAX_KERNEL_SESSIONS = 4`, depth-3 recursion with parallel calls could hit the limit. Mitigation: share the parent's kernel (different session namespace for variables), or increase the limit for RLM sessions.
- **Latency:** Each recursive level adds full RLM loop latency. A depth-2 call might take 30-60s. This is acceptable for compaction (runs in the background).

---

## 5. Resource Cleanup

**Problem:** Temp files aren't cleaned on early return paths (FINAL found mid-iteration, abort). Kernel sessions for RLM aren't explicitly cleaned up. No token tracking across the driving model + sub-LLM calls.

**What cantrip does:** Child sandboxes are disposed in `finally` blocks. A shared `UsageTracker` accumulates tokens across all recursion levels.

### Changes

**A. Wrap the RLM loop in try/finally for cleanup:**

```typescript
export async function rlmCompact(...): Promise<RlmCompactionResult> {
    const kernelSessionId = ...;
    try {
        // ... entire RLM loop ...
    } finally {
        // Dispose the kernel session (free memory)
        await disposeRlmKernelSession(kernelSessionId);
        // Clean up any remaining temp files (if still using files for anything)
        await cleanupTempFiles(requestFile, responseFile);
    }
}
```

If we implement (3) (stdin IPC), there are no temp files to clean. But the kernel session should still be disposed — we don't want idle RLM kernels sitting around. Import `disposeKernelSession` from executor (or expose a cleanup function).

**B. Token tracking via `UsageTracker`:**

Add a usage accumulator to `RlmCompactionResult`:

```typescript
export interface RlmCompactionResult {
    summary: string;
    iterations: number;
    subLlmCalls: number;
    codeBlocksExecuted: number;
    /** Accumulated token usage across driving model + all sub-LLM calls */
    usage: {
        drivingModelInputTokens: number;
        drivingModelOutputTokens: number;
        subLlmInputTokens: number;
        subLlmOutputTokens: number;
        totalTokens: number;
        totalCost: number;
    };
}
```

Track usage from each `completeSimple` call:

```typescript
// After driving model call:
const response = await completeSimple(model, ...);
usage.drivingModelInputTokens += response.usage.input + response.usage.cacheRead;
usage.drivingModelOutputTokens += response.usage.output;

// After sub-LLM call (in inputProvider):
const subResponse = await completeSimple(subQueryModel, ...);
usage.subLlmInputTokens += subResponse.usage.input + subResponse.usage.cacheRead;
usage.subLlmOutputTokens += subResponse.usage.output;
```

For recursive agents, the child's `usage` is aggregated into the parent's `subLlm*` counters.

**C. Log usage on completion:**

```typescript
logger.debug("RLM compaction complete", {
    iterations: result.iterations,
    subLlmCalls: result.subLlmCalls,
    totalTokens: result.usage.totalTokens,
    totalCost: result.usage.totalCost,
});
```

**D. Kernel session cleanup:**

Export a `disposeKernelSessionById` from `executor.ts` (or add to existing exports). Call it in the `finally` block. This is especially important when recursive agents each create their own kernel sessions.

### Files Modified
- `packages/coding-agent/src/session/compaction/rlm-compaction.ts` — `try/finally`, usage tracking, kernel disposal
- `packages/coding-agent/src/ipy/executor.ts` — Export `disposeKernelSessionById` or similar
- `packages/coding-agent/src/session/compaction/compaction.ts` — Log usage from `RlmCompactionResult` if present

---

## Implementation Order

The changes have dependencies:

```
(3) stdin IPC  ← foundational, changes how code execution works
    ↓
(1) metadata-only output  ← changes what comes back from execution
    ↓
(2) tool_use forced sandbox  ← changes how the driving loop works
    ↓
(4) recursive agents  ← builds on (2) and (3) — uses tools + stdin IPC
    ↓
(5) cleanup  ← wraps everything
```

**Recommended order:** 3 → 1 → 2 → 4 → 5

Each step should include updating the tests (`rlm-compaction.test.ts`), running type-check (`npx tsc --noEmit`), and verifying existing tests pass.

---

## Estimated Scope

| # | Improvement | Files Changed | Complexity |
|---|-------------|--------------|------------|
| 3 | stdin IPC | kernel.ts, executor.ts, rlm-compaction.ts | Medium — extends existing stdin plumbing |
| 1 | Metadata-only output | rlm-compaction.ts, executor.ts, kernel.ts, prompt.md | Medium — needs stdout/result separation |
| 2 | Tool_use forced exit | rlm-compaction.ts, prompt.md, tests | Medium — restructures the driving loop |
| 4 | Recursive agents | rlm-compaction.ts, prompt.md, compaction.ts | High — recursive `rlmCompact`, kernel isolation |
| 5 | Resource cleanup | rlm-compaction.ts, executor.ts, compaction.ts | Low — try/finally + usage tracking |
