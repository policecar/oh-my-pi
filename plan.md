# RLM Integration Plan for oh-my-pi

## Overview

Integrate the Recursive Language Model (RLM) pattern from deepfates/cantrip into oh-my-pi. Based on Zhang et al.'s RLM paper (arxiv:2512.24601), this keeps large data **outside** the LLM's prompt window in a QuickJS WASM sandbox. The LLM writes JavaScript code to explore data, delegate to sub-agents, and control a browser — all without bloating the context.

### Design Decisions (User-Confirmed)
- **Sandbox**: New QuickJS WASM tool (separate from existing Python tool)
- **Memory**: Replace existing summarization-based compaction with RLM sliding window
- **Browser**: Integrate Puppeteer functions into the sandbox
- **Use case**: Exploration/research

---

## Step 1: Add QuickJS WASM Dependencies

**Files:** `packages/coding-agent/package.json`

Add the following npm dependencies:

```
quickjs-emscripten-core ^0.29.0
@jitl/quickjs-ng-wasmfile-release-asyncify ^0.31.0
```

The **asyncify** variant is required because host functions like `llm_query()` and browser actions are async on the host side, but must appear synchronous inside the sandbox. The asyncify variant suspends the WASM execution while the host Promise resolves.

---

## Step 2: Create the RLM Tool Module

**New directory:** `packages/coding-agent/src/tools/rlm/`

### 2a. `sandbox.ts` — QuickJS Sandbox Lifecycle

Manages the WASM sandbox that persists across the session:

- `createRlmSandbox(context?: unknown)` → `{ ctx, dispose() }`
- Loads the asyncify WASM module (cached singleton)
- Creates a `QuickJSAsyncContext` with resource limits (256MB memory, 1MB stack, 30s timeout)
- Injects `context` as a global variable using a recursive `valueToHandle()` converter
- Provides `evaluate(code: string)` → `{ value: unknown, error?: string }` method
- Provides `updateContext(path: string, value: unknown)` to update sandbox globals (for memory management)
- Handles `dispose()` to free all WASM memory

### 2b. `host-functions.ts` — Core RLM Host Functions

Registers these blocking functions inside the sandbox:

| Function | Purpose |
|----------|---------|
| `console.log(...args)` | Print to host stderr for debugging |
| `submit_answer(value)` | Signal task completion (throws `SIGNAL_FINAL:` error caught by tool handler) |
| `llm_query(query, subContext?)` | Delegate to a sub-agent — spawns a nested RLM agent (if depth < maxDepth) or falls back to plain LLM completion |
| `llm_batch(tasks)` | Parallel sub-agent delegation (max 8 concurrent, max 50 total). Each task = `{ query, context? }` |

Implementation notes:
- Use `ctx.newAsyncifiedFunction()` for all async host functions
- `llm_query` at max depth falls back to `streamSimple()` with truncated context
- `submit_answer` throws a special error string that the tool handler catches to extract the final answer
- All functions registered on `ctx.global` and handles tracked for disposal

### 2c. `handle-table.ts` — Opaque Handle Table for Browser Objects

A `HandleTable` class that bridges complex host objects (Puppeteer Page, ElementHandle, etc.) across the WASM boundary:

- `store(obj, kind, desc)` → `{ __h: id, kind, desc }` (lightweight handle for sandbox)
- `resolve(handle)` → original host object
- `dispose()` — cleans up all stored references
- Handles are JSON-serializable objects with an integer ID, a `kind` string, and a human-readable `desc`

### 2d. `browser-bridge.ts` — Puppeteer Functions in Sandbox

Registers browser automation functions using the existing `BrowserTool` infrastructure:

**Navigation:** `goto(url)`, `reload()`, `goBack()`, `goForward()`
**Selectors (return handles):** `text(str)`, `button(str)`, `link(str)`, `textBox(label?)`, `image(alt?)`, `$(selector)`
**Actions:** `click(handle)`, `type(handle, text)`, `press(key)`, `scrollTo(handle)`, `focus(handle)`
**Observation:** `observe()` (returns accessibility tree text), `screenshot()` (returns base64 PNG), `title()`, `url()`, `evaluate(jsCode)` (runs JS in the browser page)
**Proximity selectors:** `near(handle)`, `above(handle)`, `below(handle)`, `toLeftOf(handle)`, `toRightOf(handle)`

Implementation:
- Each function registered via `ctx.newAsyncifiedFunction()`
- Selector functions create Puppeteer ElementHandles, store in HandleTable, return lightweight descriptor
- Action functions resolve handles from HandleTable, perform Puppeteer action
- `observe()` uses existing accessibility tree extraction from BrowserTool
- `screenshot()` captures, resizes, returns base64 data

### 2e. `prompt.ts` — RLM System Prompt Generation

Generates the RLM-specific system prompt section that teaches the LLM the sandbox API:

```
## RLM Sandbox

You have access to a JavaScript sandbox via the `rlm` tool. A variable called `context`
is pre-loaded with data. You can explore it by writing JavaScript code.

### Sandbox Physics
- All code is **blocking** (no async/await, no Promises)
- Use `var` for variables that persist across calls (not let/const)
- The sandbox is isolated — no network, no filesystem

### Available Functions
- `console.log(value)` — print for debugging (visible to you)
- `submit_answer(value)` — return your final answer
- `llm_query(query, subContext?)` — delegate analysis to a sub-agent
- `llm_batch([{query, context}, ...])` — parallel sub-agent delegation

### Browser Functions (when browser is active)
- `goto(url)` — navigate to URL
- `click(handle)`, `type(handle, text)` — interact with elements
- `observe()` — get accessibility tree of current page
- [full browser API docs...]

### Strategy
- Use code for **structured** tasks (filtering, counting, transforming)
- Use `llm_query()` for **semantic** tasks (summarizing, classifying, reasoning)
- Chunk large data and use `llm_batch()` for parallel processing

### Examples
[Worked examples of filtering, chunking, iterating, batch processing]
```

Content is dynamically generated based on:
- What context is loaded (type, size, preview via `analyzeContext()`)
- Whether browser is available
- Current memory state (if sliding window is active)

### 2f. `memory.ts` — Sliding Window Memory Management

Implements lossless conversation memory as a replacement for summarization-based compaction:

- `RlmMemoryManager` class:
  - `windowSize: number` — how many recent turns to keep in the active prompt (default: 3)
  - `manageMemory(messages: AgentMessage[], sandbox: RlmSandbox)` — called after each turn
  - Counts user turns; when count exceeds `windowSize`, moves the oldest complete turn (user + all assistant/tool responses) from the active message list into `context.conversation[]` in the sandbox
  - The sandbox's `context` becomes `{ data: <user data>, conversation: <older turns> }`
  - The agent can search older conversations by writing JS: `context.conversation.filter(t => t.content.includes("password"))`

Integration with existing compaction system:
- In `packages/coding-agent/src/session/compaction/compaction.ts`, add a new compaction strategy `"rlm"` alongside the existing `"summarize"` strategy
- When `compactionStrategy === "rlm"`, delegate to `RlmMemoryManager` instead of the summarization pipeline
- The setting is exposed as `compaction.strategy: "summarize" | "rlm"` in Settings

### 2g. `index.ts` — The RlmTool Class

The main tool implementation following `AgentTool<TParams, TDetails>`:

```typescript
const rlmSchema = Type.Object({
  code: Type.String({ description: "JavaScript code to execute in the sandbox" }),
});

class RlmTool implements AgentTool<typeof rlmSchema, RlmToolDetails> {
  name = "rlm";
  label = "RLM";
  description = "Execute JavaScript in the RLM sandbox to explore context data, ...";
  parameters = rlmSchema;
  concurrency = "exclusive" as const;  // Only one sandbox eval at a time

  execute(toolCallId, { code }, signal, onUpdate, context) {
    // 1. Evaluate code in persistent sandbox
    // 2. If SIGNAL_FINAL caught → extract answer, return as final result
    // 3. Otherwise return metadata-only: char count + 150-char preview
    //    (prevents prompt bloat — the LLM must store important state in sandbox vars)
    // 4. console.log output is included in the result
  }
}
```

Key behaviors:
- The tool returns **metadata only** (character count + truncated preview), not full output — this is the core RLM insight that prevents prompt bloat
- `submit_answer()` triggers a `TaskComplete` signal
- Console output is captured and included (for debugging visibility)
- The sandbox persists across calls within the same session

---

## Step 3: Register the Tool

**File:** `packages/coding-agent/src/tools/index.ts`

Add to `BUILTIN_TOOLS`:
```typescript
rlm: (s) => new RlmTool(s),
```

Add a setting to enable/disable:
```typescript
rlm.enabled: boolean  // default: true
```

---

## Step 4: Session-Level Sandbox Lifecycle

**Files:** `packages/coding-agent/src/session/agent-session.ts` (or wherever session state is managed)

The RLM sandbox must persist across the entire session (not created/destroyed per tool call):

1. On session start: create `RlmSandbox` instance, store on session
2. On tool call: the `RlmTool` accesses the session's sandbox
3. On context load (`--context <file>` or `/load` command): inject data into sandbox as `context`
4. On memory management: `RlmMemoryManager.manageMemory()` called after each agent turn
5. On session end/close: `sandbox.dispose()` frees WASM memory

The sandbox reference flows through `ToolSession` → `RlmTool`.

---

## Step 5: System Prompt Integration

**Files:** System prompt template (likely `packages/coding-agent/src/system-prompt.md` or similar)

When the RLM tool is enabled, append the RLM system prompt section (from `prompt.ts`) to the base system prompt. This section:
- Describes the sandbox API and physics
- Shows the context metadata (type, size, preview) without the actual data
- Includes worked examples
- Documents browser functions if browser is available

---

## Step 6: Context Loading UX

Add a way for users to load data into the RLM sandbox:

**Option A — CLI flag:** `omp --context data.json` loads the file into the sandbox at startup
**Option B — Slash command:** `/rlm-load <path>` loads a file into the sandbox mid-session
**Option C — Tool parameter:** The `rlm` tool accepts an optional `context_file` parameter

Recommendation: Implement **Option B** as a slash command, since it fits oh-my-pi's interactive model. Also support loading via the tool parameter for programmatic use.

---

## Step 7: Settings & Configuration

**File:** Settings schema

New settings:
```
rlm.enabled: boolean           # Enable/disable the RLM tool (default: true)
rlm.maxDepth: number           # Max recursive sub-agent depth (default: 1)
rlm.memoryWindowSize: number   # Sliding window size in turns (default: 3)
rlm.browserInSandbox: boolean  # Enable browser functions in sandbox (default: true)
compaction.strategy: "summarize" | "rlm"  # Memory management strategy (default: "summarize")
```

---

## File Summary

### New Files
| File | Purpose |
|------|---------|
| `packages/coding-agent/src/tools/rlm/index.ts` | RlmTool class (AgentTool implementation) |
| `packages/coding-agent/src/tools/rlm/sandbox.ts` | QuickJS sandbox lifecycle management |
| `packages/coding-agent/src/tools/rlm/host-functions.ts` | llm_query, submit_answer, console.log |
| `packages/coding-agent/src/tools/rlm/browser-bridge.ts` | Puppeteer functions in sandbox |
| `packages/coding-agent/src/tools/rlm/handle-table.ts` | Opaque handle table for browser objects |
| `packages/coding-agent/src/tools/rlm/prompt.ts` | RLM system prompt generation |
| `packages/coding-agent/src/tools/rlm/memory.ts` | Sliding window memory manager |

### Modified Files
| File | Change |
|------|--------|
| `packages/coding-agent/package.json` | Add quickjs-emscripten dependencies |
| `packages/coding-agent/src/tools/index.ts` | Register `rlm` in BUILTIN_TOOLS |
| `packages/coding-agent/src/session/compaction/compaction.ts` | Add `"rlm"` strategy branch |
| `packages/coding-agent/src/session/agent-session.ts` | Sandbox lifecycle on session |
| System prompt template | Append RLM section when enabled |
| Settings schema | Add rlm.* and compaction.strategy settings |

---

## Dependency Graph

```
RlmTool (index.ts)
├── sandbox.ts (QuickJS WASM)
│   ├── quickjs-emscripten-core
│   └── @jitl/quickjs-ng-wasmfile-release-asyncify
├── host-functions.ts
│   └── llm_query → packages/ai (streamSimple)
├── browser-bridge.ts
│   ├── handle-table.ts
│   └── packages/coding-agent/src/tools/browser.ts (Puppeteer)
├── prompt.ts
└── memory.ts
    └── sandbox.ts (updates context.conversation)
```

---

## Implementation Order

1. **sandbox.ts + host-functions.ts** — Get basic JS evaluation working with llm_query
2. **index.ts** — Wire up as AgentTool, register in BUILTIN_TOOLS
3. **prompt.ts** — Generate system prompt section with context metadata
4. **handle-table.ts + browser-bridge.ts** — Add browser functions
5. **memory.ts** — Implement sliding window, integrate with compaction system
6. **Settings + UX** — Configuration, context loading slash command
7. **Testing** — End-to-end tests with sample data, long conversations, browser automation
