import { describe, expect, it } from "bun:test";
import {
	buildSetupCode,
	formatReplMetadata,
	getRlmSystemPrompt,
} from "@oh-my-pi/pi-coding-agent/session/compaction";

// ============================================================================
// buildSetupCode
// ============================================================================

describe("buildSetupCode", () => {
	it("should generate valid Python that defines context and helpers", () => {
		const code = buildSetupCode("Hello world");
		expect(code).toContain('context = """Hello world"""');
		expect(code).toContain("def llm_query(");
		expect(code).toContain("def llm_batch(");
		expect(code).toContain("def SHOW_VARS(");
		expect(code).toContain("context_length = len(context)");
	});

	it("should use input()-based IPC for llm_query", () => {
		const code = buildSetupCode("ctx");
		expect(code).toContain("input(request)");
		expect(code).toContain("json.dumps(");
		expect(code).toContain("json.loads(");
		// Should NOT contain file-based IPC remnants
		expect(code).not.toContain("time.sleep");
		expect(code).not.toContain(".ready");
	});

	it("should define submit_answer as a sandbox function (SIGNAL_FINAL pattern)", () => {
		const code = buildSetupCode("ctx");
		expect(code).toContain("def submit_answer(");
		expect(code).toContain("_SubmitSignal");
		expect(code).toContain("raise _SubmitSignal()");
		expect(code).toContain('"type": "submit_answer"');
	});

	it("should define llm_batch with cantrip API (dict tasks, max 50)", () => {
		const code = buildSetupCode("ctx");
		expect(code).toContain("def llm_batch(tasks):");
		expect(code).toContain('task.get("query")');
		expect(code).toContain('task.get("context")');
		expect(code).toContain("len(tasks) > 50");
	});

	it("should define llm_query_batched as backward-compatibility alias", () => {
		const code = buildSetupCode("ctx");
		expect(code).toContain("llm_query_batched = lambda");
	});

	it("should escape backslashes in context", () => {
		const code = buildSetupCode("path\\to\\file");
		expect(code).toContain("path\\\\to\\\\file");
	});

	it("should escape triple quotes in context", () => {
		const code = buildSetupCode('some """quoted""" text');
		expect(code).not.toContain('"""quoted"""');
	});

	it("should define SHOW_VARS helper", () => {
		const code = buildSetupCode("ctx");
		expect(code).toContain("def SHOW_VARS():");
		expect(code).toContain("globals().items()");
	});

	it("should print context metadata at the end", () => {
		const code = buildSetupCode("ctx");
		expect(code).toContain("print(f\"Context loaded:");
	});
});

// ============================================================================
// formatReplMetadata — matches cantrip's formatRlmMetadata
// ============================================================================

describe("formatReplMetadata", () => {
	it("should return [Result: undefined] for empty string", () => {
		expect(formatReplMetadata("")).toBe("[Result: undefined]");
	});

	it("should return [Result: undefined] for the string 'undefined'", () => {
		expect(formatReplMetadata("undefined")).toBe("[Result: undefined]");
	});

	it("should format output with char count and preview", () => {
		const result = formatReplMetadata("hello world");
		expect(result).toBe('[Result: 11 chars] "hello world"');
	});

	it("should replace newlines with spaces in the preview", () => {
		const result = formatReplMetadata("line1\nline2\nline3");
		expect(result).toContain("line1 line2 line3");
	});

	it("should truncate preview at 150 chars with ellipsis", () => {
		const longOutput = "x".repeat(200);
		const result = formatReplMetadata(longOutput);
		expect(result).toContain("200 chars");
		expect(result).toContain("...");
		expect(result).toContain(`"${"x".repeat(150)}..."`);
	});

	it("should not add ellipsis for output at exactly 150 chars", () => {
		const output = "y".repeat(150);
		const result = formatReplMetadata(output);
		expect(result).toBe(`[Result: 150 chars] "${"y".repeat(150)}"`);
	});

	it("should not trim whitespace (matches cantrip behavior)", () => {
		const result = formatReplMetadata("  hello  ");
		expect(result).toBe('[Result: 9 chars] "  hello  "');
	});
});

// ============================================================================
// getRlmSystemPrompt — dynamic prompt like cantrip's getRlmSystemPrompt
// ============================================================================

describe("getRlmSystemPrompt", () => {
	it("should include context metadata", () => {
		const prompt = getRlmSystemPrompt({
			contextLength: 50000,
			contextLines: 1200,
			contextPreview: "User: hello\nAssistant: hi",
			hasRecursion: true,
		});
		expect(prompt).toContain("50000 characters");
		expect(prompt).toContain("1200 lines");
		expect(prompt).toContain("User: hello");
	});

	it("should replace newlines with spaces in preview", () => {
		const prompt = getRlmSystemPrompt({
			contextLength: 100,
			contextLines: 5,
			contextPreview: "line1\nline2\nline3",
			hasRecursion: true,
		});
		expect(prompt).toContain("line1 line2 line3");
	});

	it("should include recursion-aware sub-LLM description when hasRecursion=true", () => {
		const prompt = getRlmSystemPrompt({
			contextLength: 100,
			contextLines: 5,
			contextPreview: "ctx",
			hasRecursion: true,
		});
		expect(prompt).toContain("recursively query sub-LLMs");
		expect(prompt).toContain("Spawns a recursive sub-agent");
	});

	it("should include leaf-node description when hasRecursion=false", () => {
		const prompt = getRlmSystemPrompt({
			contextLength: 100,
			contextLines: 5,
			contextPreview: "ctx",
			hasRecursion: false,
		});
		expect(prompt).toContain("truncated to ~10K chars");
		expect(prompt).not.toContain("recursively query sub-LLMs");
	});

	it("should document sandbox physics", () => {
		const prompt = getRlmSystemPrompt({
			contextLength: 100,
			contextLines: 5,
			contextPreview: "ctx",
			hasRecursion: true,
		});
		expect(prompt).toContain("PERSISTENCE");
		expect(prompt).toContain("METADATA-ONLY");
		expect(prompt).toContain("BLOCKING");
	});

	it("should document all host functions", () => {
		const prompt = getRlmSystemPrompt({
			contextLength: 100,
			contextLines: 5,
			contextPreview: "ctx",
			hasRecursion: true,
		});
		expect(prompt).toContain("llm_query(");
		expect(prompt).toContain("llm_batch(");
		expect(prompt).toContain("submit_answer(");
		expect(prompt).toContain("SHOW_VARS()");
	});

	it("should document submit_answer as the ONLY way to finish", () => {
		const prompt = getRlmSystemPrompt({
			contextLength: 100,
			contextLines: 5,
			contextPreview: "ctx",
			hasRecursion: true,
		});
		expect(prompt).toContain("ONLY way to finish");
	});

	it("should include the output format template", () => {
		const prompt = getRlmSystemPrompt({
			contextLength: 100,
			contextLines: 5,
			contextPreview: "ctx",
			hasRecursion: true,
		});
		expect(prompt).toContain("## Goal");
		expect(prompt).toContain("## Progress");
		expect(prompt).toContain("## Key Decisions");
		expect(prompt).toContain("## Next Steps");
	});
});
