import { describe, expect, it } from "bun:test";
import {
	buildSetupCode,
	formatReplMetadata,
} from "@oh-my-pi/pi-coding-agent/session/compaction";

// ============================================================================
// buildSetupCode
// ============================================================================

describe("buildSetupCode", () => {
	it("should generate valid Python that defines context and helpers", () => {
		const code = buildSetupCode("Hello world");
		expect(code).toContain('context = """Hello world"""');
		expect(code).toContain("def llm_query(");
		expect(code).toContain("def llm_query_batched(");
		expect(code).toContain("def SHOW_VARS(");
		expect(code).toContain("context_length = len(context)");
	});

	it("should use input()-based IPC for llm_query", () => {
		const code = buildSetupCode("ctx");
		// Should use input() for stdin-based IPC, not file polling
		expect(code).toContain("input(request)");
		expect(code).toContain("json.dumps(");
		expect(code).toContain("json.loads(");
		// Should NOT contain file-based IPC remnants
		expect(code).not.toContain("time.sleep");
		expect(code).not.toContain(".ready");
	});

	it("should escape backslashes in context", () => {
		const code = buildSetupCode("path\\to\\file");
		expect(code).toContain("path\\\\to\\\\file");
	});

	it("should escape triple quotes in context", () => {
		const code = buildSetupCode('some """quoted""" text');
		// Triple quotes should be escaped so the Python string literal is valid
		expect(code).not.toContain('"""quoted"""');
	});

	it("should define llm_query_batched that calls llm_query in a loop", () => {
		const code = buildSetupCode("ctx");
		expect(code).toContain("def llm_query_batched(");
		expect(code).toContain("llm_query(p,");
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
// formatReplMetadata
// ============================================================================

describe("formatReplMetadata", () => {
	it("should return [no output] for empty string", () => {
		expect(formatReplMetadata("")).toBe("[no output]");
	});

	it("should return [no output] for whitespace-only string", () => {
		expect(formatReplMetadata("   \n  ")).toBe("[no output]");
	});

	it("should format short output with char and line counts", () => {
		const result = formatReplMetadata("hello");
		expect(result).toContain("5 chars");
		expect(result).toContain("1 lines");
		expect(result).toContain('"hello"');
	});

	it("should format multi-line output", () => {
		const result = formatReplMetadata("line1\nline2\nline3");
		expect(result).toContain("3 lines");
		expect(result).toContain("line1\\nline2\\nline3");
	});

	it("should truncate preview at 200 chars with ellipsis", () => {
		const longOutput = "x".repeat(300);
		const result = formatReplMetadata(longOutput);
		expect(result).toContain("300 chars");
		expect(result).toContain("...");
		// Preview should be 200 chars of x's
		expect(result).toContain(`"${"x".repeat(200)}..."`);
	});

	it("should not add ellipsis for output exactly at 200 chars", () => {
		const output = "y".repeat(200);
		const result = formatReplMetadata(output);
		expect(result).toContain("200 chars");
		expect(result).not.toContain("...");
	});

	it("should trim leading/trailing whitespace before processing", () => {
		const result = formatReplMetadata("  hello  \n");
		expect(result).toContain('"hello"');
	});
});
