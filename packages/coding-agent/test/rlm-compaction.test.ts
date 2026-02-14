import { describe, expect, it } from "bun:test";
import {
	buildSetupCode,
	findCodeBlocks,
	findFinalAnswer,
	findFinalVar,
} from "@oh-my-pi/pi-coding-agent/session/compaction";

// ============================================================================
// findCodeBlocks
// ============================================================================

describe("findCodeBlocks", () => {
	it("should extract a single repl code block", () => {
		const text = `Here is some code:

\`\`\`repl
print("hello")
\`\`\`

Done.`;
		const blocks = findCodeBlocks(text);
		expect(blocks).toHaveLength(1);
		expect(blocks[0].code).toBe('print("hello")');
	});

	it("should extract multiple repl code blocks", () => {
		const text = `First:

\`\`\`repl
x = 1
\`\`\`

Second:

\`\`\`repl
y = 2
print(x + y)
\`\`\``;
		const blocks = findCodeBlocks(text);
		expect(blocks).toHaveLength(2);
		expect(blocks[0].code).toBe("x = 1");
		expect(blocks[1].code).toBe("y = 2\nprint(x + y)");
	});

	it("should ignore non-repl code blocks", () => {
		const text = `\`\`\`python
print("ignored")
\`\`\`

\`\`\`repl
print("included")
\`\`\``;
		const blocks = findCodeBlocks(text);
		expect(blocks).toHaveLength(1);
		expect(blocks[0].code).toBe('print("included")');
	});

	it("should return empty array for no code blocks", () => {
		const text = "Just some text with no code blocks.";
		expect(findCodeBlocks(text)).toHaveLength(0);
	});

	it("should skip empty code blocks", () => {
		const text = `\`\`\`repl
\`\`\`

\`\`\`repl
print("real code")
\`\`\``;
		const blocks = findCodeBlocks(text);
		expect(blocks).toHaveLength(1);
		expect(blocks[0].code).toBe('print("real code")');
	});
});

// ============================================================================
// findFinalAnswer
// ============================================================================

describe("findFinalAnswer", () => {
	it("should extract FINAL() answer", () => {
		const text = `FINAL(## Summary
This is the summary.)`;
		const answer = findFinalAnswer(text);
		expect(answer).toBe("## Summary\nThis is the summary.");
	});

	it("should handle FINAL() with whitespace", () => {
		const text = `  FINAL(  The answer  )  `;
		const answer = findFinalAnswer(text);
		expect(answer).toBe("The answer");
	});

	it("should return null when no FINAL()", () => {
		const text = "Some text without a FINAL call.";
		expect(findFinalAnswer(text)).toBeNull();
	});

	it("should handle multiline FINAL content", () => {
		const text = `FINAL(## Goal
Build a widget.

## Progress
### Done
- [x] Created the widget.

## Next Steps
1. Test it.)`;
		const answer = findFinalAnswer(text);
		expect(answer).toContain("## Goal");
		expect(answer).toContain("## Next Steps");
	});
});

// ============================================================================
// findFinalVar
// ============================================================================

describe("findFinalVar", () => {
	it("should extract FINAL_VAR() variable name", () => {
		const text = "FINAL_VAR(summary)";
		expect(findFinalVar(text)).toBe("summary");
	});

	it("should handle FINAL_VAR with whitespace", () => {
		const text = "  FINAL_VAR(my_summary)  ";
		expect(findFinalVar(text)).toBe("my_summary");
	});

	it("should return null when no FINAL_VAR()", () => {
		const text = "FINAL(some text)";
		expect(findFinalVar(text)).toBeNull();
	});

	it("should match word characters only", () => {
		const text = "FINAL_VAR(summary_v2)";
		expect(findFinalVar(text)).toBe("summary_v2");
	});
});

// ============================================================================
// buildSetupCode
// ============================================================================

describe("buildSetupCode", () => {
	it("should generate valid Python that defines context and helpers", () => {
		const code = buildSetupCode("Hello world", "/tmp/req.json", "/tmp/resp.json");
		expect(code).toContain('context = """Hello world"""');
		expect(code).toContain("def llm_query(");
		expect(code).toContain("def llm_query_batched(");
		expect(code).toContain("def SHOW_VARS(");
		expect(code).toContain("context_length = len(context)");
	});

	it("should escape backslashes in context", () => {
		const code = buildSetupCode("path\\to\\file", "/tmp/req.json", "/tmp/resp.json");
		expect(code).toContain("path\\\\to\\\\file");
	});

	it("should escape triple quotes in context", () => {
		const code = buildSetupCode('some """quoted""" text', "/tmp/req.json", "/tmp/resp.json");
		// Triple quotes should be escaped
		expect(code).not.toContain('"""quoted"""');
	});

	it("should embed request/response file paths", () => {
		const code = buildSetupCode("ctx", "/tmp/my-req.json", "/tmp/my-resp.json");
		expect(code).toContain("/tmp/my-req.json");
		expect(code).toContain("/tmp/my-resp.json");
	});

	it("should include file-based IPC polling in llm_query", () => {
		const code = buildSetupCode("ctx", "/tmp/req.json", "/tmp/resp.json");
		expect(code).toContain(".ready");
		expect(code).toContain("time.sleep(0.1)");
		expect(code).toContain("TimeoutError");
	});
});
