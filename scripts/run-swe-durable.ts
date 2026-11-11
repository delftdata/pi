#!/usr/bin/env node
/**
 * Run SWE task(s) using Pi Durable (@earendil-works/pi-durable) with a local Ollama model.
 *
 * Supports:
 * - Single task execution (--repo and --problem)
 * - Multi-task suite execution from a JSON config file (--config <file>)
 * - Persistent SQLite state machine and task dependency graph
 * - Saving structured results into a timestamped file under results/
 * - Detailed token breakdown, test metrics, and plan graph
 */

import { execSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { parseArgs } from "node:util";
import { BACKGROUND_CONTEXT } from "@earendil-works/chord/context";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { createModels, createProvider } from "@earendil-works/pi-ai/models";
import type { ToolCall, ToolResultMessage } from "@earendil-works/pi-ai";
import { AssistantMessageEventStream } from "@earendil-works/pi-ai/utils/event-stream";
import { repairJson } from "@earendil-works/pi-ai/utils/json-parse";
import { NodeExecutionEnv } from "@earendil-works/pi-durable/env/node";
import {
	AssistantEntry,
	createRegistry,
	defineExtension,
	FootprintTracker,
	Harness,
	LiveDoc,
	section,
	ToolResultEntry,
	TrackingExecutionEnv,
} from "@earendil-works/pi-durable";
import { openNodeSqliteStorage } from "@earendil-works/pi-durable/storage/sqlite/node";
import { CodingTools } from "@earendil-works/pi-durable/tools";

export interface TaskDefinition {
	id: string;
	name?: string;
	workspace?: string;
	problem: string;
	testCmd: string;
	files?: Record<string, string>;
	setupCmd?: string;
}

export interface TasksConfigFile {
	tasks: TaskDefinition[];
}

function formatTimestamp(date: Date = new Date()): string {
	const pad = (n: number) => String(n).padStart(2, "0");
	const y = date.getFullYear();
	const m = pad(date.getMonth() + 1);
	const d = pad(date.getDate());
	const h = pad(date.getHours());
	const min = pad(date.getMinutes());
	const s = pad(date.getSeconds());
	return `${y}${m}${d}_${h}${min}${s}`;
}

function estimateTextTokens(text: string): number {
	return Math.ceil(text.length / 3.5);
}

function estimateMessagesTokens(messages: readonly any[]): number {
	let chars = 0;
	for (const msg of messages) {
		if (typeof msg.content === "string") {
			chars += msg.content.length;
		} else if (Array.isArray(msg.content)) {
			for (const block of msg.content) {
				if (block.type === "text") chars += (block.text || "").length;
				else if (block.type === "toolCall") chars += JSON.stringify(block).length;
			}
		}
	}
	return Math.ceil(chars / 3.5);
}

function extractToolCalls(text: string): ToolCall[] {
	const toolCalls: ToolCall[] = [];
	let pos = 0;
	while (pos < text.length) {
		const nameIdx = text.indexOf('"name"', pos);
		if (nameIdx === -1) break;
		const openBrace = text.lastIndexOf("{", nameIdx);
		if (openBrace === -1) {
			pos = nameIdx + 6;
			continue;
		}

		let depth = 0;
		let inString = false;
		let escape = false;
		let endBrace = -1;
		for (let i = openBrace; i < text.length; i++) {
			const ch = text[i];
			if (escape) {
				escape = false;
				continue;
			}
			if (ch === "\\") {
				escape = true;
				continue;
			}
			if (ch === '"') {
				inString = !inString;
				continue;
			}
			if (!inString) {
				if (ch === "{") depth++;
				else if (ch === "}") {
					depth--;
					if (depth === 0) {
						endBrace = i;
						break;
					}
				}
			}
		}

		if (endBrace !== -1) {
			const jsonStr = text.slice(openBrace, endBrace + 1);
			try {
				const repaired = repairJson(jsonStr);
				const obj = JSON.parse(repaired);
				if (typeof obj.name === "string" && obj.arguments && typeof obj.arguments === "object") {
					toolCalls.push({
						type: "toolCall",
						id: `call_${Math.random().toString(36).slice(2, 9)}`,
						name: obj.name,
						arguments: obj.arguments,
					});
				}
			} catch {
				// Continue search
			}
			pos = endBrace + 1;
		} else {
			pos = nameIdx + 6;
		}
	}
	return toolCalls.slice(0, 1);
}

function wrapStream(
	rawStream: AssistantMessageEventStream,
	modelId: string,
	streamContext: { messages: readonly any[] },
): AssistantMessageEventStream {
	const outStream = new AssistantMessageEventStream();
	(async () => {
		try {
			let accumulatedText = "";
			let foundToolCall: ToolCall | null = null;

			for await (const event of rawStream) {
				if (event.type === "text_delta") {
					accumulatedText += event.delta;
					if (!foundToolCall) {
						const extracted = extractToolCalls(accumulatedText);
						if (extracted.length > 0) {
							foundToolCall = extracted[0];
							console.log(`\n>>> [Pi Durable Executing Tool: ${foundToolCall.name}]`, JSON.stringify(foundToolCall.arguments));
							const promptTokens = estimateMessagesTokens(streamContext.messages);
							const outputTokens = estimateTextTokens(accumulatedText);
							outStream.push({
								type: "done",
								reason: "toolUse",
								message: {
									role: "assistant",
									content: [foundToolCall],
									api: "openai-completions",
									provider: "ollama",
									model: modelId,
									usage: {
										input: promptTokens,
										output: outputTokens,
										cacheRead: 0,
										cacheWrite: 0,
										reasoning: 0,
										totalTokens: promptTokens + outputTokens,
										cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0, totalFromProvider: false },
									},
									stopReason: "toolUse",
									timestamp: Date.now(),
									durationMs: 0,
								},
							});
							return;
						}
					}
				}

				if (event.type === "done") {
					if (!foundToolCall) {
						const msg = event.message;
						if (msg.content && !msg.content.some((c) => c.type === "toolCall")) {
							const text = msg.content
								.filter((c) => c.type === "text")
								.map((c) => c.text)
								.join("")
								.trim();

							const extracted = extractToolCalls(text);
							if (extracted.length > 0) {
								msg.content = extracted;
								msg.stopReason = "toolUse";
								for (const tc of extracted) {
									console.log(`\n>>> [Pi Durable Executing Tool: ${tc.name}]`, JSON.stringify(tc.arguments));
								}
							}
						}
						if (!msg.usage || msg.usage.totalTokens === 0) {
							const promptTokens = estimateMessagesTokens(streamContext.messages);
							const outputTokens = estimateTextTokens(accumulatedText);
							msg.usage = {
								input: promptTokens,
								output: outputTokens,
								cacheRead: 0,
								cacheWrite: 0,
								reasoning: 0,
								totalTokens: promptTokens + outputTokens,
								cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0, totalFromProvider: false },
							};
						}
						outStream.push({ type: "done", reason: (msg.stopReason as any) ?? "stop", message: msg });
					}
				} else if (event.type !== "error") {
					if (!foundToolCall) {
						outStream.push(event);
					}
				}
			}
		} catch (err: any) {
			const promptTokens = estimateMessagesTokens(streamContext.messages);
			outStream.push({
				type: "error",
				reason: "error",
				error: {
					role: "assistant",
					content: [],
					api: "openai-completions",
					provider: "ollama",
					model: modelId,
					usage: {
						input: promptTokens,
						output: 0,
						cacheRead: 0,
						cacheWrite: 0,
						reasoning: 0,
						totalTokens: promptTokens,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0, totalFromProvider: false },
					},
					stopReason: "error",
					errorMessage: err?.message ?? String(err),
					timestamp: Date.now(),
					durationMs: 0,
				},
			});
		}
	})();
	return outStream;
}

function parseTestOutput(output: string, exitCode: number) {
	const ranMatch = output.match(/Ran (\d+) tests/i);
	const failuresMatch = output.match(/failures=(\d+)/i);
	const errorsMatch = output.match(/errors=(\d+)/i);

	if (ranMatch) {
		const total = parseInt(ranMatch[1], 10);
		const failures = failuresMatch ? parseInt(failuresMatch[1], 10) : 0;
		const errors = errorsMatch ? parseInt(errorsMatch[1], 10) : 0;
		const passed = Math.max(0, total - failures - errors);
		return {
			total,
			passed,
			failed: failures + errors,
			failures,
			errors,
			status: exitCode === 0 && (failures + errors === 0) ? "passed" : "failed",
		};
	}

	const pytestMatch = output.match(/(?:(\d+) failed)?[,\s]*(?:(\d+) passed)/i);
	if (pytestMatch) {
		const failed = pytestMatch[1] ? parseInt(pytestMatch[1], 10) : 0;
		const passed = pytestMatch[2] ? parseInt(pytestMatch[2], 10) : 0;
		return {
			total: passed + failed,
			passed,
			failed,
			failures: failed,
			errors: 0,
			status: exitCode === 0 && failed === 0 ? "passed" : "failed",
		};
	}

	return {
		total: exitCode === 0 ? 1 : 1,
		passed: exitCode === 0 ? 1 : 0,
		failed: exitCode === 0 ? 0 : 1,
		failures: exitCode === 0 ? 0 : 1,
		errors: 0,
		status: exitCode === 0 ? "passed" : "failed",
	};
}

interface CommonOptions {
	model: string;
	ollamaUrl: string;
	resultsDir: string;
	dbName: string;
	trackFootprint: boolean;
}

async function runTask(task: TaskDefinition, options: CommonOptions) {
	const taskStartTime = Date.now();
	const launchTime = new Date(taskStartTime).toISOString();
	const targetRepo = task.workspace
		? resolve(task.workspace)
		: resolve("/tmp/swe_tasks", task.id);

	console.log(`\n==========================================================`);
	console.log(` Running Task: [${task.id}] ${task.name ?? ""}`);
	console.log(` Workspace:    ${targetRepo}`);
	console.log(` Model:        ${options.model} (${options.ollamaUrl})`);
	console.log(` Footprint:    ${options.trackFootprint ? "ENABLED" : "DISABLED"}`);
	console.log(`==========================================================`);

	// 1. Setup workspace if files or setupCmd provided
	if (task.files) {
		if (existsSync(targetRepo)) {
			execSync(`rm -rf "${targetRepo}"`);
		}
		mkdirSync(targetRepo, { recursive: true });
		execSync("git init -q", { cwd: targetRepo });
		execSync('git config user.name "SWE Tester"', { cwd: targetRepo });
		execSync('git config user.email "swe@test.local"', { cwd: targetRepo });
		for (const [filename, content] of Object.entries(task.files)) {
			await writeFile(resolve(targetRepo, filename), content, "utf-8");
		}
		execSync("git add . && git commit -q -m 'Initial commit with buggy code'", { cwd: targetRepo });
	} else if (!existsSync(targetRepo)) {
		mkdirSync(targetRepo, { recursive: true });
	}

	if (task.setupCmd) {
		console.log(`Running task setup: ${task.setupCmd}`);
		execSync(task.setupCmd, { cwd: targetRepo, stdio: "inherit" });
	}

	// 2. Run baseline test (expected to fail)
	console.log(`\nVerifying baseline test before fix (expected to fail)...`);
	try {
		execSync(`${task.testCmd} 2>&1`, { cwd: targetRepo, stdio: "pipe" });
		console.log(`Warning: Tests passed before fix.`);
	} catch {
		console.log(`>> Baseline test failed as expected.`);
	}

	// 3. Configure local Ollama model provider with tool adapter
	const baseApi = openAICompletionsApi();
	const ollama = createProvider<"openai-completions">({
		id: "ollama",
		name: "Ollama",
		baseUrl: options.ollamaUrl,
		auth: { apiKey: { name: "Ollama", resolve: async () => ({ auth: { apiKey: "ollama" } }) } },
		models: [
			{
				id: options.model,
				name: options.model,
				api: "openai-completions",
				provider: "ollama",
				baseUrl: options.ollamaUrl,
				input: ["text"],
				reasoning: false,
				contextWindow: 32768,
				maxTokens: 8192,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			},
		],
		api: {
			stream(model, streamContext, streamOptions) {
				return wrapStream(baseApi.stream(model, streamContext, streamOptions), options.model, streamContext);
			},
			streamSimple(model, streamContext, streamOptions) {
				return wrapStream(baseApi.streamSimple(model, streamContext, streamOptions), options.model, streamContext);
			},
		},
	});

	const models = createModels();
	models.setProvider(ollama);

	const registry = createRegistry();
	registry.install(CodingTools);
	registry.install(
		defineExtension({
			name: "swe-instructions",
			sections: [
				section(
					"swe-instructions",
					() =>
						"You are an expert autonomous software engineer solving an issue in a repository.\n" +
						"Important Rules:\n" +
						"- Execute ONLY ONE tool per turn. Do not output multiple tool calls or repeated commands.\n" +
						"- When calling a tool, output ONLY the JSON object format: {\"name\": \"<tool_name>\", \"arguments\": {<args>}}, then STOP.\n" +
						"- Wait for the tool result before taking the next action.\n" +
						"Workflow:\n" +
						"1. Call `read` on the target source file to inspect its exact code and signatures.\n" +
						"2. Fix the bug by calling `write` or `edit` with the correct implementation.\n" +
						"3. Use `bash` to run unit tests and verify the fix works.\n" +
						"4. If any test fails, analyze the test error, apply another `edit` or `write` to fix it, and re-run unit tests.\n" +
						"5. Once all tests pass, provide a concise final summary without calling any more tools.",
					{ tag: false },
				),
			],
		}),
	);

	const dbPath = resolve(targetRepo, options.dbName);
	const context = BACKGROUND_CONTEXT;
	const storage = await openNodeSqliteStorage(dbPath);

	const footprintTracker = options.trackFootprint ? new FootprintTracker(task.id) : null;
	const harness = await Harness.open(
		storage,
		{
			models,
			registry,
			env: () => {
				const base = new NodeExecutionEnv({ cwd: targetRepo });
				return footprintTracker ? new TrackingExecutionEnv(base, footprintTracker) : base;
			},
		},
		context,
	);

	const root = await harness.root(context, {
		agent: {
			model: { provider: "ollama", modelId: options.model },
			cwd: targetRepo,
		},
	});

	const liveWatch = await harness.watchDoc(LiveDoc, root.id, context);
	let printed = "";
	if (liveWatch) {
		liveWatch.start(async (call) => {
			if (call.kind !== "value") return;
			const block = call.value?.generation?.message?.content.find((c) => c.type === "text");
			if (block && block.type === "text" && block.text.length > printed.length) {
				process.stdout.write(block.text.slice(printed.length));
				printed = block.text;
			}
		});
	}

	harness.resume();

	console.log("\n--- Submitting Problem Statement ---");
	console.log(task.problem.trim());
	console.log("------------------------------------\n");

	const submission = await root.submit({ type: "input", content: task.problem }, context);
	const settled = await submission.wait(context);

	if (liveWatch) await liveWatch.stop();

	console.log("\n--- Task Settled ---");
	console.log("Status:", settled.status);

	if (settled.status === "done" && settled.type === "input") {
		const answer = await root.commit((tx) => tx.entry(AssistantEntry, settled.answer), context);
		const msg = answer?.model?.[0];
		if (msg && msg.role === "assistant") {
			const text = msg.content.map((c) => (c.type === "text" ? c.text : "")).join("");
			if (text && text !== printed) {
				console.log("Final response:\n", text);
			}
		}
	} else if (settled.status === "unanswered") {
		console.error("Unanswered reason:", settled.reason, settled.detail);
	}

	const entries = await root.entries({}, 50, undefined, context);
	console.log("\n--- Execution History ---");
	for (const entry of entries.items) {
		if (ToolResultEntry.is(entry)) {
			const result = entry.model?.[0] as ToolResultMessage | undefined;
			if (result && result.role === "toolResult") {
				const text = result.content.map((c) => (c.type === "text" ? c.text : "")).join("");
				console.log(`[Tool Result: ${result.toolName}] -> ${text.trim().slice(0, 150)}...`);
			}
		}
	}

	let diffPatch = "";
	try {
		diffPatch = execSync("git diff", { cwd: targetRepo, encoding: "utf-8" });
		console.log("\n--- Generated Git Diff Patch ---");
		if (diffPatch.trim()) {
			console.log(diffPatch);
		} else {
			console.log("(No changes modified in git working directory)");
		}
	} catch (e) {
		console.error("Could not run git diff:", e);
	}

	await harness.close(context);
	const footprint = footprintTracker ? footprintTracker.finalize() : null;

	// 4. Test verification
	let testResult = {
		command: task.testCmd,
		total: 0,
		passed: 0,
		failed: 0,
		failures: 0,
		errors: 0,
		status: "unknown",
		output: "",
	};

	try {
		const output = execSync(`${task.testCmd} 2>&1`, { cwd: targetRepo, encoding: "utf-8" });
		testResult = {
			command: task.testCmd,
			...parseTestOutput(output, 0),
			output: output.trim(),
		};
	} catch (err: any) {
		const combinedOutput = `${err.stdout || ""}\n${err.stderr || ""}`.trim();
		testResult = {
			command: task.testCmd,
			...parseTestOutput(combinedOutput, err.status ?? 1),
			output: combinedOutput,
		};
	}

	// 5. Extract plan graph and token breakdown from SQLite
	const db = new DatabaseSync(dbPath, { readOnly: true });
	const rawTasks = db.prepare("SELECT id, kind, status, record FROM tasks ORDER BY id ASC").all() as Array<{
		id: number;
		kind: string;
		status: string;
		record: string;
	}>;
	const rawEntries = db.prepare("SELECT id, record FROM entries ORDER BY id ASC").all() as Array<{
		id: number;
		record: string;
	}>;

	const entriesMap = new Map<number, any>();
	for (const e of rawEntries) {
		try {
			entriesMap.set(e.id, JSON.parse(e.record));
		} catch {}
	}

	const subtasks: any[] = [];
	let previousTaskId: number | null = null;

	for (const t of rawTasks) {
		const rec = JSON.parse(t.record);
		const cleanKind = t.kind.replace(/^"|"$/g, "").replace(/^pi\./, "");
		const durationMs = rec.startedAt && rec.endedAt ? rec.endedAt - rec.startedAt : undefined;

		let toolName: string | undefined;
		let toolArgs: any | undefined;
		let description = "";
		const dependencies: number[] = [];

		let subtaskTokens = {
			total: 0,
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			reasoning: 0,
		};

		if (cleanKind === "tool") {
			if (rec.owner) dependencies.push(rec.owner);
			const assistantEntry = rec.input?.assistant ? entriesMap.get(rec.input.assistant) : undefined;
			if (assistantEntry?.model?.[0]?.content) {
				const call = assistantEntry.model[0].content.find(
					(c: any) => c.type === "toolCall" && c.id === rec.input.callId,
				);
				if (call) {
					toolName = call.name;
					toolArgs = call.arguments;
				}
			}
			description = `Execute tool: ${toolName ?? "tool"}${toolArgs?.path ? ` on ${toolArgs.path}` : toolArgs?.command ? `: ${toolArgs.command}` : ""}`;
		} else if (cleanKind === "generation") {
			if (previousTaskId !== null) {
				dependencies.push(previousTaskId);
			}
			const outcomeEntryId = rec.state?.outcome?.result?.entryId;
			let entry = outcomeEntryId ? entriesMap.get(outcomeEntryId) : undefined;
			if (!entry) {
				for (const e of entriesMap.values()) {
					if (e.byTaskId === t.id && e.kind === "pi.assistant") {
						entry = e;
						break;
					}
				}
			}
			const calls = entry?.model?.[0]?.content?.filter((c: any) => c.type === "toolCall");
			if (calls && calls.length > 0) {
				description = `Model turn: formulated tool call to ${calls[0].name}`;
			} else {
				description = `Model turn: evaluated state and finalized response`;
			}
			if (entry?.model?.[0]?.usage) {
				const u = entry.model[0].usage;
				subtaskTokens = {
					total:
						u.totalTokens ||
						(u.input || 0) + (u.output || 0) + (u.cacheRead || 0) + (u.cacheWrite || 0),
					input: u.input || 0,
					output: u.output || 0,
					cacheRead: u.cacheRead || 0,
					cacheWrite: u.cacheWrite || 0,
					reasoning: u.reasoning || 0,
				};
			}
		}

		const subtaskFootprint = (footprintTracker && rec.startedAt && rec.endedAt)
			? footprintTracker.getFootprintForInterval(rec.startedAt, rec.endedAt, t.id)
			: undefined;

		subtasks.push({
			id: t.id,
			kind: cleanKind,
			description,
			...(toolName ? { tool: toolName } : {}),
			...(toolArgs ? { arguments: toolArgs } : {}),
			dependencies,
			status: rec.state?.outcome?.status || t.status,
			startedAt: rec.startedAt ? new Date(rec.startedAt).toISOString() : undefined,
			endedAt: rec.endedAt ? new Date(rec.endedAt).toISOString() : undefined,
			durationMs,
			tokens: subtaskTokens,
			...(subtaskFootprint ? { footprint: subtaskFootprint } : {}),
		});

		previousTaskId = t.id;
	}

	const tokenTotals = {
		total: 0,
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		reasoning: 0,
	};

	for (const entry of entriesMap.values()) {
		if (entry.kind === "pi.assistant" && Array.isArray(entry.model)) {
			for (const msg of entry.model) {
				if (msg.usage) {
					tokenTotals.input += msg.usage.input || 0;
					tokenTotals.output += msg.usage.output || 0;
					tokenTotals.cacheRead += msg.usage.cacheRead || 0;
					tokenTotals.cacheWrite += msg.usage.cacheWrite || 0;
					tokenTotals.reasoning += msg.usage.reasoning || 0;
					tokenTotals.total +=
						msg.usage.totalTokens ||
						((msg.usage.input || 0) +
							(msg.usage.output || 0) +
							(msg.usage.cacheRead || 0) +
							(msg.usage.cacheWrite || 0));
				}
			}
		}
	}

	const taskEndTime = Date.now();
	const totalDurationMs = taskEndTime - taskStartTime;
	const timestampStr = formatTimestamp(new Date(taskStartTime));
	const resultsFilename = `${task.id}_${timestampStr}.json`;
	const resultsPath = resolve(options.resultsDir, resultsFilename);

	const taskResult = {
		taskId: task.id,
		taskName: task.name,
		resultsFilename,
		launchTime,
		endTime: new Date(taskEndTime).toISOString(),
		duration: {
			totalMs: totalDurationMs,
			formatted: `${(totalDurationMs / 1000).toFixed(2)}s`,
		},
		tracking: {
			enabled: options.trackFootprint,
			overheadMs: footprint?.overhead?.totalMs ?? 0,
			overheadFormatted: footprint?.overhead?.formatted ?? "0.00ms",
		},
		model: {
			id: options.model,
			provider: "ollama",
			url: options.ollamaUrl,
		},
		task: {
			workspace: targetRepo,
			problemStatement: task.problem.trim(),
			status: settled.status,
		},
		tokens: tokenTotals,
		tests: testResult,
		plan: {
			summary: `Executed ${subtasks.length} subtasks (${subtasks.filter((s) => s.kind === "generation").length} generations, ${subtasks.filter((s) => s.kind === "tool").length} tool calls)`,
			subtasks,
		},
		footprint: footprint ?? null,
		gitDiff: diffPatch.trim(),
	};

	mkdirSync(options.resultsDir, { recursive: true });
	await writeFile(resultsPath, JSON.stringify(taskResult, null, 2), "utf-8");

	console.log(`\n----------------------------------------------------------`);
	console.log(`Task [${task.id}] Completed:`);
	console.log(`Saved:          ${resultsPath}`);
	console.log(`Duration:       ${taskResult.duration.formatted}`);
	console.log(`Tokens Total:   ${taskResult.tokens.total} (Input: ${taskResult.tokens.input}, Output: ${taskResult.tokens.output}, CacheRead: ${taskResult.tokens.cacheRead})`);
	console.log(`Tests:          ${taskResult.tests.passed}/${taskResult.tests.total} passed (Status: ${taskResult.tests.status})`);
	console.log(`Plan Subtasks:  ${taskResult.plan.subtasks.length}`);
	if (footprint) {
		console.log(`Footprint:      Read Set:  ${footprint.summary.totalReads} resources (${footprint.summary.uniqueResourcesRead.join(", ")})`);
		console.log(`                Write Set: ${footprint.summary.totalWrites} resources (${footprint.summary.uniqueResourcesWritten.join(", ")})`);
		console.log(`                Exec Set:  ${footprint.summary.totalExecutions} commands`);
		console.log(`                Overhead:  ${footprint.overhead?.formatted ?? "0.00ms"}`);
	} else {
		console.log(`Footprint:      DISABLED (no tracking overhead added)`);
	}
	console.log(`----------------------------------------------------------\n`);

	return taskResult;
}

// CLI entry point
const { values } = parseArgs({
	options: {
		config: { type: "string", short: "c" },
		task: { type: "string", short: "t" },
		repo: { type: "string", short: "r" },
		problem: { type: "string", short: "p" },
		problemFile: { type: "string", short: "f" },
		model: { type: "string", short: "m", default: "qwen2.5-coder:7b" },
		ollamaUrl: { type: "string", default: "http://localhost:11434/v1" },
		db: { type: "string", default: "swe_task_state.db" },
		resultsDir: { type: "string", default: "results" },
		testCmd: { type: "string", default: "python3 test_slugify.py" },
		footprint: { type: "string", default: "true" },
		noFootprint: { type: "boolean" },
		help: { type: "boolean", short: "h" },
	},
});

if (values.help || (!values.config && !values.repo)) {
	console.log(`
Usage:
  # Run tasks from a config file (single or multi-task suite):
  node --import <resolver> scripts/run-swe-durable.ts --config swe-tasks.json [options]

  # Run a specific task from config:
  node --import <resolver> scripts/run-swe-durable.ts --config swe-tasks.json --task palindrome-validator

  # Run on an existing repository directly:
  node --import <resolver> scripts/run-swe-durable.ts --repo <path> --problem <text> [options]

Options:
  -c, --config <file>        Path to JSON config file with task definitions
  -t, --task <id>            Filter to run only this task ID from config
  -r, --repo <path>          Target git repository directory (for standalone run)
  -p, --problem <text>       Problem statement / issue description
  -f, --problemFile <file>   Path to file containing problem statement
  -m, --model <id>           Ollama model ID (default: qwen2.5-coder:7b)
      --ollamaUrl <url>      Ollama API URL (default: http://localhost:11434/v1)
      --db <file>            SQLite database file for durable state (default: swe_task_state.db)
      --resultsDir <dir>     Directory to save timestamped result JSONs (default: results)
      --testCmd <cmd>        Command to run unit tests (default: python3 test_slugify.py)
      --footprint <bool>     Enable state footprint tracking (default: true)
      --noFootprint          Disable state footprint tracking to benchmark baseline
  -h, --help                 Show this help message
`);
	process.exit(values.help ? 0 : 1);
}

const trackFootprint = !values.noFootprint && values.footprint !== "false";

const commonOptions: CommonOptions = {
	model: values.model!,
	ollamaUrl: values.ollamaUrl!,
	resultsDir: resolve(process.cwd(), values.resultsDir!),
	dbName: values.db!,
	trackFootprint,
};

const suiteStartTime = Date.now();
const suiteTimestamp = formatTimestamp(new Date(suiteStartTime));
const taskResults: Array<Awaited<ReturnType<typeof runTask>>> = [];

if (values.config) {
	const configPath = resolve(values.config);
	if (!existsSync(configPath)) {
		console.error(`Config file not found: ${configPath}`);
		process.exit(1);
	}
	const content = await readFile(configPath, "utf-8");
	const parsed = JSON.parse(content);
	let tasks: TaskDefinition[] = Array.isArray(parsed) ? parsed : parsed.tasks;
	if (!tasks || !Array.isArray(tasks) || tasks.length === 0) {
		console.error(`No tasks found in config file: ${configPath}`);
		process.exit(1);
	}

	if (values.task) {
		tasks = tasks.filter((t) => t.id === values.task);
		if (tasks.length === 0) {
			console.error(`Task with id '${values.task}' not found in config.`);
			process.exit(1);
		}
	}

	console.log(`Found ${tasks.length} task(s) to execute from ${basename(configPath)}.`);
	for (const task of tasks) {
		const res = await runTask(task, commonOptions);
		taskResults.push(res);
	}

	// Write suite summary if multiple tasks were executed
	if (taskResults.length > 1) {
		const suiteDurationMs = Date.now() - suiteStartTime;
		const passedCount = taskResults.filter((r) => r.tests.status === "passed").length;
		const totalTokens = taskResults.reduce((acc, r) => acc + r.tokens.total, 0);

		// Cross-task footprint conflict detection (arXiv:2410.03140)
		const conflicts: any[] = [];
		if (commonOptions.trackFootprint) {
			for (let i = 0; i < taskResults.length; i++) {
				for (let j = i + 1; j < taskResults.length; j++) {
					if (taskResults[i].footprint && taskResults[j].footprint) {
						const report = FootprintTracker.detectConflicts(taskResults[i].footprint!, taskResults[j].footprint!);
						if (report.hasConflicts) {
							conflicts.push(report);
						}
					}
				}
			}
		}

		const totalOverheadMs = taskResults.reduce((acc, r) => acc + (r.tracking?.overheadMs ?? 0), 0);

		const suiteSummary = {
			suiteTimestamp,
			startTime: new Date(suiteStartTime).toISOString(),
			endTime: new Date().toISOString(),
			duration: {
				totalMs: suiteDurationMs,
				formatted: `${(suiteDurationMs / 1000).toFixed(2)}s`,
			},
			tracking: {
				enabled: commonOptions.trackFootprint,
				totalOverheadMs: Number(totalOverheadMs.toFixed(3)),
				totalOverheadFormatted: `${totalOverheadMs.toFixed(2)}ms`,
			},
			model: {
				id: commonOptions.model,
				url: commonOptions.ollamaUrl,
			},
			totalTasks: taskResults.length,
			passedTasks: passedCount,
			failedTasks: taskResults.length - passedCount,
			totalTokens,
			conflicts: {
				detected: conflicts.length > 0,
				totalConflicts: conflicts.length,
				details: conflicts,
			},
			tasks: taskResults.map((r) => ({
				id: r.taskId,
				name: r.taskName,
				status: r.tests.status,
				testsPassed: r.tests.passed,
				testsTotal: r.tests.total,
				duration: r.duration.formatted,
				tokens: r.tokens.total,
				footprintSummary: r.footprint?.summary ?? null,
				resultsFile: r.resultsFilename,
			})),
		};

		const suiteSummaryPath = resolve(commonOptions.resultsDir, `suite_${suiteTimestamp}.json`);
		await writeFile(suiteSummaryPath, JSON.stringify(suiteSummary, null, 2), "utf-8");
		console.log(`\n==========================================================`);
		console.log(` Suite Execution Summary: ${passedCount}/${taskResults.length} tasks passed`);
		console.log(` Tracking:                ${commonOptions.trackFootprint ? `ENABLED (Overhead: ${totalOverheadMs.toFixed(2)}ms)` : "DISABLED"}`);
		console.log(` Suite Results File:      ${suiteSummaryPath}`);
		console.log(` Total Tokens:            ${totalTokens}`);
		console.log(` Total Duration:          ${suiteSummary.duration.formatted}`);
		console.log(`==========================================================\n`);
	}
} else {
	const targetRepo = resolve(values.repo!);
	let problemStatement = values.problem ?? "";
	if (values.problemFile) {
		problemStatement = await readFile(resolve(values.problemFile), "utf-8");
	}
	if (!problemStatement.trim()) {
		console.error("Please provide a problem statement via --problem or --problemFile");
		process.exit(1);
	}

	const standaloneTask: TaskDefinition = {
		id: basename(targetRepo),
		problem: problemStatement,
		workspace: targetRepo,
		testCmd: values.testCmd!,
	};

	await runTask(standaloneTask, commonOptions);
}
