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
	type FootprintComparisonReport,
	Harness,
	LiveDoc,
	type PredictedFootprint,
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

function sanitizeModelJson(text: string): string {
	// 1. Convert Python triple quotes: content: '''...''' or """..."""
	let cleaned = text.replace(/:\s*('''|""")([\s\S]*?)\1/g, (_, __, content) => {
		return ": " + JSON.stringify(content);
	});
	// 2. Convert single-quoted JSON keys: 'name': 'write'
	cleaned = cleaned.replace(/'([a-zA-Z0-9_]+)'\s*:/g, '"$1":');
	return cleaned;
}

function extractToolCalls(text: string): ToolCall[] {
	const sanitized = sanitizeModelJson(text);
	const toolCalls: ToolCall[] = [];
	let pos = 0;
	while (pos < sanitized.length) {
		const nameIdx = sanitized.indexOf('"name"', pos);
		if (nameIdx === -1) break;
		const openBrace = sanitized.lastIndexOf("{", nameIdx);
		if (openBrace === -1) {
			pos = nameIdx + 6;
			continue;
		}

		let depth = 0;
		let inString = false;
		let escape = false;
		let endBrace = -1;
		for (let i = openBrace; i < sanitized.length; i++) {
			const ch = sanitized[i];
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
			const jsonStr = sanitized.slice(openBrace, endBrace + 1);
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

function runVerificationTests(testCmd: string, targetRepo: string) {
	try {
		const output = execSync(`${testCmd} 2>&1`, { cwd: targetRepo, encoding: "utf-8" });
		return {
			command: testCmd,
			...parseTestOutput(output, 0),
			output: output.trim(),
		};
	} catch (err: any) {
		const combinedOutput = `${err.stdout || ""}\n${err.stderr || ""}`.trim();
		return {
			command: testCmd,
			...parseTestOutput(combinedOutput, err.status ?? 1),
			output: combinedOutput,
		};
	}
}

async function predictFootprint(
	task: TaskDefinition,
	targetRepo: string,
	options: CommonOptions,
): Promise<PredictedFootprint> {
	let repoFiles: string[] = [];
	try {
		const out = execSync("git ls-files", { cwd: targetRepo, encoding: "utf-8" });
		repoFiles = out.split("\n").map((f) => f.trim()).filter(Boolean);
	} catch {
		repoFiles = Object.keys(task.files ?? {});
	}

	const systemPrompt =
		"You are an expert autonomous software engineering planner.\n" +
		"Analyze the workspace files, the problem statement, and the test command.\n" +
		"Predict the state footprint the agent will need to touch:\n" +
		"1. `read`: list of file paths the agent will read.\n" +
		"2. `write`: list of file paths the agent will edit or create.\n" +
		"3. `execute`: list of commands the agent will run in terminal (e.g. test command).\n" +
		"4. `reasoning`: brief explanation of the prediction.\n\n" +
		"Respond ONLY with a JSON object conforming to:\n" +
		"{\n" +
		'  "read": ["filename"],\n' +
		'  "write": ["filename"],\n' +
		'  "execute": ["command"],\n' +
		'  "reasoning": "string"\n' +
		"}";

	const userPrompt =
		`Workspace Files:\n${repoFiles.map((f) => `- ${f}`).join("\n") || "(empty)"}\n\n` +
		`Problem Statement:\n${task.problem.trim()}\n\n` +
		`Test Command:\n${task.testCmd}\n`;

	try {
		const res = await fetch(`${options.ollamaUrl}/chat/completions`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				model: options.model,
				messages: [
					{ role: "system", content: systemPrompt },
					{ role: "user", content: userPrompt },
				],
				temperature: 0.1,
				stream: false,
			}),
		});

		if (!res.ok) {
			throw new Error(`HTTP ${res.status}: ${await res.text()}`);
		}

		const data = (await res.json()) as any;
		const rawContent = data.choices?.[0]?.message?.content ?? "";
		return parsePredictedFootprint(rawContent, repoFiles, task.testCmd);
	} catch (err) {
		console.warn(`Planner prediction fallback due to: ${err instanceof Error ? err.message : String(err)}`);
		return fallbackPredictedFootprint(repoFiles, task.testCmd);
	}
}

function parsePredictedFootprint(raw: string, repoFiles: string[], testCmd: string): PredictedFootprint {
	try {
		const cleaned = repairJson(raw);
		const jsonMatch = cleaned.match(/\{[\s\S]*\}/);
		if (jsonMatch) {
			const parsed = JSON.parse(jsonMatch[0]);
			const rawRead = parsed.read ?? parsed.files_to_read ?? parsed.readFiles ?? parsed.reads;
			const rawWrite = parsed.write ?? parsed.files_to_write ?? parsed.writeFiles ?? parsed.writes;
			const rawExec = parsed.execute ?? parsed.commands_to_execute ?? parsed.executeCommands ?? parsed.commands;

			const read = Array.isArray(rawRead) ? rawRead.map(String) : [];
			const write = Array.isArray(rawWrite) ? rawWrite.map(String) : [];
			const execute = Array.isArray(rawExec) ? rawExec.map(String) : [];

			return {
				read: Array.from(new Set(read)),
				write: Array.from(new Set(write)),
				execute: Array.from(new Set(execute)),
				reasoning: typeof parsed.reasoning === "string" ? parsed.reasoning : undefined,
			};
		}
	} catch {}

	return fallbackPredictedFootprint(repoFiles, testCmd);
}

function fallbackPredictedFootprint(repoFiles: string[], testCmd: string): PredictedFootprint {
	const nonTestFiles = repoFiles.filter((f) => !f.startsWith("test_") && !f.includes(".test."));
	return {
		read: repoFiles,
		write: nonTestFiles.length > 0 ? nonTestFiles : repoFiles,
		execute: [testCmd],
		reasoning: "Fallback heuristic footprint prediction based on workspace files.",
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
		await writeFile(resolve(targetRepo, ".gitignore"), `${options.dbName}\n${options.dbName}-*\n`, "utf-8");
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

	// 3. Planner Predicted Footprint
	console.log(`\n--- Planner Formulating Predicted Footprint ---`);
	const predictedFootprint = await predictFootprint(task, targetRepo, options);
	console.log(`Predicted Read Set:    ${predictedFootprint.read.join(", ") || "(none)"}`);
	console.log(`Predicted Write Set:   ${predictedFootprint.write.join(", ") || "(none)"}`);
	console.log(`Predicted Execute Set: ${predictedFootprint.execute.join(", ") || "(none)"}`);
	if (predictedFootprint.reasoning) {
		console.log(`Reasoning:             ${predictedFootprint.reasoning}`);
	}
	console.log(`-----------------------------------------------\n`);

	// 4. Configure local Ollama model provider with tool adapter
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
						"2. Fix the bug by calling `write` or `edit` with the correct implementation. (If `edit` fails to match exact whitespace, call `write` to overwrite the target file with the complete corrected code).\n" +
						"3. Use `bash` to run unit tests and verify the fix works.\n" +
						"4. If any test fails, analyze the test error, apply another `write` or `edit` to fix it, and re-run unit tests.\n" +
						"5. Do NOT stop until all unit tests pass. Once all tests pass, provide a concise final summary without calling any more tools.",
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

	const maxAttempts = 5;
	const maxStallAttempts = 2;
	let currentPrompt = task.problem;
	let attempt = 0;
	let bestPassedTests = -1;
	let stallCount = 0;
	let isSatisfied = false;
	let fallbackTriggered = false;
	let lastSettledStatus = "unknown";
	const iterationHistory: Array<{
		attempt: number;
		testsPassed: number;
		testsTotal: number;
		testStatus: string;
		status: string;
		diffPresent: boolean;
	}> = [];

	while (attempt < maxAttempts && !isSatisfied) {
		attempt++;
		if (attempt > 1) {
			console.log(`\n==========================================================`);
			console.log(` Dynamic Adaptive Attempt ${attempt}/${maxAttempts}`);
			console.log(`==========================================================`);
		}

		const submission = await root.submit({ type: "input", content: currentPrompt }, context);
		const settled = await submission.wait(context);
		lastSettledStatus = settled.status;

		if (settled.status === "done" && settled.type === "input") {
			const answer = await root.commit((tx) => tx.entry(AssistantEntry, settled.answer), context);
			const msg = answer?.model?.[0];
			if (msg && msg.role === "assistant") {
				const text = msg.content.map((c) => (c.type === "text" ? c.text : "")).join("");
				if (text && text !== printed) {
					console.log("\nResponse:\n", text);
				}
			}
		} else if (settled.status === "unanswered") {
			console.error("Unanswered reason:", settled.reason, settled.detail);
		}

		// Run verification tests
		const currentTestResult = runVerificationTests(task.testCmd, targetRepo);
		console.log(
			`\n[Attempt ${attempt} Result] Tests: ${currentTestResult.passed}/${currentTestResult.total} passed (Status: ${currentTestResult.status})`,
		);

		let currentDiff = "";
		try {
			currentDiff = execSync("git diff", { cwd: targetRepo, encoding: "utf-8" }).trim();
		} catch {}

		iterationHistory.push({
			attempt,
			testsPassed: currentTestResult.passed,
			testsTotal: currentTestResult.total,
			testStatus: currentTestResult.status,
			status: settled.status,
			diffPresent: currentDiff.length > 0,
		});

		if (currentTestResult.status === "passed") {
			console.log(`\n>> All verification tests passed! Dynamic transaction committed.`);
			isSatisfied = true;
			break;
		}

		// Check progress
		if (currentTestResult.passed > bestPassedTests) {
			console.log(
				`>> Progress detected: tests passed improved from ${bestPassedTests === -1 ? 0 : bestPassedTests} to ${currentTestResult.passed}/${currentTestResult.total}.`,
			);
			bestPassedTests = currentTestResult.passed;
			stallCount = 0;
		} else {
			stallCount++;
			console.log(`>> No improvement (stall count: ${stallCount}/${maxStallAttempts}).`);
		}

		if (stallCount >= maxStallAttempts) {
			console.warn(`>> Stagnation detected: agent made no progress in ${stallCount} consecutive attempts. Triggering fallback.`);
			fallbackTriggered = true;
			// Revert workspace to clean baseline
			try {
				execSync(`git checkout . && git clean -fd -e "${options.dbName}*"`, { cwd: targetRepo, stdio: "pipe" });
				console.log(`>> Fallback: Reverted workspace to baseline clean state.`);
			} catch {}
			break;
		}

		if (attempt < maxAttempts) {
			if (currentDiff.length === 0) {
				currentPrompt =
					"No modifications have been made to the repository yet, or your previous edit failed to match. " +
					"Please inspect the target source file and call `write` with the complete corrected file contents, " +
					`then run '${task.testCmd}' using bash to verify all tests pass.`;
			} else {
				currentPrompt =
					`Unit tests did not pass (${currentTestResult.passed}/${currentTestResult.total} passed).\n` +
					`Test output:\n${currentTestResult.output}\n\n` +
					"Please analyze the exact failure above, apply another fix using `write` or `edit`, " +
					`and run '${task.testCmd}' using bash to verify.`;
			}
			console.log(`\n--- Feeding Adaptive Follow-Up to Agent ---`);
			console.log(currentPrompt.slice(0, 200) + "...\n");
		}
	}

	if (liveWatch) await liveWatch.stop();

	console.log("\n--- Task Settled ---");
	console.log("Status:", isSatisfied ? "passed" : fallbackTriggered ? "stalled_fallback" : lastSettledStatus);

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

	const footprintComparison = footprint
		? FootprintTracker.compareFootprints(predictedFootprint, footprint, targetRepo)
		: undefined;

	if (footprintComparison) {
		console.log(`\n==========================================================`);
		console.log(` Footprint Comparison (Planner Prediction vs Actual Execution)`);
		console.log(`==========================================================`);
		console.log(
			`Reads:    Precision: ${(footprintComparison.read.precision * 100).toFixed(0)}%, Recall: ${(footprintComparison.read.recall * 100).toFixed(0)}%`,
		);
		console.log(`          Predicted: [${footprintComparison.read.predicted.join(", ") || "(none)"}]`);
		console.log(`          Actual:    [${footprintComparison.read.actual.join(", ") || "(none)"}]`);
		console.log(`          Matched:   [${footprintComparison.read.truePositives.join(", ") || "(none)"}]`);
		if (footprintComparison.read.falsePositives.length > 0) {
			console.log(`          Predicted but unread: [${footprintComparison.read.falsePositives.join(", ")}]`);
		}
		if (footprintComparison.read.falseNegatives.length > 0) {
			console.log(`          Unpredicted reads:    [${footprintComparison.read.falseNegatives.join(", ")}]`);
		}

		console.log(
			`Writes:   Precision: ${(footprintComparison.write.precision * 100).toFixed(0)}%, Recall: ${(footprintComparison.write.recall * 100).toFixed(0)}%`,
		);
		console.log(`          Predicted: [${footprintComparison.write.predicted.join(", ") || "(none)"}]`);
		console.log(`          Actual:    [${footprintComparison.write.actual.join(", ") || "(none)"}]`);
		console.log(`          Matched:   [${footprintComparison.write.truePositives.join(", ") || "(none)"}]`);
		if (footprintComparison.write.falsePositives.length > 0) {
			console.log(`          Predicted but unwritten: [${footprintComparison.write.falsePositives.join(", ")}]`);
		}
		if (footprintComparison.write.falseNegatives.length > 0) {
			console.log(`          Unpredicted writes:      [${footprintComparison.write.falseNegatives.join(", ")}]`);
		}

		console.log(
			`Executes: Precision: ${(footprintComparison.execute.precision * 100).toFixed(0)}%, Recall: ${(footprintComparison.execute.recall * 100).toFixed(0)}%`,
		);
		console.log(`          Predicted: [${footprintComparison.execute.predicted.join(", ") || "(none)"}]`);
		console.log(`          Actual:    [${footprintComparison.execute.actual.join(", ") || "(none)"}]`);
		console.log(`          Matched:   [${footprintComparison.execute.truePositives.join(", ") || "(none)"}]`);

		console.log(
			`Summary:  Exact Match: ${footprintComparison.summary.exactMatch ? "YES" : "NO"} | Overall F1: ${(footprintComparison.summary.overallF1 * 100).toFixed(0)}% (Precision: ${(footprintComparison.summary.overallPrecision * 100).toFixed(0)}%, Recall: ${(footprintComparison.summary.overallRecall * 100).toFixed(0)}%)`,
		);
		console.log(`==========================================================\n`);
	}

	// 4. Test verification
	const testResult = runVerificationTests(task.testCmd, targetRepo);

	// 5. Extract plan graph and token breakdown from SQLite
	let rawTasks: Array<{ id: number; kind: string; status: string; record: string }> = [];
	let rawEntries: Array<{ id: number; record: string }> = [];
	if (existsSync(dbPath)) {
		const db = new DatabaseSync(dbPath, { readOnly: true });
		rawTasks = db.prepare("SELECT id, kind, status, record FROM tasks ORDER BY id ASC").all() as any;
		rawEntries = db.prepare("SELECT id, record FROM entries ORDER BY id ASC").all() as any;
	}

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
			status: isSatisfied ? "passed" : fallbackTriggered ? "stalled_fallback" : lastSettledStatus,
		},
		adaptiveLoop: {
			totalAttempts: attempt,
			maxAttempts,
			isSatisfied,
			fallbackTriggered,
			iterations: iterationHistory,
		},
		tokens: tokenTotals,
		tests: testResult,
		plan: {
			summary: `Executed ${subtasks.length} subtasks (${subtasks.filter((s) => s.kind === "generation").length} generations, ${subtasks.filter((s) => s.kind === "tool").length} tool calls)`,
			subtasks,
		},
		predictedFootprint,
		footprintComparison: footprintComparison ?? null,
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
	if (footprintComparison) {
		console.log(
			`Prediction:     Overall F1: ${(footprintComparison.summary.overallF1 * 100).toFixed(0)}% (Read Recall: ${(footprintComparison.read.recall * 100).toFixed(0)}%, Write Recall: ${(footprintComparison.write.recall * 100).toFixed(0)}%, Exact Match: ${footprintComparison.summary.exactMatch})`,
		);
	}
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
				predictionAccuracy: r.footprintComparison
					? {
							f1: r.footprintComparison.summary.overallF1,
							precision: r.footprintComparison.summary.overallPrecision,
							recall: r.footprintComparison.summary.overallRecall,
							exactMatch: r.footprintComparison.summary.exactMatch,
						}
					: null,
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
