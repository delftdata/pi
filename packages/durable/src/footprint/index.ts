/**
 * State Footprint Tracker for Autonomous Agents (inspired by arXiv:2410.03140 / 2610.03140).
 *
 * Tracks the complete set of states, files, processes, and environment variables an agent
 * observes (Read Set) and mutates (Write Set) during its execution, at both task and subtask levels.
 * Provides conflict detection (Read-Write and Write-Write) across concurrent agent tasks.
 */

import { createHash } from "node:crypto";
import type { Context } from "@earendil-works/chord/context";
import type {
	BinaryReader,
	DirReader,
	ExecutionEnv,
	ExecutionError,
	FileChange,
	FileError,
	FileInfo,
	FileListing,
	FileWatcher,
	ListFilesOptions,
	Result,
	ShellExecOptions,
	ShellExecResult,
	TextLineReader,
	WatchTarget,
} from "../env/index.ts";

export interface ReadFootprintItem {
	kind: "file" | "env" | "state";
	resource: string;
	path?: string;
	hash?: string;
	bytes?: number;
	lineRange?: { start: number; end: number };
	subtaskId?: number | string;
	timestamp: string;
	timestampMs: number;
	metadata?: Record<string, unknown>;
}

export interface WriteFootprintItem {
	kind: "file" | "state";
	resource: string;
	path?: string;
	operation: "create" | "update" | "delete" | "edit";
	beforeHash?: string | null;
	afterHash?: string;
	bytesWritten?: number;
	diffPatch?: string;
	subtaskId?: number | string;
	timestamp: string;
	timestampMs: number;
	metadata?: Record<string, unknown>;
}

export interface ExecuteFootprintItem {
	kind: "process";
	resource: string;
	command: string;
	cwd?: string;
	exitCode?: number;
	durationMs?: number;
	subtaskId?: number | string;
	timestamp: string;
	timestampMs: number;
	metadata?: Record<string, unknown>;
}

export interface FootprintSummary {
	totalReads: number;
	totalWrites: number;
	totalExecutions: number;
	uniqueResourcesRead: string[];
	uniqueResourcesWritten: string[];
	uniqueCommandsExecuted: string[];
}

export interface StateFootprint {
	taskId?: string;
	subtaskId?: number | string;
	startedAt: string;
	endedAt?: string;
	readSet: ReadFootprintItem[];
	writeSet: WriteFootprintItem[];
	executeSet: ExecuteFootprintItem[];
	summary: FootprintSummary;
	overhead?: {
		totalMs: number;
		formatted: string;
	};
}

export interface ConflictReport {
	hasConflicts: boolean;
	readWriteConflicts: Array<{ taskA: string; taskB: string; resource: string }>;
	writeWriteConflicts: Array<{ taskA: string; taskB: string; resource: string }>;
}

export function computeHash(content: string | Uint8Array): string {
	return createHash("sha256").update(content).digest("hex").slice(0, 16);
}

export class FootprintTracker {
	readonly taskId?: string;
	readonly startedAt: string;
	private endedAt?: string;
	private activeSubtaskId?: number | string;
	private overheadMs: number = 0;

	private readonly reads: ReadFootprintItem[] = [];
	private readonly writes: WriteFootprintItem[] = [];
	private readonly executes: ExecuteFootprintItem[] = [];

	constructor(taskId?: string) {
		this.taskId = taskId;
		this.startedAt = new Date().toISOString();
	}

	setSubtask(subtaskId?: number | string): void {
		this.activeSubtaskId = subtaskId;
	}

	addOverhead(ms: number): void {
		this.overheadMs += ms;
	}

	getOverheadMs(): number {
		return this.overheadMs;
	}

	recordRead(
		path: string,
		content: string | Uint8Array,
		options: {
			lineRange?: { start: number; end: number };
			subtaskId?: number | string;
			metadata?: Record<string, unknown>;
		} = {},
	): ReadFootprintItem {
		const start = performance.now();
		const canonical = `file:${path}`;
		const hash = computeHash(content);
		const bytes = typeof content === "string" ? Buffer.byteLength(content, "utf-8") : content.byteLength;
		const now = Date.now();
		const item: ReadFootprintItem = {
			kind: "file",
			resource: canonical,
			path,
			hash,
			bytes,
			lineRange: options.lineRange,
			subtaskId: options.subtaskId ?? this.activeSubtaskId,
			timestamp: new Date(now).toISOString(),
			timestampMs: now,
			metadata: options.metadata,
		};
		this.reads.push(item);
		this.overheadMs += performance.now() - start;
		return item;
	}

	recordWrite(
		path: string,
		beforeContent: string | Uint8Array | null,
		afterContent: string | Uint8Array,
		options: { diffPatch?: string; subtaskId?: number | string; metadata?: Record<string, unknown> } = {},
	): WriteFootprintItem {
		const start = performance.now();
		const canonical = `file:${path}`;
		const beforeHash = beforeContent !== null ? computeHash(beforeContent) : null;
		const afterHash = computeHash(afterContent);
		const bytesWritten =
			typeof afterContent === "string" ? Buffer.byteLength(afterContent, "utf-8") : afterContent.byteLength;
		const operation = beforeContent === null ? "create" : options.diffPatch ? "edit" : "update";
		const now = Date.now();

		const item: WriteFootprintItem = {
			kind: "file",
			resource: canonical,
			path,
			operation,
			beforeHash,
			afterHash,
			bytesWritten,
			diffPatch: options.diffPatch,
			subtaskId: options.subtaskId ?? this.activeSubtaskId,
			timestamp: new Date(now).toISOString(),
			timestampMs: now,
			metadata: options.metadata,
		};
		this.writes.push(item);
		this.overheadMs += performance.now() - start;
		return item;
	}

	recordDelete(
		path: string,
		beforeContent: string | Uint8Array | null = null,
		options: { subtaskId?: number | string } = {},
	): WriteFootprintItem {
		const start = performance.now();
		const canonical = `file:${path}`;
		const beforeHash = beforeContent !== null ? computeHash(beforeContent) : null;
		const now = Date.now();
		const item: WriteFootprintItem = {
			kind: "file",
			resource: canonical,
			path,
			operation: "delete",
			beforeHash,
			afterHash: "",
			bytesWritten: 0,
			subtaskId: options.subtaskId ?? this.activeSubtaskId,
			timestamp: new Date(now).toISOString(),
			timestampMs: now,
		};
		this.writes.push(item);
		this.overheadMs += performance.now() - start;
		return item;
	}

	recordExecute(
		command: string,
		options: {
			cwd?: string;
			exitCode?: number;
			durationMs?: number;
			subtaskId?: number | string;
			metadata?: Record<string, unknown>;
		} = {},
	): ExecuteFootprintItem {
		const start = performance.now();
		const canonical = `proc:${command}`;
		const now = Date.now();
		const item: ExecuteFootprintItem = {
			kind: "process",
			resource: canonical,
			command,
			cwd: options.cwd,
			exitCode: options.exitCode,
			durationMs: options.durationMs,
			subtaskId: options.subtaskId ?? this.activeSubtaskId,
			timestamp: new Date(now).toISOString(),
			timestampMs: now,
			metadata: options.metadata,
		};
		this.executes.push(item);
		this.overheadMs += performance.now() - start;
		return item;
	}

	finalize(): StateFootprint {
		if (!this.endedAt) {
			this.endedAt = new Date().toISOString();
		}
		return this.getFootprint();
	}

	getFootprint(): StateFootprint {
		return {
			taskId: this.taskId,
			startedAt: this.startedAt,
			endedAt: this.endedAt,
			readSet: this.reads,
			writeSet: this.writes,
			executeSet: this.executes,
			summary: {
				totalReads: this.reads.length,
				totalWrites: this.writes.length,
				totalExecutions: this.executes.length,
				uniqueResourcesRead: Array.from(new Set(this.reads.map((r) => r.resource))),
				uniqueResourcesWritten: Array.from(new Set(this.writes.map((w) => w.resource))),
				uniqueCommandsExecuted: Array.from(new Set(this.executes.map((e) => e.command))),
			},
			overhead: {
				totalMs: Number(this.overheadMs.toFixed(3)),
				formatted: `${this.overheadMs.toFixed(2)}ms`,
			},
		};
	}

	getSubtaskFootprint(subtaskId: number | string): StateFootprint {
		const subIdStr = String(subtaskId);
		const reads = this.reads.filter((r) => String(r.subtaskId) === subIdStr);
		const writes = this.writes.filter((w) => String(w.subtaskId) === subIdStr);
		const executes = this.executes.filter((e) => String(e.subtaskId) === subIdStr);

		return {
			taskId: this.taskId,
			subtaskId,
			startedAt: reads[0]?.timestamp ?? writes[0]?.timestamp ?? executes[0]?.timestamp ?? this.startedAt,
			endedAt:
				executes[executes.length - 1]?.timestamp ??
				writes[writes.length - 1]?.timestamp ??
				reads[reads.length - 1]?.timestamp,
			readSet: reads,
			writeSet: writes,
			executeSet: executes,
			summary: {
				totalReads: reads.length,
				totalWrites: writes.length,
				totalExecutions: executes.length,
				uniqueResourcesRead: Array.from(new Set(reads.map((r) => r.resource))),
				uniqueResourcesWritten: Array.from(new Set(writes.map((w) => w.resource))),
				uniqueCommandsExecuted: Array.from(new Set(executes.map((e) => e.command))),
			},
		};
	}

	getFootprintForInterval(startMs: number, endMs: number, subtaskId?: number | string): StateFootprint {
		const reads = this.reads.filter(
			(r) =>
				(subtaskId !== undefined && String(r.subtaskId) === String(subtaskId)) ||
				(r.timestampMs >= startMs && r.timestampMs <= endMs),
		);
		const writes = this.writes.filter(
			(w) =>
				(subtaskId !== undefined && String(w.subtaskId) === String(subtaskId)) ||
				(w.timestampMs >= startMs && w.timestampMs <= endMs),
		);
		const executes = this.executes.filter(
			(e) =>
				(subtaskId !== undefined && String(e.subtaskId) === String(subtaskId)) ||
				(e.timestampMs >= startMs && e.timestampMs <= endMs),
		);

		return {
			taskId: this.taskId,
			subtaskId,
			startedAt: new Date(startMs).toISOString(),
			endedAt: new Date(endMs).toISOString(),
			readSet: reads,
			writeSet: writes,
			executeSet: executes,
			summary: {
				totalReads: reads.length,
				totalWrites: writes.length,
				totalExecutions: executes.length,
				uniqueResourcesRead: Array.from(new Set(reads.map((r) => r.resource))),
				uniqueResourcesWritten: Array.from(new Set(writes.map((w) => w.resource))),
				uniqueCommandsExecuted: Array.from(new Set(executes.map((e) => e.command))),
			},
		};
	}

	static detectConflicts(taskA: StateFootprint, taskB: StateFootprint): ConflictReport {
		const nameA = taskA.subtaskId !== undefined ? `subtask_${taskA.subtaskId}` : (taskA.taskId ?? "taskA");
		const nameB = taskB.subtaskId !== undefined ? `subtask_${taskB.subtaskId}` : (taskB.taskId ?? "taskB");

		const readSetA = new Set(taskA.readSet.map((r) => r.resource));
		const writeSetA = new Set(taskA.writeSet.map((w) => w.resource));
		const readSetB = new Set(taskB.readSet.map((r) => r.resource));
		const writeSetB = new Set(taskB.writeSet.map((w) => w.resource));

		const readWriteConflicts: Array<{ taskA: string; taskB: string; resource: string }> = [];
		const writeWriteConflicts: Array<{ taskA: string; taskB: string; resource: string }> = [];

		for (const res of readSetA) {
			if (writeSetB.has(res)) {
				readWriteConflicts.push({ taskA: nameA, taskB: nameB, resource: res });
			}
		}

		for (const res of readSetB) {
			if (writeSetA.has(res)) {
				readWriteConflicts.push({ taskA: nameB, taskB: nameA, resource: res });
			}
		}

		for (const res of writeSetA) {
			if (writeSetB.has(res)) {
				writeWriteConflicts.push({ taskA: nameA, taskB: nameB, resource: res });
			}
		}

		return {
			hasConflicts: readWriteConflicts.length > 0 || writeWriteConflicts.length > 0,
			readWriteConflicts,
			writeWriteConflicts,
		};
	}
}

/**
 * An ExecutionEnv wrapper that transparently captures the agent's state footprint
 * across filesystem reads, writes, and shell command executions.
 */
export class TrackingExecutionEnv implements ExecutionEnv {
	private readonly inner: ExecutionEnv;
	readonly tracker: FootprintTracker;

	constructor(inner: ExecutionEnv, tracker: FootprintTracker) {
		this.inner = inner;
		this.tracker = tracker;
	}

	get id(): string {
		return this.inner.id;
	}

	get cwd(): string {
		return this.inner.cwd;
	}

	set cwd(value: string) {
		this.inner.cwd = value;
	}

	absolutePath(path: string, context: Context): Promise<Result<string, FileError>> {
		return this.inner.absolutePath(path, context);
	}

	joinPath(parts: string[], context: Context): Promise<Result<string, FileError>> {
		return this.inner.joinPath(parts, context);
	}

	canonicalPath(path: string, context: Context): Promise<Result<string, FileError>> {
		return this.inner.canonicalPath(path, context);
	}

	exists(path: string, context: Context): Promise<Result<boolean, FileError>> {
		return this.inner.exists(path, context);
	}

	async readTextFile(path: string, context: Context): Promise<Result<string, FileError>> {
		const result = await this.inner.readTextFile(path, context);
		if (result.ok) {
			this.tracker.recordRead(path, result.value);
		}
		return result;
	}

	async openTextLineReader(path: string, context: Context): Promise<Result<TextLineReader, FileError>> {
		const result = await this.inner.openTextLineReader(path, context);
		if (result.ok) {
			const full = await this.inner.readTextFile(path, context);
			if (full.ok) {
				this.tracker.recordRead(path, full.value);
			}
		}
		return result;
	}

	async readTextLines(
		path: string,
		options: { maxLines?: number } | undefined,
		context: Context,
	): Promise<Result<string[], FileError>> {
		const result = await this.inner.readTextLines(path, options, context);
		if (result.ok) {
			this.tracker.recordRead(path, result.value.join("\n"), {
				lineRange: { start: 1, end: result.value.length },
			});
		}
		return result;
	}

	async readBinaryFile(path: string, context: Context): Promise<Result<Uint8Array, FileError>> {
		const result = await this.inner.readBinaryFile(path, context);
		if (result.ok) {
			this.tracker.recordRead(path, result.value);
		}
		return result;
	}

	async openBinaryReader(
		path: string,
		options: { noFollow?: boolean } | undefined,
		context: Context,
	): Promise<Result<BinaryReader, FileError>> {
		const result = await this.inner.openBinaryReader(path, options, context);
		if (result.ok) {
			const full = await this.inner.readBinaryFile(path, context);
			if (full.ok) {
				this.tracker.recordRead(path, full.value);
			}
		}
		return result;
	}

	async writeFile(path: string, content: string | Uint8Array, context: Context): Promise<Result<void, FileError>> {
		const t0 = performance.now();
		let beforeContent: Uint8Array | null = null;
		const readBefore = await this.inner.readBinaryFile(path, context);
		if (readBefore.ok) {
			beforeContent = readBefore.value;
		}
		this.tracker.addOverhead(performance.now() - t0);

		const result = await this.inner.writeFile(path, content, context);
		if (result.ok) {
			this.tracker.recordWrite(path, beforeContent, content);
		}
		return result;
	}

	async appendFile(path: string, content: string | Uint8Array, context: Context): Promise<Result<void, FileError>> {
		const t0 = performance.now();
		let beforeContent: Uint8Array | null = null;
		const readBefore = await this.inner.readBinaryFile(path, context);
		if (readBefore.ok) {
			beforeContent = readBefore.value;
		}
		this.tracker.addOverhead(performance.now() - t0);

		const result = await this.inner.appendFile(path, content, context);
		if (result.ok) {
			const after = await this.inner.readBinaryFile(path, context);
			this.tracker.recordWrite(path, beforeContent, after.ok ? after.value : content);
		}
		return result;
	}

	truncateFile(path: string, size: number, context: Context): Promise<Result<void, FileError>> {
		return this.inner.truncateFile(path, size, context);
	}

	flushFile(path: string, context: Context): Promise<Result<void, FileError>> {
		return this.inner.flushFile(path, context);
	}

	renameFile(sourcePath: string, destinationPath: string, context: Context): Promise<Result<void, FileError>> {
		return this.inner.renameFile(sourcePath, destinationPath, context);
	}

	fileInfo(path: string, context: Context): Promise<Result<FileInfo, FileError>> {
		return this.inner.fileInfo(path, context);
	}

	listDir(path: string, context: Context): Promise<Result<FileInfo[], FileError>> {
		return this.inner.listDir(path, context);
	}

	listFiles(
		path: string,
		options: ListFilesOptions | undefined,
		context: Context,
	): Promise<Result<FileListing, FileError>> {
		if (this.inner.listFiles) {
			return this.inner.listFiles(path, options, context);
		}
		return Promise.resolve({
			ok: false,
			error: { code: "not_supported", message: "listFiles not supported" } as FileError,
		});
	}

	openDirReader(path: string, context: Context): Promise<Result<DirReader, FileError>> {
		return this.inner.openDirReader(path, context);
	}

	createDir(
		path: string,
		options: { recursive?: boolean } | undefined,
		context: Context,
	): Promise<Result<void, FileError>> {
		return this.inner.createDir(path, options, context);
	}

	async remove(
		path: string,
		options: { recursive?: boolean; force?: boolean } | undefined,
		context: Context,
	): Promise<Result<void, FileError>> {
		const t0 = performance.now();
		let beforeContent: Uint8Array | null = null;
		const readBefore = await this.inner.readBinaryFile(path, context);
		if (readBefore.ok) {
			beforeContent = readBefore.value;
		}
		this.tracker.addOverhead(performance.now() - t0);

		const result = await this.inner.remove(path, options, context);
		if (result.ok) {
			this.tracker.recordDelete(path, beforeContent);
		}
		return result;
	}

	createTempDir(prefix: string | undefined, context: Context): Promise<Result<string, FileError>> {
		return this.inner.createTempDir(prefix, context);
	}

	createTempFile(
		options: { prefix?: string; suffix?: string } | undefined,
		context: Context,
	): Promise<Result<string, FileError>> {
		return this.inner.createTempFile(options, context);
	}

	watch(
		targets: readonly WatchTarget[],
		onChange: (change: FileChange) => void,
		context: Context,
	): Promise<Result<FileWatcher, FileError>> {
		return this.inner.watch(targets, onChange, context);
	}

	async exec(
		command: string | readonly string[],
		options: ShellExecOptions | undefined,
		context: Context,
	): Promise<Result<ShellExecResult, ExecutionError>> {
		const cmdString = Array.isArray(command) ? command.join(" ") : String(command);
		const start = Date.now();
		const result = await this.inner.exec(command, options, context);
		const durationMs = Date.now() - start;

		this.tracker.recordExecute(cmdString, {
			cwd: options?.cwd ?? this.inner.cwd,
			exitCode: result.ok ? result.value.exitCode : undefined,
			durationMs,
		});

		return result;
	}

	cleanup(context: Context): Promise<void> {
		return this.inner.cleanup(context);
	}
}
