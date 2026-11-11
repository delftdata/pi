import { describe, expect, it } from "vitest";
import { FootprintTracker, type PredictedFootprint } from "../src/index.ts";

describe("FootprintTracker", () => {
	it("records reads, writes, and executes", () => {
		const tracker = new FootprintTracker("task_1");
		tracker.setSubtask("sub_1");

		tracker.recordRead("/workspace/foo.ts", "console.log('hello');");
		tracker.recordWrite("/workspace/bar.ts", null, "export const x = 1;");
		tracker.recordExecute("npm test", { exitCode: 0, durationMs: 120 });

		const fp = tracker.finalize();
		expect(fp.readSet).toHaveLength(1);
		expect(fp.readSet[0].resource).toBe("file:/workspace/foo.ts");
		expect(fp.readSet[0].subtaskId).toBe("sub_1");

		expect(fp.writeSet).toHaveLength(1);
		expect(fp.writeSet[0].resource).toBe("file:/workspace/bar.ts");
		expect(fp.writeSet[0].operation).toBe("create");

		expect(fp.executeSet).toHaveLength(1);
		expect(fp.executeSet[0].command).toBe("npm test");
		expect(fp.executeSet[0].exitCode).toBe(0);
	});

	it("detects conflicts between concurrent footprints", () => {
		const trackerA = new FootprintTracker("task_A");
		trackerA.recordWrite("/workspace/shared.ts", null, "const a = 1;");
		const fpA = trackerA.finalize();

		const trackerB = new FootprintTracker("task_B");
		trackerB.recordRead("/workspace/shared.ts", "const a = 1;");
		const fpB = trackerB.finalize();

		const conflicts = FootprintTracker.detectConflicts(fpA, fpB);
		expect(conflicts.hasConflicts).toBe(true);
		expect(conflicts.readWriteConflicts).toHaveLength(1);
		expect(conflicts.readWriteConflicts[0].resource).toBe("file:/workspace/shared.ts");
	});

	it("compares predicted footprint vs actual footprint", () => {
		const predicted: PredictedFootprint = {
			read: ["foo.py", "test_foo.py"],
			write: ["foo.py"],
			execute: ["python3 test_foo.py"],
			reasoning: "Fix bug in foo.py and run unit test",
		};

		const tracker = new FootprintTracker("task_2");
		tracker.recordRead("/tmp/repo/foo.py", "def foo(): pass");
		tracker.recordRead("/tmp/repo/test_foo.py", "def test_foo(): assert True");
		tracker.recordWrite("/tmp/repo/foo.py", "def foo(): pass", "def foo(): return 1");
		tracker.recordExecute("python3 test_foo.py", { exitCode: 0 });

		const actual = tracker.finalize();
		const comparison = FootprintTracker.compareFootprints(predicted, actual, "/tmp/repo");

		expect(comparison.read.precision).toBe(1.0);
		expect(comparison.read.recall).toBe(1.0);
		expect(comparison.read.truePositives).toContain("foo.py");
		expect(comparison.read.truePositives).toContain("test_foo.py");

		expect(comparison.write.precision).toBe(1.0);
		expect(comparison.write.recall).toBe(1.0);
		expect(comparison.write.truePositives).toContain("foo.py");

		expect(comparison.execute.precision).toBe(1.0);
		expect(comparison.execute.recall).toBe(1.0);

		expect(comparison.summary.exactMatch).toBe(true);
		expect(comparison.summary.overallPrecision).toBe(1.0);
		expect(comparison.summary.overallRecall).toBe(1.0);
	});

	it("identifies false positives and false negatives in footprint predictions", () => {
		const predicted: PredictedFootprint = {
			read: ["a.py", "unneeded.py"],
			write: ["b.py"],
			execute: ["pytest test_b.py"],
		};

		const tracker = new FootprintTracker("task_3");
		tracker.recordRead("/tmp/repo/a.py", "a");
		tracker.recordRead("/tmp/repo/surprise.py", "surprise");
		tracker.recordWrite("/tmp/repo/b.py", "old", "new");
		tracker.recordExecute("pytest test_b.py", { exitCode: 0 });

		const actual = tracker.finalize();
		const comparison = FootprintTracker.compareFootprints(predicted, actual, "/tmp/repo");

		expect(comparison.read.truePositives).toEqual(["a.py"]);
		expect(comparison.read.falsePositives).toEqual(["unneeded.py"]);
		expect(comparison.read.falseNegatives).toEqual(["surprise.py"]);
		expect(comparison.summary.exactMatch).toBe(false);
	});
});
