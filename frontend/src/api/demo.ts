import type { SearchInput, StatusResponse } from "./search";
import { ApiError } from "./client";
import type { Candidate, SearchAccepted, SearchResult } from "./types";

// Fixture stand-in for the real API, so every screen is reachable before inference/
// exists. Exercises the identical submit → poll → terminal path.
// Pick a scenario with ?demo=found|rejected|failed|timeout.
type Scenario = "found" | "rejected" | "failed" | "timeout";

const SCENARIOS: Scenario[] = ["found", "rejected", "failed", "timeout"];

function currentScenario(): Scenario {
  const raw = new URLSearchParams(window.location.search).get("demo");
  return SCENARIOS.includes(raw as Scenario) ? (raw as Scenario) : "found";
}

const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

const fakeTaskId = () =>
  Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join("");

function demoCandidates(): Candidate[] {
  const scores = [0.94, 0.91, 0.88, 0.85, 0.81, 0.78, 0.74, 0.71];
  return scores.map((score, index) => ({
    image_id: `demo-${index + 1}`,
    score,
    rank: index + 1,
    image_path: `demo/${index + 1}.jpg`,
    image_url: `/demo/${index + 1}.jpg`,
    vehicle_id: "vehicle-042",
    camera_id: `cam-${(index % 3) + 1}`,
  }));
}

function buildResult(taskId: string, scenario: Scenario): SearchResult {
  const base = {
    task_id: taskId,
    model_name: "demo-fixture",
    latency_ms: 184.2,
    finished_at: new Date().toISOString(),
  };

  if (scenario === "failed") {
    return {
      ...base,
      status: "failed",
      candidates: [],
      top_score: null,
      rejected: false,
      error: "модель недоступна: CUDA out of memory",
    };
  }

  if (scenario === "rejected") {
    return {
      ...base,
      status: "done",
      candidates: [],
      top_score: 0.41,
      rejected: true,
      error: null,
    };
  }

  const candidates = demoCandidates();
  return {
    ...base,
    status: "done",
    candidates,
    top_score: candidates[0]!.score,
    rejected: false,
    error: null,
  };
}

let acceptedAt = 0;

export const demoSearchApi = {
  async submit(_input: SearchInput, _signal: AbortSignal): Promise<SearchAccepted> {
    await sleep(350);
    acceptedAt = Date.now();
    return { task_id: fakeTaskId(), status: "pending" };
  },

  async status(taskId: string, _signal: AbortSignal): Promise<StatusResponse> {
    const scenario = currentScenario();

    // Never resolves, so the UI runs its real 60s timeout.
    if (scenario === "timeout") return { kind: "pending", taskId };

    // Stay pending briefly so the polling UI is actually seen.
    if (Date.now() - acceptedAt < 2_500) return { kind: "pending", taskId };

    return { kind: "done", result: buildResult(taskId, scenario) };
  },

  async health(): Promise<boolean> {
    return true;
  },
};

export { ApiError };
