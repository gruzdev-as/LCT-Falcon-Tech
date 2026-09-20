import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../src/api/client";
import type { StatusResponse } from "../src/api/search";
import type { SearchResult } from "../src/api/types";

// Shrink the polling schedule so the suite runs in milliseconds instead of a minute.
// The logic under test is the state machine, not the exact delays.
vi.mock("../src/lib/constants", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../src/lib/constants")>()),
  POLL_FIRST_DELAY_MS: 1,
  POLL_MAX_DELAY_MS: 2,
  POLL_BACKOFF: 1,
  POLL_TIMEOUT_MS: 80,
  NOT_FOUND_GRACE_MS: 5,
  MAX_NETWORK_ERRORS: 3,
}));

const submit = vi.fn();
const status = vi.fn();
const health = vi.fn();

vi.mock("../src/api/index", () => ({
  searchApi: {
    submit: (...args: unknown[]) => submit(...args),
    status: (...args: unknown[]) => status(...args),
    health: (...args: unknown[]) => health(...args),
  },
}));

const { useSearchTask } = await import("../src/lib/useSearchTask");

const input = {
  file: new File(["x"], "car.jpg", { type: "image/jpeg" }),
  bbox: { x: 0, y: 0, width: 100, height: 100 },
  topK: 10,
};

function result(overrides: Partial<SearchResult> = {}): SearchResult {
  return {
    task_id: "abc123",
    status: "done",
    candidates: [],
    top_score: null,
    rejected: false,
    model_name: "test",
    latency_ms: 10,
    error: null,
    finished_at: new Date().toISOString(),
    ...overrides,
  };
}

const pending = (taskId = "abc123"): StatusResponse => ({ kind: "pending", taskId });
const done = (r: SearchResult): StatusResponse => ({ kind: "done", result: r });

beforeEach(() => {
  submit.mockReset().mockResolvedValue({ task_id: "abc123", status: "pending" });
  status.mockReset();
  health.mockReset().mockResolvedValue(true);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useSearchTask", () => {
  it("polls until a result arrives and reports candidates", async () => {
    const hit = result({
      candidates: [
        {
          image_id: "a",
          score: 0.9,
          rank: 1,
          image_path: null,
          image_url: null,
          vehicle_id: null,
          camera_id: null,
        },
      ],
      top_score: 0.9,
    });
    status.mockResolvedValueOnce(pending()).mockResolvedValueOnce(done(hit));

    const { result: hook } = renderHook(() => useSearchTask());
    await act(async () => {
      await hook.current.start(input);
    });

    await waitFor(() => expect(hook.current.state.phase).toBe("found"));
    expect(status).toHaveBeenCalledTimes(2);
  });

  it("treats rejected as a result, not an error", async () => {
    status.mockResolvedValue(done(result({ rejected: true, top_score: 0.41 })));

    const { result: hook } = renderHook(() => useSearchTask());
    await act(async () => {
      await hook.current.start(input);
    });

    await waitFor(() => expect(hook.current.state.phase).toBe("rejected"));
  });

  it("reports a failed task as failed even though it also has zero candidates", async () => {
    // The ordering guard: checking `rejected` first would render an inference crash
    // as a confident "this car is not in the gallery".
    status.mockResolvedValue(
      done(result({ status: "failed", error: "CUDA out of memory", rejected: true })),
    );

    const { result: hook } = renderHook(() => useSearchTask());
    await act(async () => {
      await hook.current.start(input);
    });

    await waitFor(() => expect(hook.current.state.phase).toBe("failed"));
    if (hook.current.state.phase === "failed") {
      expect(hook.current.state.message).toBe("CUDA out of memory");
    }
  });

  it("gives up after the wall-clock budget", async () => {
    status.mockResolvedValue(pending());

    const { result: hook } = renderHook(() => useSearchTask());
    await act(async () => {
      await hook.current.start(input);
    });

    await waitFor(() => expect(hook.current.state.phase).toBe("timedOut"), {
      timeout: 2000,
    });
  });

  it("tolerates a 404 during the grace window, then calls the task expired", async () => {
    status.mockRejectedValue(new ApiError(404, "not_found", "unknown or expired task"));

    const { result: hook } = renderHook(() => useSearchTask());
    await act(async () => {
      await hook.current.start(input);
    });

    await waitFor(() => expect(hook.current.state.phase).toBe("expired"), {
      timeout: 2000,
    });
    // It kept polling through the grace window rather than giving up on the first 404.
    expect(status.mock.calls.length).toBeGreaterThan(1);
  });

  it("surfaces a submit failure without polling", async () => {
    submit.mockRejectedValue(new ApiError(422, "validation_error", "bbox lies outside the image"));

    const { result: hook } = renderHook(() => useSearchTask());
    await act(async () => {
      await hook.current.start(input);
    });

    await waitFor(() => expect(hook.current.state.phase).toBe("error"));
    expect(status).not.toHaveBeenCalled();
  });

  it("keeps polling through a transient 5xx", async () => {
    status
      .mockRejectedValueOnce(new ApiError(502, "http_error", "bad gateway"))
      .mockResolvedValueOnce(done(result({ rejected: true })));

    const { result: hook } = renderHook(() => useSearchTask());
    await act(async () => {
      await hook.current.start(input);
    });

    await waitFor(() => expect(hook.current.state.phase).toBe("rejected"));
  });

  it("stops polling on unmount", async () => {
    status.mockResolvedValue(pending());

    const { result: hook, unmount } = renderHook(() => useSearchTask());
    await act(async () => {
      await hook.current.start(input);
    });
    await waitFor(() => expect(status).toHaveBeenCalled());

    unmount();
    const callsAtUnmount = status.mock.calls.length;

    await new Promise((resolve) => setTimeout(resolve, 40));
    // A few in-flight calls may still land; what must not happen is the chain
    // continuing to issue requests indefinitely.
    expect(status.mock.calls.length).toBeLessThanOrEqual(callsAtUnmount + 1);
  });

  it("reset returns to idle", async () => {
    status.mockResolvedValue(done(result({ rejected: true })));

    const { result: hook } = renderHook(() => useSearchTask());
    await act(async () => {
      await hook.current.start(input);
    });
    await waitFor(() => expect(hook.current.state.phase).toBe("rejected"));

    act(() => hook.current.reset());
    expect(hook.current.state.phase).toBe("idle");
  });
});
