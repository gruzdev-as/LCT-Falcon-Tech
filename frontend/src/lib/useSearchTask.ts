import { useCallback, useEffect, useReducer, useRef } from "react";

import { ApiError, asApiError, isAbortError } from "../api/client";
import { searchApi } from "../api/index";
import type { SearchInput } from "../api/search";
import type { SearchResult } from "../api/types";
import {
  MAX_NETWORK_ERRORS,
  NOT_FOUND_GRACE_MS,
  POLL_BACKOFF,
  POLL_FIRST_DELAY_MS,
  POLL_MAX_DELAY_MS,
  POLL_TIMEOUT_MS,
} from "./constants";

/** A union rather than loading/error booleans, so the status panel's switch is exhaustive. */
export type SearchState =
  | { phase: "idle" }
  | { phase: "uploading" }
  | { phase: "polling"; taskId: string; elapsedMs: number }
  | { phase: "found"; result: SearchResult }
  | { phase: "rejected"; result: SearchResult }
  | { phase: "failed"; taskId: string; message: string }
  | { phase: "expired"; taskId: string }
  | { phase: "timedOut"; taskId: string }
  | { phase: "error"; error: ApiError };

type Action =
  | { type: "reset" }
  | { type: "submit" }
  | { type: "accepted"; taskId: string }
  | { type: "tick"; elapsedMs: number }
  | { type: "result"; result: SearchResult }
  | { type: "expired"; taskId: string }
  | { type: "timedOut"; taskId: string }
  | { type: "error"; error: ApiError };

function reducer(state: SearchState, action: Action): SearchState {
  switch (action.type) {
    case "reset":
      return { phase: "idle" };
    case "submit":
      return { phase: "uploading" };
    case "accepted":
      return { phase: "polling", taskId: action.taskId, elapsedMs: 0 };
    case "tick":
      return state.phase === "polling" ? { ...state, elapsedMs: action.elapsedMs } : state;
    case "result": {
      const result = action.result;
      // ORDER MATTERS: a failed task also has zero candidates, so testing `rejected`
      // first would render an inference crash as "this car is not in the gallery".
      if (result.status === "failed") {
        return {
          phase: "failed",
          taskId: result.task_id,
          message: result.error ?? "инференс завершился ошибкой",
        };
      }
      if (result.rejected || result.candidates.length === 0) {
        return { phase: "rejected", result };
      }
      return { phase: "found", result };
    }
    case "expired":
      return { phase: "expired", taskId: action.taskId };
    case "timedOut":
      return { phase: "timedOut", taskId: action.taskId };
    case "error":
      return { phase: "error", error: action.error };
  }
}

export function useSearchTask() {
  const [state, dispatch] = useReducer(reducer, { phase: "idle" });
  const abortRef = useRef<AbortController | null>(null);
  const timerRef = useRef<number | null>(null);
  const runRef = useRef(0);

  const stop = useCallback(() => {
    runRef.current += 1;
    abortRef.current?.abort();
    abortRef.current = null;
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Without this the timer chain outlives the component and keeps issuing requests.
  useEffect(() => stop, [stop]);

  const start = useCallback(
    async (input: SearchInput) => {
      stop();
      const run = runRef.current;
      // The controller kills the in-flight request; the run id covers what it cannot —
      // a response already resolved, and a timer callback already queued.
      const alive = () => run === runRef.current;
      const controller = new AbortController();
      abortRef.current = controller;

      dispatch({ type: "submit" });

      let taskId: string;
      try {
        taskId = (await searchApi.submit(input, controller.signal)).task_id;
      } catch (err) {
        if (!alive() || isAbortError(err)) return;
        dispatch({ type: "error", error: asApiError(err) });
        return;
      }
      if (!alive()) return;
      dispatch({ type: "accepted", taskId });

      const startedAt = Date.now();
      let delay = POLL_FIRST_DELAY_MS;
      let networkErrors = 0;

      const poll = async () => {
        if (!alive()) return;

        // Wall clock, not a tick count: background tabs clamp timers to >= 1s.
        const elapsedMs = Date.now() - startedAt;
        if (elapsedMs > POLL_TIMEOUT_MS) {
          dispatch({ type: "timedOut", taskId });
          return;
        }
        dispatch({ type: "tick", elapsedMs });

        try {
          const res = await searchApi.status(taskId, controller.signal);
          if (!alive()) return;
          networkErrors = 0;
          if (res.kind === "done") {
            dispatch({ type: "result", result: res.result });
            return;
          }
        } catch (err) {
          if (!alive() || isAbortError(err)) return;

          if (err instanceof ApiError && err.status === 404) {
            if (elapsedMs > NOT_FOUND_GRACE_MS) {
              dispatch({ type: "expired", taskId });
              return;
            }
          } else if (err instanceof ApiError && err.status >= 400 && err.status < 500) {
            dispatch({ type: "error", error: err });
            return;
          } else if (++networkErrors >= MAX_NETWORK_ERRORS) {
            dispatch({ type: "error", error: asApiError(err) });
            return;
          }
          // otherwise a blip or a backend restart — keep polling
        }

        // Recursive setTimeout, not setInterval: an interval stacks requests when a
        // poll outlives it and cannot back off.
        delay = Math.min(delay * POLL_BACKOFF, POLL_MAX_DELAY_MS);
        timerRef.current = window.setTimeout(poll, delay);
      };

      timerRef.current = window.setTimeout(poll, POLL_FIRST_DELAY_MS);
    },
    [stop],
  );

  const reset = useCallback(() => {
    stop();
    dispatch({ type: "reset" });
  }, [stop]);

  return { state, start, reset };
}
