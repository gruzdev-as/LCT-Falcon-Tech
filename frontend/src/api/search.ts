import { API_BASE } from "../lib/env";
import { ApiError, toApiError } from "./client";
import type { BBox, SearchAccepted, SearchResult } from "./types";

export interface SearchInput {
  /** The file exactly as dropped. Never re-encoded. */
  file: File;
  bbox: BBox;
  topK: number;
}

export type StatusResponse =
  | { kind: "pending"; taskId: string }
  | { kind: "done"; result: SearchResult };

/**
 * Three things here are load-bearing:
 *  - no trailing slash — the route is @router.post(""), and a 307 loses the body;
 *  - no Content-Type header — the browser must generate the multipart boundary;
 *  - top_k is a form field and bbox a JSON string, since multipart has no nesting.
 */
export async function submitSearch(
  input: SearchInput,
  signal: AbortSignal,
): Promise<SearchAccepted> {
  const form = new FormData();
  form.append("file", input.file, input.file.name || "query.jpg");
  form.append("bbox", JSON.stringify(input.bbox));
  form.append("top_k", String(input.topK));

  const res = await fetch(`${API_BASE}/search`, { method: "POST", body: form, signal });
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as SearchAccepted;
}

export async function fetchSearchStatus(
  taskId: string,
  signal: AbortSignal,
): Promise<StatusResponse> {
  const res = await fetch(`${API_BASE}/search/${taskId}`, {
    signal,
    cache: "no-store",
    headers: { Accept: "application/json" },
  });

  // Discriminate on the status code: res.ok is true for 202, and the pending body is a
  // structural subset of SearchResult, so treating it as one yields candidates:
  // undefined and renders as a false "not found".
  if (res.status === 202) return { kind: "pending", taskId };
  if (!res.ok) throw await toApiError(res);
  return { kind: "done", result: (await res.json()) as SearchResult };
}

/** Unversioned, and proxied separately by nginx. */
export async function checkHealth(signal?: AbortSignal): Promise<boolean> {
  try {
    const res = await fetch("/health", { signal, cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}

export { ApiError };
