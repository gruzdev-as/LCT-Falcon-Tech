import type { ErrorResponse } from "./types";

export class ApiError extends Error {
  // Explicit fields rather than constructor parameter properties: those are not
  // erasable syntax and tsconfig sets erasableSyntaxOnly.
  readonly status: number;
  readonly code: string;
  readonly details?: unknown;

  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

interface FastApiDetail {
  loc: (string | number)[];
  msg: string;
  type: string;
}

function isDomainError(body: unknown): body is ErrorResponse {
  return (
    typeof body === "object" &&
    body !== null &&
    typeof (body as ErrorResponse).error === "string" &&
    typeof (body as ErrorResponse).message === "string"
  );
}

function isFastApiValidationError(body: unknown): body is { detail: FastApiDetail[] } {
  const detail = (body as { detail?: unknown })?.detail;
  return Array.isArray(detail) && detail.length > 0 && typeof detail[0]?.msg === "string";
}

/**
 * Three bodies can arrive here: the backend's {error, message, details}, FastAPI's own
 * {detail: [...]}, and HTML when nginx rejects the request itself. res.json() throwing
 * is expected, not exceptional.
 */
export async function toApiError(res: Response): Promise<ApiError> {
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // Not JSON.
  }

  if (isDomainError(body)) {
    return new ApiError(res.status, body.error, body.message, body.details);
  }

  if (isFastApiValidationError(body)) {
    const first = body.detail[0]!;
    const field = first.loc.filter((part) => part !== "body").join(".");
    return new ApiError(
      res.status,
      "validation_error",
      field ? `${field}: ${first.msg}` : first.msg,
      body.detail,
    );
  }

  if (typeof (body as { detail?: unknown })?.detail === "string") {
    return new ApiError(res.status, "http_error", (body as { detail: string }).detail);
  }

  if (res.status === 413) {
    return new ApiError(413, "too_large", "Файл слишком большой для сервера.");
  }

  return new ApiError(res.status, "http_error", `Запрос не удался (${res.status})`);
}

export function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

export function asApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err;
  return new ApiError(0, "network_error", err instanceof Error ? err.message : "Сеть недоступна");
}
