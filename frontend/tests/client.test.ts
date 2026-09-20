import { describe, expect, it } from "vitest";

import { ApiError, asApiError, toApiError } from "../src/api/client";

const jsonResponse = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });

describe("toApiError", () => {
  it("reads the backend's domain error shape", async () => {
    const err = await toApiError(
      jsonResponse(422, {
        error: "validation_error",
        message: "file exceeds 20 MiB",
        details: { size_bytes: 24117248 },
      }),
    );

    expect(err.status).toBe(422);
    expect(err.code).toBe("validation_error");
    expect(err.message).toBe("file exceeds 20 MiB");
    expect(err.details).toEqual({ size_bytes: 24117248 });
  });

  it("keeps a multi-line message intact", async () => {
    // A bbox under 16px comes back as a wrapped pydantic error, newlines and all.
    const message =
      "could not parse bbox: 1 validation error for BBox\n  Value error, bbox side must be at least 16px";
    const err = await toApiError(
      jsonResponse(422, { error: "validation_error", message, details: {} }),
    );

    expect(err.message).toBe(message);
  });

  it("flattens FastAPI's own validation shape", async () => {
    const err = await toApiError(
      jsonResponse(422, {
        detail: [
          { loc: ["body", "top_k"], msg: "Input should be less than or equal to 100", type: "less_than_equal" },
        ],
      }),
    );

    expect(err.code).toBe("validation_error");
    expect(err.message).toBe("top_k: Input should be less than or equal to 100");
  });

  it("handles a plain-string detail", async () => {
    const err = await toApiError(jsonResponse(405, { detail: "Method Not Allowed" }));
    expect(err.message).toBe("Method Not Allowed");
  });

  it("survives a non-JSON body and names the nginx 413", async () => {
    // nginx rejects an oversized upload with an HTML error page, so res.json() throws.
    const html = new Response("<html><body><h1>413 Request Entity Too Large</h1></body></html>", {
      status: 413,
      headers: { "Content-Type": "text/html" },
    });

    const err = await toApiError(html);
    expect(err.status).toBe(413);
    expect(err.code).toBe("too_large");
  });

  it("falls back to the status code for an unrecognised body", async () => {
    const err = await toApiError(new Response("", { status: 502 }));
    expect(err.status).toBe(502);
    expect(err.code).toBe("http_error");
  });
});

describe("asApiError", () => {
  it("passes an ApiError through unchanged", () => {
    const original = new ApiError(404, "not_found", "gone");
    expect(asApiError(original)).toBe(original);
  });

  it("wraps a thrown TypeError from fetch as a network error", () => {
    const err = asApiError(new TypeError("Failed to fetch"));
    expect(err.status).toBe(0);
    expect(err.code).toBe("network_error");
  });
});
