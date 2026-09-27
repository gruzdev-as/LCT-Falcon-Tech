// Mirrors common/src/configs/constants.py. Duplicated only for instant feedback —
// the backend enforces these for real.

export const MIN_SIDE_PX = 16;
export const MAX_IMAGE_BYTES = 20 * 1024 * 1024;

export const ALLOWED_IMAGE_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/bmp",
] as const;

// A ceiling, not a count: inference returns only the candidates that clear its
// refusal, usually two or three. Ten is what the results column fits.
export const TOP_K = 10;

// Polling schedule: 400 → 600 → 900 → 1350 → 2000 → 2000…
export const POLL_FIRST_DELAY_MS = 400;
export const POLL_MAX_DELAY_MS = 2_000;
export const POLL_BACKOFF = 1.5;
export const POLL_TIMEOUT_MS = 60_000;

/** The pending marker and the result key are written by two processes, so an early
    404 is a race, not an expired task. */
export const NOT_FOUND_GRACE_MS = 2_000;

export const MAX_NETWORK_ERRORS = 3;
