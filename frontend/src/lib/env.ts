// VITE_* is inlined at build time: setting one on a running container does nothing.

/** Relative so the Vite proxy and nginx both resolve it. Never make this absolute. */
export const API_BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

export const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === "true";
