import type { BBox } from "../api/types";
import { MIN_SIDE_PX } from "./constants";

export interface Point {
  x: number;
  y: number;
}

export interface Size {
  width: number;
  height: number;
}

export const clamp = (value: number, lo: number, hi: number): number =>
  Math.min(Math.max(value, lo), hi);

/** min/abs absorb negative deltas, which is the whole answer to dragging any direction. */
export function boxFromPoints(a: Point, b: Point): BBox {
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    width: Math.abs(a.x - b.x),
    height: Math.abs(a.y - b.y),
  };
}

/**
 * Viewport coordinates → ORIGINAL image pixels.
 *
 * The editor sizes the image with max-width/max-height rather than object-fit, so the
 * element rect is the drawn content box and there is no letterbox offset to subtract.
 */
export function toOriginal(
  clientX: number,
  clientY: number,
  rect: DOMRect,
  natural: Size,
): Point {
  const scaleX = natural.width / rect.width;
  const scaleY = natural.height / rect.height;
  return {
    x: clamp(Math.round((clientX - rect.left) * scaleX), 0, natural.width),
    y: clamp(Math.round((clientY - rect.top) * scaleY), 0, natural.height),
  };
}

export function toRenderedRect(
  box: BBox,
  rendered: Size,
  natural: Size,
): { left: number; top: number; width: number; height: number } {
  const scaleX = rendered.width / natural.width;
  const scaleY = rendered.height / natural.height;
  return {
    left: box.x * scaleX,
    top: box.y * scaleY,
    width: box.width * scaleX,
    height: box.height * scaleY,
  };
}

/**
 * The region of the ORIGINAL a card should show, in original pixels.
 *
 * The gallery stores whole frames, so a card has to crop. The box is padded out to the
 * tile's aspect ratio rather than stretched into it: a car keeps its proportions, and
 * the tile fills with the pixels around the car instead of bars.
 */
export function cropWindow(box: BBox, natural: Size, aspect: number): BBox {
  let width = box.width;
  let height = box.height;
  if (width / height < aspect) width = height * aspect;
  else height = width / aspect;

  // A box near an edge can ask for more than the frame holds. Shrink both sides by the
  // same factor, so the ratio survives and the window still fits.
  const scale = Math.min(1, natural.width / width, natural.height / height);
  width *= scale;
  height *= scale;

  return {
    width,
    height,
    x: clamp(box.x + box.width / 2 - width / 2, 0, natural.width - width),
    y: clamp(box.y + box.height / 2 - height / 2, 0, natural.height - height),
  };
}

/** Keep a box inside the frame without changing its size. */
export function clampBoxInside(box: BBox, natural: Size): BBox {
  const width = Math.min(box.width, natural.width);
  const height = Math.min(box.height, natural.height);
  return {
    width,
    height,
    x: clamp(box.x, 0, natural.width - width),
    y: clamp(box.y, 0, natural.height - height),
  };
}

/**
 * Measured in ORIGINAL pixels. On a 6000px photo shown at 900px a valid 16px box is
 * 2.4px on screen — gating on rendered size would reject boxes the server accepts.
 */
export const isBoxValid = (box: BBox | null): box is BBox =>
  box !== null && box.width >= MIN_SIDE_PX && box.height >= MIN_SIDE_PX;

export type Handle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

export const HANDLES: Handle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

// Per handle: which corner stays anchored (1 = the far edge) and which axes the
// pointer may move. Collapses draw, edge-resize and corner-resize into one path.
const SPEC: Record<Handle, { ax: 0 | 1; ay: 0 | 1; fx: boolean; fy: boolean }> = {
  nw: { ax: 1, ay: 1, fx: true, fy: true },
  n: { ax: 0, ay: 1, fx: false, fy: true },
  ne: { ax: 0, ay: 1, fx: true, fy: true },
  e: { ax: 0, ay: 0, fx: true, fy: false },
  se: { ax: 0, ay: 0, fx: true, fy: true },
  s: { ax: 0, ay: 0, fx: false, fy: true },
  sw: { ax: 1, ay: 0, fx: true, fy: true },
  w: { ax: 1, ay: 0, fx: true, fy: false },
};

/** Dragging a handle past the opposite edge flips the box and normalizes it. */
export function applyResize(start: BBox, handle: Handle, p: Point): BBox {
  const { ax, ay, fx, fy } = SPEC[handle];
  const anchor = {
    x: ax ? start.x + start.width : start.x,
    y: ay ? start.y + start.height : start.y,
  };
  const frozen = {
    x: ax ? start.x : start.x + start.width,
    y: ay ? start.y : start.y + start.height,
  };
  return boxFromPoints(anchor, { x: fx ? p.x : frozen.x, y: fy ? p.y : frozen.y });
}

export const HANDLE_POSITION: Record<Handle, string> = {
  nw: "-top-1.5 -left-1.5 cursor-nwse-resize",
  n: "-top-1.5 left-1/2 -translate-x-1/2 cursor-ns-resize",
  ne: "-top-1.5 -right-1.5 cursor-nesw-resize",
  e: "top-1/2 -right-1.5 -translate-y-1/2 cursor-ew-resize",
  se: "-bottom-1.5 -right-1.5 cursor-nwse-resize",
  s: "-bottom-1.5 left-1/2 -translate-x-1/2 cursor-ns-resize",
  sw: "-bottom-1.5 -left-1.5 cursor-nesw-resize",
  w: "top-1/2 -left-1.5 -translate-y-1/2 cursor-ew-resize",
};
