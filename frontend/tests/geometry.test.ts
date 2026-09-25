import { describe, expect, it } from "vitest";

import {
  applyResize,
  boxFromPoints,
  clampBoxInside,
  cropWindow,
  isBoxValid,
  toOriginal,
  toRenderedRect,
} from "../src/lib/geometry";

/** Stand-in for a DOMRect; only left/top/width/height are read. */
const rect = (left: number, top: number, width: number, height: number) =>
  ({ left, top, width, height }) as DOMRect;

describe("boxFromPoints", () => {
  it("normalizes a drag in any direction to the same box", () => {
    const a = { x: 10, y: 20 };
    const b = { x: 110, y: 220 };
    const expected = { x: 10, y: 20, width: 100, height: 200 };

    expect(boxFromPoints(a, b)).toEqual(expected);
    expect(boxFromPoints(b, a)).toEqual(expected);
    expect(boxFromPoints({ x: 110, y: 20 }, { x: 10, y: 220 })).toEqual(expected);
    expect(boxFromPoints({ x: 10, y: 220 }, { x: 110, y: 20 })).toEqual(expected);
  });

  it("never produces a negative side", () => {
    const box = boxFromPoints({ x: 500, y: 500 }, { x: 0, y: 0 });
    expect(box.width).toBeGreaterThanOrEqual(0);
    expect(box.height).toBeGreaterThanOrEqual(0);
  });
});

describe("toOriginal", () => {
  it("scales a downscaled image back to original pixels", () => {
    // A 6000x4000 photo drawn at 900x600 — scale 6.667.
    const natural = { width: 6000, height: 4000 };
    const r = rect(100, 50, 900, 600);

    expect(toOriginal(100, 50, r, natural)).toEqual({ x: 0, y: 0 });
    expect(toOriginal(1000, 650, r, natural)).toEqual({ x: 6000, y: 4000 });
    expect(toOriginal(550, 350, r, natural)).toEqual({ x: 3000, y: 2000 });
  });

  it("scales an upscaled image back down", () => {
    // A 320x240 thumbnail drawn at 960x720 — scale 0.333.
    const natural = { width: 320, height: 240 };
    const r = rect(0, 0, 960, 720);

    expect(toOriginal(480, 360, r, natural)).toEqual({ x: 160, y: 120 });
  });

  it("subtracts the element offset so scroll position cannot leak in", () => {
    const natural = { width: 100, height: 100 };
    expect(toOriginal(250, 400, rect(200, 300, 100, 100), natural)).toEqual({
      x: 50,
      y: 100,
    });
  });

  it("clamps a pointer dragged outside the image to the frame", () => {
    const natural = { width: 100, height: 100 };
    const r = rect(0, 0, 100, 100);

    expect(toOriginal(-50, -50, r, natural)).toEqual({ x: 0, y: 0 });
    expect(toOriginal(9999, 9999, r, natural)).toEqual({ x: 100, y: 100 });
  });

  it("round-trips through toRenderedRect", () => {
    const natural = { width: 4000, height: 3000 };
    const rendered = { width: 800, height: 600 };
    const box = { x: 1000, y: 750, width: 500, height: 300 };

    expect(toRenderedRect(box, rendered, natural)).toEqual({
      left: 200,
      top: 150,
      width: 100,
      height: 60,
    });
  });
});

describe("isBoxValid", () => {
  it("measures the 16px floor in original pixels", () => {
    expect(isBoxValid({ x: 0, y: 0, width: 16, height: 16 })).toBe(true);
    expect(isBoxValid({ x: 0, y: 0, width: 15, height: 100 })).toBe(false);
    expect(isBoxValid({ x: 0, y: 0, width: 100, height: 15 })).toBe(false);
    expect(isBoxValid(null)).toBe(false);
  });

  it("accepts a box that is tiny on screen but valid in original pixels", () => {
    // 16 original px on a 6000px photo shown at 900px is 2.4 rendered px. Gating on
    // rendered size would wrongly reject this.
    const natural = { width: 6000, height: 4000 };
    const rendered = { width: 900, height: 600 };
    const box = { x: 0, y: 0, width: 16, height: 16 };

    expect(isBoxValid(box)).toBe(true);
    expect(toRenderedRect(box, rendered, natural).width).toBeLessThan(3);
  });
});

describe("applyResize", () => {
  const start = { x: 100, y: 100, width: 200, height: 200 };

  it("anchors the opposite corner when dragging a corner handle", () => {
    expect(applyResize(start, "se", { x: 400, y: 500 })).toEqual({
      x: 100,
      y: 100,
      width: 300,
      height: 400,
    });
    expect(applyResize(start, "nw", { x: 50, y: 50 })).toEqual({
      x: 50,
      y: 50,
      width: 250,
      height: 250,
    });
  });

  it("freezes one axis for an edge handle", () => {
    // Dragging the east edge moves x2 only; y and height are untouched.
    const resized = applyResize(start, "e", { x: 350, y: 9999 });
    expect(resized).toEqual({ x: 100, y: 100, width: 250, height: 200 });

    const north = applyResize(start, "n", { x: 9999, y: 50 });
    expect(north).toEqual({ x: 100, y: 50, width: 200, height: 250 });
  });

  it("flips through zero when a handle is dragged past the opposite edge", () => {
    // North handle dragged below the south edge: the box inverts and normalizes.
    const flipped = applyResize(start, "n", { x: 0, y: 400 });
    expect(flipped).toEqual({ x: 100, y: 300, width: 200, height: 100 });
    expect(flipped.height).toBeGreaterThan(0);
  });
});

describe("clampBoxInside", () => {
  const natural = { width: 1000, height: 800 };

  it("keeps the size and pushes the box back into the frame", () => {
    expect(clampBoxInside({ x: 950, y: 700, width: 200, height: 200 }, natural)).toEqual({
      x: 800,
      y: 600,
      width: 200,
      height: 200,
    });
    expect(clampBoxInside({ x: -50, y: -50, width: 100, height: 100 }, natural)).toEqual({
      x: 0,
      y: 0,
      width: 100,
      height: 100,
    });
  });

  it("shrinks a box larger than the image", () => {
    expect(clampBoxInside({ x: 0, y: 0, width: 5000, height: 5000 }, natural)).toEqual({
      x: 0,
      y: 0,
      width: 1000,
      height: 800,
    });
  });
});

describe("cropWindow", () => {
  const natural = { width: 1000, height: 1000 };
  const ASPECT = 4 / 3;

  it("pads a tall box sideways instead of stretching it", () => {
    const view = cropWindow({ x: 400, y: 400, width: 100, height: 200 }, natural, ASPECT);

    expect(view.height).toBe(200);
    expect(view.width / view.height).toBeCloseTo(ASPECT);
    // The car stays centred in what the tile shows.
    expect(view.x + view.width / 2).toBeCloseTo(450);
  });

  it("pads a wide box vertically", () => {
    const view = cropWindow({ x: 100, y: 500, width: 400, height: 100 }, natural, ASPECT);

    expect(view.width).toBe(400);
    expect(view.height).toBeCloseTo(300);
  });

  it("slides a window at the edge inside the frame rather than shrinking it", () => {
    const view = cropWindow({ x: 0, y: 0, width: 400, height: 100 }, natural, ASPECT);

    expect(view.x).toBe(0);
    expect(view.y).toBe(0);
    expect(view.width).toBe(400);
    expect(view.height).toBeCloseTo(300);
  });

  it("scales down a window the frame cannot hold, keeping the ratio", () => {
    const view = cropWindow({ x: 0, y: 0, width: 100, height: 40 }, { width: 100, height: 50 }, ASPECT);

    expect(view.width / view.height).toBeCloseTo(ASPECT);
    expect(view.height).toBeLessThanOrEqual(50);
    expect(view.width).toBeLessThanOrEqual(100);
    expect(view.y).toBe(0);
  });
});
