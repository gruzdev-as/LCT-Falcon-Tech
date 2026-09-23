import { useRef, useState, type PointerEvent as ReactPointerEvent } from "react";

import type { BBox } from "../api/types";
import { MIN_SIDE_PX } from "../lib/constants";
import {
  applyResize,
  boxFromPoints,
  clampBoxInside,
  HANDLE_POSITION,
  HANDLES,
  isBoxValid,
  toOriginal,
  toRenderedRect,
  type Handle,
  type Point,
  type Size,
} from "../lib/geometry";
import { useElementSize } from "../lib/useElementSize";

type Drag =
  | { kind: "draw"; pointerId: number; anchor: Point; rectWidth: number }
  | { kind: "move"; pointerId: number; start: BBox; grab: Point }
  | { kind: "resize"; pointerId: number; start: BBox; handle: Handle };

interface Props {
  /** Mount with key={src} so a new image resets state. */
  src: string;
  box: BBox | null;
  onChange: (box: BBox | null) => void;
  onNaturalSize?: (size: Size) => void;
}

/** Box state is kept ONLY in original image pixels; rendered geometry is derived. */
export function BBoxEditor({ src, box, onChange, onNaturalSize }: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const surfaceRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<Drag | null>(null);
  const [natural, setNatural] = useState<Size | null>(null);
  const rendered = useElementSize(imgRef);

  const handleLoad = () => {
    const img = imgRef.current;
    if (!img) return;
    const size = { width: img.naturalWidth, height: img.naturalHeight };
    setNatural(size);
    onNaturalSize?.(size);
  };

  // Read the rect per event: caching it on pointerdown breaks if the page scrolls.
  const pointFrom = (event: ReactPointerEvent): Point | null => {
    const surface = surfaceRef.current;
    if (!surface || !natural) return null;
    return toOriginal(event.clientX, event.clientY, surface.getBoundingClientRect(), natural);
  };

  const canStart = (event: ReactPointerEvent) =>
    natural !== null && event.isPrimary && event.button === 0 && dragRef.current === null;

  const startDraw = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!canStart(event)) return;
    const anchor = pointFrom(event);
    const surface = surfaceRef.current;
    if (!anchor || !surface) return;

    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      kind: "draw",
      pointerId: event.pointerId,
      anchor,
      rectWidth: surface.getBoundingClientRect().width,
    };
    onChange(null);
  };

  const startMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!canStart(event) || !box) return;
    const grab = pointFrom(event);
    if (!grab) return;

    event.stopPropagation(); // otherwise the surface below also starts a draw
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { kind: "move", pointerId: event.pointerId, start: box, grab };
  };

  const startResize = (handle: Handle) => (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (!canStart(event) || !box) return;
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { kind: "resize", pointerId: event.pointerId, start: box, handle };
  };

  // One handler on the surface: pointer capture retargets, and React still bubbles.
  const handleMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId || !natural) return;

    const p = pointFrom(event);
    if (!p) return;

    switch (drag.kind) {
      case "draw":
        onChange(boxFromPoints(drag.anchor, p));
        break;
      case "resize":
        onChange(applyResize(drag.start, drag.handle, p));
        break;
      case "move":
        onChange(
          clampBoxInside(
            {
              ...drag.start,
              x: drag.start.x + (p.x - drag.grab.x),
              y: drag.start.y + (p.y - drag.grab.y),
            },
            natural,
          ),
        );
        break;
    }
  };

  // pointerup releases the capture implicitly; pointercancel covers gesture stealing.
  const endDrag = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    dragRef.current = null;

    // A plain click gives a 0×0 box. Treat a tiny drag as a click so no phantom box
    // appears. The one place a rendered-pixel threshold is right: motor precision.
    if (drag.kind === "draw" && box && natural && drag.rectWidth > 0) {
      const slop = 4 * (natural.width / drag.rectWidth);
      if (box.width < slop && box.height < slop) onChange(null);
    }
  };

  const valid = isBoxValid(box);
  const overlay = box && natural && rendered ? toRenderedRect(box, rendered, natural) : null;

  return (
    <div className="relative inline-block overflow-hidden rounded-xl select-none">
      <img
        ref={imgRef}
        src={src}
        alt="Загруженное изображение"
        draggable={false} // else the native image drag hijacks every gesture
        onLoad={handleLoad}
        className="block max-h-[70vh] max-w-full"
      />

      <div
        ref={surfaceRef}
        // touch-none stops the browser claiming the gesture, and suppresses the
        // compatibility mouse events that would fire a phantom second drag.
        className="absolute inset-0 touch-none cursor-crosshair"
        onPointerDown={startDraw}
        onPointerMove={handleMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
      >
        {overlay && box && (
          <div
            style={overlay}
            onPointerDown={startMove}
            className={`absolute cursor-move ring-2 ${
              valid ? "ring-brand-400" : "ring-danger-400"
            } shadow-[0_0_0_100vmax_rgba(10,8,18,0.6)]`}
          >
            {HANDLES.map((handle) => (
              <button
                key={handle}
                type="button"
                aria-label={`Изменить размер: ${handle}`}
                onPointerDown={startResize(handle)}
                className={`absolute size-3 rounded-sm border border-surface-950 ${
                  valid ? "bg-brand-400" : "bg-danger-400"
                } ${HANDLE_POSITION[handle]}`}
              />
            ))}

            <span className="absolute -top-7 left-0 rounded bg-surface-900/90 px-2 py-0.5 text-xs tabular-nums whitespace-nowrap text-ink-300">
              {Math.round(box.width)} × {Math.round(box.height)} px
              {valid ? "" : ` · минимум ${MIN_SIDE_PX}`}
            </span>
          </div>
        )}
      </div>

      {!box && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <span className="rounded-lg bg-surface-950/80 px-3 py-1.5 text-sm text-ink-300">
            Выделите автомобиль рамкой
          </span>
        </div>
      )}
    </div>
  );
}
