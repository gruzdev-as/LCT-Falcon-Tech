import { useEffect, useState, type RefObject } from "react";

import type { Size } from "./geometry";

/**
 * Track an element's rendered size. ResizeObserver rather than a window listener: it
 * also fires for font loads, flex reflow and the image finishing decode.
 *
 * Size only — position must be read per event, since the element also moves on scroll.
 */
export function useElementSize(ref: RefObject<Element | null>): Size | null {
  const [size, setSize] = useState<Size | null>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const rect = entry.contentRect;
      setSize({ width: rect.width, height: rect.height });
    });

    observer.observe(element);
    return () => observer.disconnect();
  }, [ref]);

  return size;
}
