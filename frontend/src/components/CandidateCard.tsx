import { useState } from "react";

import type { Candidate } from "../api/types";
import { cropWindow, type Size } from "../lib/geometry";

/** Matches the aspect-4/3 tile below; the window is padded out to it. */
const TILE_ASPECT = 4 / 3;

export function CandidateCard({ candidate }: { candidate: Candidate }) {
  const [broken, setBroken] = useState(false);
  const [natural, setNatural] = useState<Size | null>(null);

  // The gallery stores whole frames, so the card crops to the vehicle. Natural size only
  // arrives with the load event, and an older gallery may carry no box at all — in both
  // cases the frame is shown whole rather than guessed at.
  const view = candidate.bbox && natural ? cropWindow(candidate.bbox, natural, TILE_ASPECT) : null;
  const cropped = view
    ? {
        position: "absolute" as const,
        maxWidth: "none",
        width: `${(natural!.width / view.width) * 100}%`,
        height: `${(natural!.height / view.height) * 100}%`,
        left: `${(-view.x / view.width) * 100}%`,
        top: `${(-view.y / view.height) * 100}%`,
      }
    : undefined;

  return (
    <figure className="group overflow-hidden rounded-xl border border-surface-800 bg-surface-900">
      <div className="relative aspect-4/3 overflow-hidden bg-surface-850">
        {candidate.image_url && !broken ? (
          <img
            src={candidate.image_url}
            alt={`Кандидат ${candidate.rank}`}
            loading="lazy"
            // Presigned links expire after S3_PRESIGN_TTL.
            onError={() => setBroken(true)}
            onLoad={(event) =>
              setNatural({
                width: event.currentTarget.naturalWidth,
                height: event.currentTarget.naturalHeight,
              })
            }
            style={cropped}
            // Without a box the frame is cover-fitted; the pending state hides the jump
            // from whole frame to crop on the first paint.
            className={
              view
                ? "transition-opacity duration-150"
                : candidate.bbox
                  ? "size-full object-cover opacity-0"
                  : "size-full object-cover"
            }
          />
        ) : (
          <div className="flex size-full items-center justify-center px-4 text-center text-xs text-ink-600">
            {broken ? "Ссылка истекла — повторите поиск" : "Изображение недоступно"}
          </div>
        )}

        <span className="absolute top-2 left-2 rounded-md bg-surface-950/80 px-1.5 py-0.5 text-xs font-medium tabular-nums text-ink-300">
          #{candidate.rank}
        </span>
      </div>

      <figcaption className="flex items-center justify-between gap-2 px-3 py-2.5">
        <span className="text-sm font-semibold tabular-nums text-brand-300">
          {(candidate.score * 100).toFixed(1)}%
        </span>
        {candidate.camera_id && (
          <span className="truncate text-xs text-ink-600">{candidate.camera_id}</span>
        )}
      </figcaption>
    </figure>
  );
}
