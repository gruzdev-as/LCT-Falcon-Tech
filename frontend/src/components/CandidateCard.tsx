import { useState } from "react";

import type { Candidate } from "../api/types";

export function CandidateCard({ candidate }: { candidate: Candidate }) {
  const [broken, setBroken] = useState(false);

  return (
    <figure className="group overflow-hidden rounded-xl border border-surface-800 bg-surface-900">
      <div className="relative aspect-4/3 bg-surface-850">
        {candidate.image_url && !broken ? (
          <img
            src={candidate.image_url}
            alt={`Кандидат ${candidate.rank}`}
            loading="lazy"
            // Presigned links expire after S3_PRESIGN_TTL.
            onError={() => setBroken(true)}
            className="size-full object-cover"
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
