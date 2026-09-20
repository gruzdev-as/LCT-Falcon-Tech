import { useCallback, useEffect, useState } from "react";

import { searchApi } from "../api/index";
import type { BBox } from "../api/types";
import { DEFAULT_TOP_K, MIN_SIDE_PX } from "../lib/constants";
import { isBoxValid } from "../lib/geometry";
import { useSearchTask } from "../lib/useSearchTask";
import { BBoxEditor } from "./BBoxEditor";
import { ImageDropzone } from "./ImageDropzone";
import { StatusPanel } from "./StatusPanel";
import { TopKControl } from "./TopKControl";

interface Upload {
  file: File;
  /** Revoked when replaced: an unrevoked blob URL pins the decoded bitmap. */
  url: string;
}

export function SearchWidget() {
  const [upload, setUpload] = useState<Upload | null>(null);
  const [box, setBox] = useState<BBox | null>(null);
  const [topK, setTopK] = useState(DEFAULT_TOP_K);
  const [fileError, setFileError] = useState<string | null>(null);
  const [workerMissing, setWorkerMissing] = useState<boolean | null>(null);

  const { state, start, reset } = useSearchTask();

  // Cleanup runs with the previous upload on change and the current one on unmount.
  useEffect(
    () => () => {
      if (upload) URL.revokeObjectURL(upload.url);
    },
    [upload],
  );

  // A timeout is ambiguous: the API may be down, or up with nothing consuming the
  // queue. One health check tells them apart.
  useEffect(() => {
    if (state.phase !== "timedOut") return;
    let cancelled = false;
    void searchApi.health().then((ok) => {
      if (!cancelled) setWorkerMissing(ok);
    });
    return () => {
      cancelled = true;
    };
  }, [state.phase]);

  const acceptFile = useCallback(
    (file: File) => {
      setFileError(null);
      setBox(null);
      setWorkerMissing(null);
      reset();
      setUpload({ file, url: URL.createObjectURL(file) });
    },
    [reset],
  );

  const clearUpload = useCallback(() => {
    reset();
    setBox(null);
    setWorkerMissing(null);
    setUpload(null);
  }, [reset]);

  const busy = state.phase === "uploading" || state.phase === "polling";
  const ready = isBoxValid(box);

  const submit = () => {
    if (!upload || !ready) return;
    setWorkerMissing(null);
    void start({ file: upload.file, bbox: box, topK });
  };

  return (
    <section className="space-y-6">
      {!upload ? (
        <ImageDropzone onAccept={acceptFile} onReject={setFileError} />
      ) : (
        <div className="space-y-5 rounded-2xl border border-surface-800 bg-surface-900/50 p-5">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="truncate text-sm text-ink-300">{upload.file.name}</p>
            <button
              type="button"
              onClick={clearUpload}
              className="rounded-lg px-3 py-1.5 text-sm text-ink-500 transition-colors hover:bg-surface-800 hover:text-ink-100"
            >
              Выбрать другое
            </button>
          </div>

          {/* key remounts on a new image, resetting size, drag ref and observer. */}
          <div className="flex justify-center">
            <BBoxEditor key={upload.url} src={upload.url} box={box} onChange={setBox} />
          </div>

          <div className="flex flex-wrap items-center justify-between gap-4 border-t border-surface-800 pt-4">
            <TopKControl value={topK} onChange={setTopK} disabled={busy} />

            <div className="flex items-center gap-3">
              {box && !ready && (
                <span className="text-sm text-danger-400">
                  Рамка меньше {MIN_SIDE_PX} px
                </span>
              )}
              <button
                type="button"
                onClick={submit}
                disabled={!ready || busy}
                className="rounded-xl bg-brand-600 px-5 py-2.5 font-medium text-white transition-colors hover:bg-brand-500 disabled:cursor-not-allowed disabled:bg-surface-800 disabled:text-ink-600"
              >
                {busy ? "Ищем…" : "Найти автомобиль"}
              </button>
            </div>
          </div>
        </div>
      )}

      {fileError && (
        <p className="rounded-xl border border-danger-400/40 bg-surface-900 px-5 py-4 text-sm text-danger-400">
          {fileError}
        </p>
      )}

      <StatusPanel state={state} workerMissing={workerMissing} />
    </section>
  );
}
