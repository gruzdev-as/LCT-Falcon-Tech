import { useRef, useState, type ChangeEvent, type DragEvent } from "react";

import { ALLOWED_IMAGE_TYPES, MAX_IMAGE_BYTES } from "../lib/constants";
import { formatBytes, validateFile } from "../lib/file";

interface Props {
  onAccept: (file: File) => void;
  onReject: (message: string) => void;
}

/** Hands the raw File straight through: the bytes dropped are the bytes uploaded. */
export function ImageDropzone({ onAccept, onReject }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const take = (file: File | undefined) => {
    if (!file) return;
    const problem = validateFile(file);
    if (problem) {
      onReject(problem);
      return;
    }
    onAccept(file);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    take(event.dataTransfer.files[0]);
  };

  const handlePick = (event: ChangeEvent<HTMLInputElement>) => {
    take(event.target.files?.[0]);
    // so picking the same file twice still fires a change event
    event.target.value = "";
  };

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-16 text-center transition-colors ${
        dragging
          ? "border-brand-400 bg-brand-500/10"
          : "border-surface-700 bg-surface-900/50 hover:border-brand-500/60 hover:bg-surface-900"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ALLOWED_IMAGE_TYPES.join(",")}
        onChange={handlePick}
        className="hidden"
      />

      <div className="mb-4 flex size-12 items-center justify-center rounded-xl bg-brand-500/15 text-brand-300">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className="size-6">
          <path d="M12 16V4m0 0L8 8m4-4 4 4" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" strokeLinecap="round" />
        </svg>
      </div>

      <p className="text-base font-medium text-ink-100">
        Перетащите фотографию или нажмите, чтобы выбрать
      </p>
      <p className="mt-2 text-sm text-ink-500">
        JPEG, PNG, WebP или BMP · до {formatBytes(MAX_IMAGE_BYTES)}
      </p>
    </div>
  );
}
