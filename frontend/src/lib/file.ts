import { ALLOWED_IMAGE_TYPES, MAX_IMAGE_BYTES } from "./constants";

/**
 * Metadata-only checks: this reads two properties and decodes nothing. Anything that
 * needs the pixels is the backend's job.
 *
 * @returns an error message, or null when the file looks acceptable.
 */
export function validateFile(file: File): string | null {
  if (file.size === 0) {
    return "Файл пустой";
  }

  if (file.size > MAX_IMAGE_BYTES) {
    return `Файл больше ${formatBytes(MAX_IMAGE_BYTES)} — ${formatBytes(file.size)}`;
  }

  // An empty type happens with some file managers; let the server sniff the format
  // rather than refusing something that may be fine.
  if (
    file.type &&
    !ALLOWED_IMAGE_TYPES.includes(file.type as (typeof ALLOWED_IMAGE_TYPES)[number])
  ) {
    return `Формат не поддерживается: ${file.type}. Нужен JPEG, PNG, WebP или BMP`;
  }

  return null;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} КБ`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} МБ`;
}
