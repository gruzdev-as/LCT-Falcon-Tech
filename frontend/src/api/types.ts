// Mirrors common/src/configs/schemas.py. The Python side is the source of truth —
// do not rename fields or add an alias layer.

export type TaskStatus = "pending" | "processing" | "done" | "failed";

/** Absolute pixels of the ORIGINAL image, xywh from the top-left corner. */
export interface BBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Candidate {
  image_id: string;
  score: number;
  rank: number;
  image_path: string | null;
  /** Presigned URL. Expires after S3_PRESIGN_TTL. */
  image_url: string | null;
  /**
   * Where the vehicle sits inside that image. The gallery stores whole frames, exactly
   * like an uploaded query, so showing a candidate means cropping by this.
   */
  bbox: BBox | null;
  vehicle_id: string | null;
  camera_id: string | null;
}

export interface SearchResult {
  task_id: string;
  status: TaskStatus;
  candidates: Candidate[];
  top_score: number | null;
  /** True when the best score fell below the threshold. A correct answer, not an error. */
  rejected: boolean;
  model_name: string | null;
  latency_ms: number | null;
  error: string | null;
  finished_at: string;
}

export interface SearchAccepted {
  task_id: string;
  status: TaskStatus;
}

export interface SearchPending {
  task_id: string;
  status: TaskStatus;
}

export interface ErrorResponse {
  error: string;
  message: string;
  details: Record<string, unknown>;
}
