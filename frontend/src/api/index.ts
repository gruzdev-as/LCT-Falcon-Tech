import { DEMO_MODE } from "../lib/env";
import { demoSearchApi } from "./demo";
import { checkHealth, fetchSearchStatus, submitSearch } from "./search";
import type { SearchInput, StatusResponse } from "./search";
import type { SearchAccepted } from "./types";

export interface SearchApi {
  submit(input: SearchInput, signal: AbortSignal): Promise<SearchAccepted>;
  status(taskId: string, signal: AbortSignal): Promise<StatusResponse>;
  health(signal?: AbortSignal): Promise<boolean>;
}

const realSearchApi: SearchApi = {
  submit: submitSearch,
  status: fetchSearchStatus,
  health: checkHealth,
};

/** The only place the demo switch applies; callers stay unaware of which is behind it. */
export const searchApi: SearchApi = DEMO_MODE ? demoSearchApi : realSearchApi;
