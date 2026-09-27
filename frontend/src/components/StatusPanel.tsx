import type { SearchState } from "../lib/useSearchTask";
import { POLL_TIMEOUT_MS } from "../lib/constants";
import { CandidateCard } from "./CandidateCard";

function Notice({
  tone,
  title,
  children,
}: {
  tone: "neutral" | "danger";
  title: string;
  children?: React.ReactNode;
}) {
  const ring = tone === "danger" ? "border-danger-400/40" : "border-surface-700";
  return (
    <div className={`rounded-xl border ${ring} bg-surface-900 px-5 py-6`}>
      <p className="font-medium text-ink-100">{title}</p>
      {children && <div className="mt-2 text-sm leading-relaxed text-ink-500">{children}</div>}
    </div>
  );
}

/** Exhaustive over SearchState: a phase without a screen is a type error. */
export function StatusPanel({
  state,
  workerMissing,
}: {
  state: SearchState;
  workerMissing: boolean | null;
}) {
  switch (state.phase) {
    // Holds the results column open so the page does not look half-empty before a search.
    case "idle":
      return (
        <div className="flex min-h-80 flex-col items-center justify-center self-stretch rounded-2xl border border-surface-800 bg-surface-900/50 px-6 py-12 text-center">
          <p className="font-medium text-ink-300">Здесь появятся найденные кадры</p>
          <p className="mt-2 max-w-sm text-sm leading-relaxed text-ink-600">
            Загрузите фото, выделите автомобиль и запустите поиск — кандидаты выстроятся по
            убыванию сходства.
          </p>
        </div>
      );

    case "uploading":
      return <Notice tone="neutral" title="Отправляем изображение…" />;

    case "polling": {
      const seconds = Math.floor(state.elapsedMs / 1000);
      const budget = Math.floor(POLL_TIMEOUT_MS / 1000);
      return (
        <Notice tone="neutral" title="Ищем совпадения…">
          Задача {state.taskId.slice(0, 8)} в очереди · {seconds} из {budget} с
        </Notice>
      );
    }

    case "found":
      return (
        <div className="space-y-5 rounded-2xl border border-surface-800 bg-surface-900/50 p-5">
          <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-surface-800 pb-4">
            <h2 className="text-lg font-semibold text-ink-100">
              Найдено: {state.result.candidates.length}
            </h2>
            <p className="text-sm text-ink-500">
              лучшее совпадение{" "}
              <span className="font-medium text-brand-300 tabular-nums">
                {((state.result.top_score ?? 0) * 100).toFixed(1)}%
              </span>
              {state.result.latency_ms !== null && (
                <> · {state.result.latency_ms.toFixed(0)} мс</>
              )}
            </p>
          </div>
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-2 xl:grid-cols-3">
            {state.result.candidates.map((candidate) => (
              <CandidateCard key={candidate.image_id} candidate={candidate} />
            ))}
          </div>
        </div>
      );

    // Rejection is a correct answer: calm and neutral, never red.
    case "rejected":
      return (
        <Notice tone="neutral" title="Такого автомобиля нет в галерее">
          {state.result.top_score !== null ? (
            <>
              Самый похожий кадр —{" "}
              <span className="font-medium text-ink-300 tabular-nums">
                {(state.result.top_score * 100).toFixed(1)}%
              </span>
              , но проверка совпадения не подтвердила, что это та же машина.
            </>
          ) : (
            "Галерея пока пуста."
          )}
        </Notice>
      );

    case "failed":
      return (
        <Notice tone="danger" title="Поиск завершился ошибкой">
          <span className="break-words whitespace-pre-line">{state.message}</span>
        </Notice>
      );

    case "expired":
      return (
        <Notice tone="danger" title="Задача не найдена">
          Результат по задаче {state.taskId.slice(0, 8)} истёк или был удалён. Запустите
          поиск заново.
        </Notice>
      );

    case "timedOut":
      return (
        <Notice tone="danger" title="Ответа не дождались">
          {workerMissing === true
            ? "API доступен, но задачу никто не взял в работу — воркер inference/ не запущен."
            : workerMissing === false
              ? "API недоступен. Проверьте, что бэкенд запущен."
              : "Задача принята, но результат не пришёл за отведённое время."}
        </Notice>
      );

    case "error":
      return (
        <Notice tone="danger" title="Запрос не удался">
          <span className="break-words whitespace-pre-line">{state.error.message}</span>
        </Notice>
      );
  }
}
