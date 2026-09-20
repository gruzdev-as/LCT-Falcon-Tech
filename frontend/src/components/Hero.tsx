export function Hero() {
  return (
    <header className="relative overflow-hidden border-b border-surface-800">
      <div
        aria-hidden
        className="pointer-events-none absolute -top-40 left-1/2 h-[28rem] w-[56rem] -translate-x-1/2 rounded-full bg-brand-600/20 blur-[120px]"
      />

      <div className="relative mx-auto max-w-5xl px-6 pt-20 pb-14 sm:pt-28 sm:pb-20">
        <p className="mb-5 inline-flex items-center gap-2 rounded-full border border-brand-500/30 bg-brand-500/10 px-3 py-1 text-xs font-medium text-brand-300">
          <span className="size-1.5 rounded-full bg-brand-400" />
          Визуальная реидентификация
        </p>

        <h1 className="text-4xl font-semibold tracking-tight text-balance sm:text-6xl">
          Найти этот же автомобиль
          <span className="block text-brand-400">по одному кадру</span>
        </h1>

        <p className="mt-6 max-w-2xl text-lg leading-relaxed text-pretty text-ink-300">
          Загрузите фотографию, выделите машину рамкой — сервис найдёт все изображения{" "}
          <em className="text-ink-100 not-italic">того же физического автомобиля</em> в
          галерее и отсортирует их по близости.
        </p>

        <p className="mt-4 max-w-2xl text-sm leading-relaxed text-ink-500">
          Поиск идёт только по внешнему виду. Номера в датасете заблюрены и не
          используются как признак — ни прямо, ни косвенно.
        </p>
      </div>
    </header>
  );
}
