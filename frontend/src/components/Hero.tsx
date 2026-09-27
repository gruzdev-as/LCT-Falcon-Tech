export function Hero() {
  return (
    <header className="relative overflow-hidden border-b border-surface-800">
      <div
        aria-hidden
        className="pointer-events-none absolute -top-40 left-1/2 h-[28rem] w-[56rem] -translate-x-1/2 rounded-full bg-brand-600/20 blur-[120px]"
      />

      <div className="relative mx-auto max-w-7xl px-6 pt-20 pb-14 sm:pt-28 sm:pb-20">
        <h1 className="text-4xl font-semibold tracking-tight text-balance sm:text-6xl">
          Найти этот же автомобиль
          <span className="block text-brand-400">по одному кадру</span>
        </h1>
      </div>
    </header>
  );
}
