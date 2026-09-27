const STEPS = [
  {
    title: "Кадр и рамка",
    text: "Загрузите изображение и выделите на нём автомобиль.",
  },
  {
    title: "Эмбеддинг",
    text: "Нейросеть EVA-02 переводит вырезанную машину в вектор, который описывает её внешний вид.",
  },
  {
    title: "Поиск и проверка",
    text: "Ближайшие кадры находятся в векторной базе, затем отдельный классификатор решает, та ли это машина. Ответ: кандидаты по убыванию сходства или отказ.",
  },
];

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

        <p className="mt-6 max-w-2xl text-lg leading-relaxed text-pretty text-ink-300">
          Загрузите фото и обведите рамкой нужную машину. Сервис найдёт в галерее все кадры
          этого же автомобиля. Если уверенного совпадения нет или такого автомобиля нет в базе
          данных — мы сообщим об этом.
        </p>

        <ol className="mt-10 grid gap-4 sm:grid-cols-3">
          {STEPS.map((step, index) => (
            <li
              key={step.title}
              className="rounded-xl border border-surface-800 bg-surface-900/50 px-5 py-4"
            >
              <p className="flex items-baseline gap-2 font-medium text-ink-100">
                <span className="text-sm tabular-nums text-brand-400">{index + 1}</span>
                {step.title}
              </p>
              <p className="mt-2 text-sm leading-relaxed text-ink-500">{step.text}</p>
            </li>
          ))}
        </ol>
      </div>
    </header>
  );
}
