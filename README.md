# LCT Falcon Tech — визуальный поиск автомобиля по изображению

На вход — фотография и bbox одной машины. На выход — все изображения **этого же
физического автомобиля** из галереи, отсортированные по близости.

Поиск идёт **только по внешнему виду**: номера в датасете заблюрены и не должны
использоваться как признак ни прямо, ни косвенно.

## Как работает сервис

Четыре стадии. Этой терминологией пользуемся в коде, логах и в документации.

| Стадия | Что происходит |
| --- | --- |
| **Ingestion** | Приняли картинку и bbox, провалидировали, положили оригинал в хранилище |
| **Processing** | Кроп по bbox, модель, эмбеддинг фиксированной размерности (float32) |
| **Analysis** | Векторный поиск по галерее: косинус по L2-нормированным векторам |
| **Result** | Сортировка по убыванию близости, top-N — или пусто, если ниже порога |

**Rejection mode — часть контракта.** Если лучший кандидат не дотянул до порога,
возвращается пустой список и флаг `rejected`. Добивать выдачу слабыми совпадениями
нельзя: лучше честно сказать «не нашли».

## Компоненты

| Сервис | Роль |
| --- | --- |
| `backend/` | FastAPI. Приём, валидация, постановка задачи, отдача результата |
| `inference/` | Воркер. Кроп, эмбеддинг, поиск в Qdrant, запись результата |
| `init/` | Одноразовый bootstrap: артефакты, схема БД, наполнение хранилищ |
| `common/` | Общий код: контракты, конфиги, ORM-модели, клиенты Redis / Qdrant / S3 |
| `frontend/` | React + TypeScript (Vite) |
| `training/` | Ноутбуки и эксперименты, в прод не едут |

| Инфраструктура | Зачем | Версия |
| --- | --- | --- |
| Redis Streams | Шина задач между backend и inference | `7.4.11-alpine` |
| Qdrant | Векторная база: эмбеддинги галереи | `v1.19.1` |
| Postgres | Метаданные: галерея и история поисков | `17.6-alpine` |
| RustFS (S3) | Оригиналы изображений | `1.0.0-rc.6` |

Всё поднимается одним `docker compose`. Версии образов зафиксированы, `latest` не
используем.

## Поток запроса

Поиск асинхронный: API сразу отдаёт `task_id`, клиент опрашивает результат.

```
POST /api/v1/search   (multipart: file, bbox, top_k)
  ├─ валидация картинки и bbox
  ├─ оригинал в S3
  ├─ SET  task:{id} = pending           (TTL 1 час)
  ├─ XADD falcon:tasks  EmbeddingTask
  ├─ INSERT search_queries              (bbox, top_k, формат и размеры кадра)
  └─ 202 {"task_id": "..."}

        inference (consumer group inference-workers)
          ├─ читает объект из S3, режет по bbox
          ├─ эмбеддинг → поиск top-k в Qdrant
          ├─ SET result:{id} = SearchResult, DEL task:{id}
          └─ XACK

GET /api/v1/search/{task_id}
  ├─ есть result:{id}  → UPDATE search_queries + INSERT search_candidates (идемпотентно)
  │                    → 200 + кандидаты
  ├─ есть task:{id}    → 202 {"status": "processing"}
  └─ нет ничего        → 404
```

Ключи в Redis живут час, поэтому историю поиска фиксирует Postgres: строка запроса
пишется на POST, исход и кандидаты — на первом GET, который увидел результат.
Запись best effort: упавший Postgres стоит строки метаданных, но не поиска.

Два архитектурных решения, которые стоит знать:

- **Backend ничего не считает и не хранит состояния в процессе.** Ни модели, ни
  тензоров, ни кэшей в памяти, ни фоновых задач. Он создаёт задачу и читает ключ.
  Всё состояние — в Redis, Qdrant и S3, поэтому реплику можно перезапустить в любой
  момент.
- **Qdrant принадлежит inference.** Воркер сам делает векторный поиск и кладёт в
  результат уже готовый список кандидатов. Поэтому результат едет в ключ
  `result:{task_id}`, а не во второй стрим: стрим — это лог без произвольного
  доступа, а GET нужно достать одну задачу по id.

## Что уже готово

- [x] Контракты в `common/src/configs/schemas.py`: `BBox`, `EmbeddingTask`, `SearchResult`
- [x] Оба эндпоинта бэкенда, валидация картинки и bbox, единый обработчик ошибок
- [x] Клиенты Redis (async) и S3, общий `setup_logging`
- [x] `docker compose`: redis, qdrant, rustfs, backend, inference (2 реплики), frontend
- [x] Фронт: лендинг с загрузкой, рамкой по bbox и выдачей; nginx проксирует `/api/`
- [x] **`inference/` — воркер.** Строго по одной задаче, масштабируется репликами;
  кроп → эмбеддинг → Qdrant → `result:{id}`;
  подбирает задачи упавших реплик (`XAUTOCLAIM`), «ядовитые» после N попыток отдаёт как `failed`
- [ ] Загрузка галереи (`POST /gallery/images`) — без неё искать не по чему
- [ ] Модель: пока заглушка `stub` (детерминированный вектор от пикселей кропа), настоящая
  ReID-модель встаёт новой реализацией `Embedder`; калибровка порога
- [x] **Postgres + SQLAlchemy для метаданных.** `search_queries` и `search_candidates`
  пишет бэкенд: запрос на POST, исход и кандидаты — идемпотентно на первом GET
- [ ] Метрики качества поверх этой истории

## Как запустить

```bash
uv sync                                          # dev + common + backend
uv sync --group inference                        # + torch, только если нужен
cp .env.example .env

docker compose up -d redis qdrant postgres rustfs
docker compose run --rm init                     # схема БД и наполнение хранилищ
uv run uvicorn backend.app.server:app --reload
uv run python -m inference.app.main              # воркер; можно запустить несколько

cd frontend && npm install && npm run dev        # :5173, проксирует /api на :8000
```

Либо всё сразу в контейнерах:

```bash
docker compose up --build -d                     # фронт на :8080
docker compose up -d --scale inference=4         # больше воркеров — одна consumer group
```

Галерея пока пустая, поэтому любой поиск честно заканчивается `rejected: true`.
Чтобы посмотреть выдачу без бэкенда:

```bash
VITE_DEMO_MODE=true npm run dev                  # ?demo=found|rejected|failed|timeout
```

### Куда заходить

| Что | Адрес |
| --- | --- |
| **Фронтенд** | http://localhost:8080 (в компоузе) или http://localhost:5173 (dev) |
| **Swagger UI** | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |
| Схема OpenAPI | http://localhost:8000/openapi.json |
| Консоль RustFS | http://localhost:9001 (`rustfsadmin` / `rustfsadmin`) |
| Дашборд Qdrant | http://localhost:6333/dashboard |

Из Swagger `POST /api/v1/search` дёргается руками: файл выбирается через диалог, в
поле `bbox` вставляется JSON-строка вида `{"x":10,"y":10,"width":200,"height":150}`.

Проверить через curl:

```bash
curl -F file=@car.jpg \
     -F 'bbox={"x":10,"y":10,"width":200,"height":150}' \
     localhost:8000/api/v1/search                 # → 202 {"task_id": "..."}

curl -i localhost:8000/api/v1/search/<task_id>    # → 202, потом 200
```

Пока воркер работает, GET отвечает `202`; как только результат записан — `200`.
Если воркеры не запущены, задача ждёт в стриме и GET остаётся на `202`.

## Договорённости

- **Python 3.12, uv.** Один корневой `pyproject.toml`, одна `.venv`, одна `uv.lock`.
  У каждого сервиса своя группа в `[dependency-groups]`.
- **ruff** для линта и формата, длина строки 120. Гонять перед коммитом:
  `uv run ruff check --fix . && uv run ruff format .`
- **Раскладка сервиса:** `app/` — транспорт (FastAPI, роутеры), `src/` — всё
  остальное. `app/` может импортировать `src/`, наоборот — нет.
- **Postgres только через SQLAlchemy 2.0**, без сырого SQL и курсоров драйвера.
- Всё, что нужно двум сервисам, живёт в `common/` — один раз, без дублей.

Подробности по каждому сервису — в `<service>/.claude/CLAUDE.md`; корневой
`.claude/CLAUDE.md` описывает правила, общие для всего репозитория.
