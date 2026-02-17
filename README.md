## RAG поиск по базе знаний notion  для подготовки к собеседованию на Android Engineer  

## Задача

Построить MCP-сервер, который подключает Notion как внешний источник знаний к существующему RAG-пайплайну:
1. Чтение контента из Notion через API
2. Конвертация в чанки + генерация эмбеддингов
3. Хранение/обновление в локальном RAG-индексе
4. Семантический поиск и QA по контенту из Notion через MCP-инструменты

**Результат:** MCP-сервер с 3 инструментами — sync, search, ask — поверх RAG-памяти из Notion.

---

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Server (server.py)                │
│         Model Context Protocol через stdio              │
├──────────┬──────────────────┬──────────────────────────┤
│sync_notion│search_notion_memory│      ask_notion         │
└──────┬───┴─────────┬────────┴──────────┬───────────────┘
       │             │                   │
       ▼             ▼                   ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│ notion_sync  │ │ rag_adapter  │ │   rag_adapter    │
│  Full/Incr.  │ │  .search()   │ │   .ask()         │
└──────┬───────┘ └──────┬───────┘ └────────┬─────────┘
       │                │                  │
       ▼                └──────┬───────────┘
┌──────────────┐               ▼
│notion_client │ ┌───────────────────────────────────┐
│ Notion API   │ │     RAG Pipeline (indexer/)        │
│ httpx async  │ │  chunker → embeddings → retriever  │
└──────────────┘ │  → reranker → llm → citations      │
                 └───────────────────────────────────┘
```

### Файловая структура

```
├── server.py            # MCP-сервер (точка входа)
├── config.py            # Конфигурация: NotionConfig, SyncConfig, ServerConfig
├── notion_client.py     # Async клиент Notion API (страницы, базы, блоки)
├── notion_sync.py       # Логика синхронизации (full + incremental)
├── rag_adapter.py       # Адаптер: Notion-индекс → RAG-пайплайн
├── test_sync.py         # Тестовый скрипт (end-to-end)
├── requirements.txt     # Зависимости
├── main.py              # CLI для базового RAG (из дня 18)
└── indexer/
    ├── __init__.py      # Экспорты пакета
    ├── settings.py      # Конфигурация пайплайна
    ├── chunker.py       # Чанкинг текста с перекрытием
    ├── embeddings.py    # Провайдер эмбеддингов (OpenAI)
    ├── index.py         # JSON-индекс документов
    ├── pipeline.py      # Оркестратор индексации
    ├── retriever.py     # Косинусный поиск по эмбеддингам
    ├── llm.py           # Провайдер LLM (OpenAI)
    ├── reranker.py      # LLM-реранкинг чанков
    ├── rag.py           # RAG: baseline + filtered + cited
    └── citations.py     # Валидация цитат в ответах
```

---

## RAG-пайплайн

Используется кастомный RAG-пайплайн без внешних фреймворков (LangChain, LlamaIndex и т.д.) — все компоненты написаны вручную на чистом Python.

### Компоненты

| Этап | Модуль | Что делает |
|------|--------|------------|
| **Chunking** | `indexer/chunker.py` | Fixed-size чанки с перекрытием (800 символов, overlap 150). Детерминированные ID чанков через SHA-256 |
| **Embeddings** | `indexer/embeddings.py` | OpenAI `text-embedding-3-small` (1536 dims). Батчинг по 64, retry с экспоненциальным backoff |
| **Index** | `indexer/index.py` | Локальный JSON-индекс: документы → чанки → эмбеддинги. Без внешних БД |
| **Retrieval** | `indexer/retriever.py` | Brute-force cosine similarity (чистый Python, без numpy/FAISS). Top-k поиск по всем чанкам |
| **Reranking** | `indexer/reranker.py` | LLM-based reranking: GPT оценивает релевантность каждого чанка (score 0–1), пересортировка |
| **Generation** | `indexer/llm.py` | OpenAI `gpt-4o-mini`, temperature 0.3, max 1024 токенов |
| **Citations** | `indexer/citations.py` | Пост-валидация цитат: regex-извлечение chunk ID из ответа, проверка против allowed set |

### Три режима RAG

```
1. Baseline (answer_question)
   embed query → cosine search top-k → augmented prompt → LLM → ответ

2. Filtered (answer_question_filtered)
   embed query → cosine search → LLM reranking → threshold filter → fallback → LLM → RAGResult

3. Cited (answer_question_cited)
   embed query → cosine search → LLM reranking → threshold filter → cited prompt
   → LLM → citation validation → retry (до 2 попыток) → RAGResult + citations
```

### Threshold filtering и Fallback

- Чанки ниже порога `threshold` (по умолчанию 0.75) отсекаются
- Если все чанки ниже порога, работает fallback-стратегия:
  - `TOP_1` — возвращает лучший чанк с предупреждением
  - `INSUFFICIENT_CONTEXT` — возвращает пустой ответ

### Особенности

- **Без внешних зависимостей для ML**: cosine similarity на чистом Python, без numpy/scipy/FAISS
- **Детерминированные chunk ID**: SHA-256 хеш контента → стабильные ID при переиндексации
- **Incremental sync**: сравнение `last_edited_time` страниц Notion, переиндексация только изменённых
- **Context budget**: лимит ~12K символов (~3K токенов) на контекст, чанки обрезаются при переполнении
- **Citation enforcement**: LLM обязана ссылаться на chunk ID, при провале валидации — автоматический retry с более строгим промптом

---

## Установка

```bash
# Создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# Установить зависимости
pip install -r requirements.txt

# Установить API ключи
export NOTION_API_KEY="ntn_..."
export OPENAI_API_KEY="sk-..."
```

---

## MCP-инструменты

### 1. `sync_notion` — синхронизация Notion → RAG-индекс

Загружает страницы из Notion, разбивает на чанки, генерирует эмбеддинги и сохраняет в локальный индекс.

| Параметр | Тип | Описание |
|----------|-----|----------|
| `page_ids` | `string[]` | ID страниц Notion для синхронизации |
| `database_ids` | `string[]` | ID баз данных Notion |
| `mode` | `"full" \| "incremental"` | Полная пересборка или только изменения |

**Incremental sync** сравнивает `last_edited_time` каждой страницы с сохранённым состоянием и переиндексирует только изменённые.

### 2. `search_notion_memory` — семантический поиск

Ищет по индексированному контенту из Notion через косинусное сходство эмбеддингов.

| Параметр | Тип | Описание |
|----------|-----|----------|
| `query` | `string` | Поисковый запрос (обязательный) |
| `top_k` | `integer` | Макс. количество результатов (по умолчанию 5) |
| `threshold` | `number` | Мин. score 0.0–1.0 (по умолчанию 0.0) |

### 3. `ask_notion` — QA по контенту из Notion

Полный RAG-пайплайн: retrieve → filter → augment → LLM → ответ.

| Параметр | Тип | Описание |
|----------|-----|----------|
| `question` | `string` | Вопрос (обязательный) |
| `top_k` | `integer` | Кол-во чанков для контекста (по умолчанию 5) |
| `threshold` | `number` | Порог релевантности (по умолчанию 0.3) |
| `enforce_citations` | `boolean` | Требовать цитаты в ответе |

---

## Запуск

### MCP-сервер

```bash
python server.py
```

Сервер работает через stdio (Model Context Protocol). Для подключения к Claude Desktop добавить в `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "notion-rag-memory": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/path/to/notion_mcp_with_rag_memory",
      "env": {
        "NOTION_API_KEY": "ntn_...",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### Тестирование

```bash
# Запустить тесты с конкретной страницей
python test_sync.py <notion-page-id>

# Пример
python test_sync.py 1db33305-b568-80a0-b17e-f95c9374a246
```

Тест проверяет: Notion клиент → Full sync → Incremental sync → Search → Ask → Stats.

---

## Notion: настройка интеграции

1. Перейти на [notion.so/my-integrations](https://www.notion.so/my-integrations)
2. Создать новую интеграцию, скопировать API-ключ
3. Открыть нужную страницу в Notion → `...` → **Connections** → выбрать интеграцию
4. ID страницы — 32-символьная hex-строка из URL:
   `https://notion.so/My-Page-abc123def456` → `abc123def456`

---

## Как работает синхронизация

```
Notion API                          Локальный RAG
─────────                           ─────────────
Page → get_page()                   document_id = notion_{page_id}
     → get_block_children()         source = notion://{page_id}
     → extract_block_text()
     → plain text                   ↓
                                    TextChunker → chunks
                                    ↓
                                    OpenAI Embeddings → vectors
                                    ↓
                                    DocumentIndex → notion_index.json
                                    SyncState → notion_sync_state.json
```

### Поддерживаемые блоки Notion

- Параграфы, заголовки (H1/H2/H3)
- Списки (маркированные, нумерованные)
- Цитаты, callout, toggle
- Код (с языком)
- To-do (с чекбоксами)
- Таблицы, разделители, формулы
- Вложенные блоки (один уровень)

---

## Конфигурация

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `NOTION_API_KEY` | — | API-ключ интеграции Notion |
| `OPENAI_API_KEY` | — | API-ключ OpenAI |
| `index_path` | notion_index.json | Путь к индексу |
| `sync_state_path` | notion_sync_state.json | Путь к состоянию синхронизации |
| `chunk_size` | 800 | Размер чанка в символах |
| `chunk_overlap` | 150 | Перекрытие между чанками |
| `embedding model` | text-embedding-3-small | Модель эмбеддингов |
| `LLM model` | gpt-4o-mini | Модель для генерации ответов |

---

## Результаты тестирования

Синхронизация 10 страниц из Notion (конспекты по Android/Kotlin/CS):

| Метрика | Значение |
|---------|----------|
| Страниц синхронизировано | 10 |
| Чанков создано | 469 |
| Поиск "ООП" | top score 0.65 — правильная страница |
| Поиск "Kotlin дженерики" | top score 0.60 — правильная страница |
| QA "lateinit" | корректный ответ с 5 источниками |
| Incremental sync | 0 переиндексировано (без изменений) |

---

## Зависимости

- Python 3.10+
- `openai >= 1.0.0` — эмбеддинги и LLM
- `mcp >= 1.0.0` — Model Context Protocol SDK
- `httpx >= 0.27.0` — async HTTP клиент для Notion API
