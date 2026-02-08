# Local LLM Gateway (DeepSeek + SQLite)

This is a local HTTP gateway for DeepSeek with session context and long-term memory stored in SQLite. It is designed to be migrated to a server later without code changes.

## Setup

```bash
python3 -m venv /Users/mac/Desktop/pet/local_server/.venv
source /Users/mac/Desktop/pet/local_server/.venv/bin/activate
pip install -r /Users/mac/Desktop/pet/local_server/requirements.txt
```

## Configure

Edit `/Users/mac/Desktop/pet/local_server/.env` and set:

```
DEEPSEEK_API_KEY=YOUR_DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DB_PATH=/Users/mac/Desktop/pet/local_server/data/app.db
MEMORY_TOP_K=5
CONTEXT_TURNS=8
```

## Run

```bash
source /Users/mac/Desktop/pet/local_server/.venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API

### POST /chat

Request:
```json
{
  "user_id": "u123",
  "text": "你好",
  "session_id": null
}
```

Response:
```json
{
  "reply": "你好，有什么可以帮你？",
  "session_id": "s_...",
  "memory_used": []
}
```

### POST /memory/add

Request:
```json
{
  "user_id": "u123",
  "content": "用户喜欢喝拿铁",
  "importance": 2
}
```

Response:
```json
{ "ok": true }
```

## Notes

- Memories are currently selected by most recent `updated_at`. You can later swap to embeddings/vector search without changing the API.
