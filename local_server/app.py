import os
import json
import sqlite3
import time
from typing import List, Optional
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

DB_PATH = os.getenv("DB_PATH", "/Users/mac/Desktop/pet/local_server/data/app.db")
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "5"))
CONTEXT_TURNS = int(os.getenv("CONTEXT_TURNS", "8"))

app = FastAPI()


class ChatRequest(BaseModel):
    user_id: str
    text: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    memory_used: List[str]


def _db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            created_at INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            created_at INTEGER,
            last_active INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            type TEXT,
            content TEXT,
            confidence REAL,
            importance INTEGER DEFAULT 1,
            created_at INTEGER,
            updated_at INTEGER,
            source TEXT
        )
        """
    )
    _migrate_memories_table(conn)
    conn.commit()
    conn.close()


@app.on_event("startup")
def on_startup():
    _init_db()
    if DEEPSEEK_API_KEY:
        print(f"[LLM] Loaded DEEPSEEK_API_KEY (len={len(DEEPSEEK_API_KEY)})")
    else:
        print("[LLM WARN] DEEPSEEK_API_KEY is empty")


def _ensure_user(conn: sqlite3.Connection, user_id: str):
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE id=?", (user_id,))
    if cur.fetchone() is None:
        cur.execute("INSERT INTO users (id, created_at) VALUES (?, ?)", (user_id, int(time.time())))
        conn.commit()


def _get_or_create_session(conn: sqlite3.Connection, user_id: str, session_id: Optional[str]) -> str:
    cur = conn.cursor()
    if session_id:
        cur.execute("SELECT id FROM sessions WHERE id=? AND user_id=?", (session_id, user_id))
        if cur.fetchone() is not None:
            cur.execute("UPDATE sessions SET last_active=? WHERE id=?", (int(time.time()), session_id))
            conn.commit()
            return session_id

    new_id = f"s_{int(time.time()*1000)}_{user_id}"
    cur.execute(
        "INSERT INTO sessions (id, user_id, created_at, last_active) VALUES (?, ?, ?, ?)",
        (new_id, user_id, int(time.time()), int(time.time())),
    )
    conn.commit()
    return new_id


def _save_message(conn: sqlite3.Connection, session_id: str, role: str, content: str):
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, int(time.time())),
    )
    conn.commit()


def _get_recent_messages(conn: sqlite3.Connection, session_id: str, turns: int) -> List[dict]:
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, turns * 2),
    )
    rows = cur.fetchall()[::-1]
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def _get_recent_memories(conn: sqlite3.Connection, user_id: str, k: int) -> List[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT type, content FROM memories WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
        (user_id, k),
    )
    rows = cur.fetchall()
    return [f'{r["type"]}: {r["content"]}' if r["type"] else r["content"] for r in rows]


def _build_messages(user_text: str, context: List[dict], memories: List[str]) -> List[dict]:
    system = "You are a helpful voice assistant. Keep responses concise."
    if memories:
        memory_block = "\n".join(f"- {m}" for m in memories)
        system += f"\nUser memory:\n{memory_block}"
    msgs = [{"role": "system", "content": system}] + context + [{"role": "user", "content": user_text}]
    return msgs


def _call_deepseek(messages: List[dict]) -> str:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="Missing DEEPSEEK_API_KEY")

    # OpenAI-compatible client for DeepSeek
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DeepSeek error: {e}")

def _migrate_memories_table(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(memories)")
    cols = {row[1] for row in cur.fetchall()}
    if "type" not in cols:
        cur.execute("ALTER TABLE memories ADD COLUMN type TEXT")
    if "confidence" not in cols:
        cur.execute("ALTER TABLE memories ADD COLUMN confidence REAL")
    if "source" not in cols:
        cur.execute("ALTER TABLE memories ADD COLUMN source TEXT")
    conn.commit()


def _extract_memories_prompt(conversation: str) -> List[dict]:
    system = (
        "You are a memory extractor. Extract user information suitable for long-term memory and output JSON array. "
        "Allowed type: preference, fact, habit. Must include confidence (0-1). "
        "If no memory, return []. Do not record sensitive info "
        "(ID number, phone, bank card, precise address, medical diagnosis)."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": conversation},
    ]
    try:
        raw = _call_deepseek(messages)
    except HTTPException:
        return []
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`").strip()
        data = json.loads(cleaned)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _memory_is_allowed(item: dict, blacklist: List[str]) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get("type") not in {"preference", "fact", "habit"}:
        return False
    confidence = item.get("confidence")
    if not isinstance(confidence, (int, float)) or confidence < 0.7:
        return False
    content = item.get("content")
    if not isinstance(content, str) or not content.strip():
        return False
    lowered = content.lower()
    return all(word not in lowered for word in blacklist)


def _save_memories(conn: sqlite3.Connection, user_id: str, items: List[dict], source: str):
    blacklist = [
        "身份证",
        "手机号",
        "银行卡",
        "精确地址",
        "医疗诊断",
        "id number",
        "phone number",
        "bank card",
        "precise address",
        "medical diagnosis",
    ]
    now = int(time.time())
    for item in items:
        if not _memory_is_allowed(item, blacklist):
            continue
        conn.execute(
            """
            INSERT INTO memories (user_id, type, content, confidence, created_at, updated_at, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                item.get("type"),
                item.get("content").strip(),
                float(item.get("confidence")),
                now,
                now,
                source,
            ),
        )
    conn.commit()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conn = _db()
    try:
        _ensure_user(conn, req.user_id)
        session_id = _get_or_create_session(conn, req.user_id, req.session_id)

        _save_message(conn, session_id, "user", req.text)

        memories = _get_recent_memories(conn, req.user_id, MEMORY_TOP_K)
        context = _get_recent_messages(conn, session_id, CONTEXT_TURNS)

        messages = _build_messages(req.text, context, memories)
        reply = _call_deepseek(messages)

        _save_message(conn, session_id, "assistant", reply)
        conversation = f"User: {req.text}\nAssistant: {reply}"
        memory_items = _extract_memories_prompt(conversation)
        _save_memories(conn, req.user_id, memory_items, source="auto")
        return ChatResponse(reply=reply, session_id=session_id, memory_used=memories)
    finally:
        conn.close()


class MemoryAddRequest(BaseModel):
    user_id: str
    type: str
    content: str
    confidence: float = 0.9


@app.post("/memory/add")
def memory_add(req: MemoryAddRequest):
    conn = _db()
    _ensure_user(conn, req.user_id)
    now = int(time.time())
    conn.execute(
        """
        INSERT INTO memories (user_id, type, content, confidence, created_at, updated_at, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (req.user_id, req.type, req.content, req.confidence, now, now, "manual"),
    )
    conn.commit()
    conn.close()
    return {"ok": True}


@app.get("/memory/list")
def memory_list(user_id: str):
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, type, content, confidence, created_at, updated_at, source
        FROM memories WHERE user_id=? ORDER BY updated_at DESC
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "type": r["type"],
            "content": r["content"],
            "confidence": r["confidence"],
            "created_at": r["created_at"],
            "updated_at": r["updated_at"],
            "source": r["source"],
        }
        for r in rows
    ]


class MemoryDeleteRequest(BaseModel):
    user_id: str
    memory_id: int


@app.post("/memory/delete")
def memory_delete(req: MemoryDeleteRequest):
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM memories WHERE id=? AND user_id=?",
        (req.memory_id, req.user_id),
    )
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return {"ok": True, "deleted": deleted}
