#!/usr/bin/env python3
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Tuple

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
DB_PATH = os.getenv("DB_PATH", "/Users/mac/Desktop/pet/local_server/data/app.db")
USER_ID = os.getenv("TEST_USER_ID", f"qa_{int(time.time())}")
TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "5"))


def http_request(method: str, path: str, body: Dict[str, Any] | None = None) -> Tuple[int, Any]:
    url = BASE_URL.rstrip("/") + path
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw) if raw else None
        except Exception:
            return e.code, raw


def assert_true(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def check_migration_columns(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(memories)")
    cols = {row[1] for row in cur.fetchall()}
    for col in ("type", "confidence", "source"):
        assert_true(col in cols, f"missing column {col} in memories table")


def list_memories(user_id: str) -> List[Dict[str, Any]]:
    status, data = http_request("GET", f"/memory/list?user_id={urllib.parse.quote(user_id)}")
    assert_true(status == 200, f"list memories failed: {status} {data}")
    assert_true(isinstance(data, list), "list response is not a list")
    return data


def add_memory(user_id: str, mtype: str, content: str, confidence: float = 0.9) -> int:
    status, data = http_request(
        "POST",
        "/memory/add",
        {"user_id": user_id, "type": mtype, "content": content, "confidence": confidence},
    )
    assert_true(status == 200, f"add memory failed: {status} {data}")
    assert_true(data and data.get("ok") is True, f"add memory bad response: {data}")
    # fetch to get id
    mems = list_memories(user_id)
    assert_true(len(mems) > 0, "no memories after add")
    return mems[0]["id"]


def delete_memory(user_id: str, memory_id: int) -> Dict[str, Any]:
    status, data = http_request(
        "POST",
        "/memory/delete",
        {"user_id": user_id, "memory_id": memory_id},
    )
    assert_true(status == 200, f"delete memory failed: {status} {data}")
    return data


def check_ordering(conn: sqlite3.Connection, user_id: str):
    # Insert three records and then override updated_at for deterministic order.
    now = int(time.time())
    rows = []
    for i in range(3):
        conn.execute(
            """
            INSERT INTO memories (user_id, type, content, confidence, created_at, updated_at, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, "fact", f"ordering_{i}", 0.9, now, now, "manual"),
        )
        rows.append(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    conn.commit()

    # Set updated_at so rows[2] newest, rows[0] oldest
    conn.execute("UPDATE memories SET updated_at=? WHERE id=?", (now - 10, rows[0]))
    conn.execute("UPDATE memories SET updated_at=? WHERE id=?", (now - 5, rows[1]))
    conn.execute("UPDATE memories SET updated_at=? WHERE id=?", (now - 1, rows[2]))
    conn.commit()

    mems = list_memories(user_id)
    ids = [m["id"] for m in mems if m.get("content", "").startswith("ordering_")]
    assert_true(ids[:3] == [rows[2], rows[1], rows[0]], f"ordering mismatch: {ids[:3]} vs {rows}")


def main() -> int:
    print(f"BASE_URL={BASE_URL}")
    print(f"DB_PATH={DB_PATH}")
    print(f"TEST_USER_ID={USER_ID}")

    # DB checks
    try:
        conn = sqlite3.connect(DB_PATH)
    except Exception as e:
        print(f"[FAIL] cannot open DB: {e}")
        return 2

    try:
        check_migration_columns(conn)
        print("[PASS] migration columns present")
    except Exception as e:
        print(f"[FAIL] migration columns: {e}")
        return 2

    # API checks
    try:
        # list should work even if empty
        mems = list_memories(USER_ID)
        print(f"[PASS] list memories: {len(mems)} existing")

        mid = add_memory(USER_ID, "fact", "qa_memory_1", 0.9)
        print(f"[PASS] add memory id={mid}")

        # delete wrong user
        resp = delete_memory("other_user", mid)
        assert_true(resp.get("deleted") == 0, f"expected deleted=0 for wrong user, got {resp}")
        print("[PASS] delete wrong user blocked")

        # delete correct user
        resp = delete_memory(USER_ID, mid)
        assert_true(resp.get("deleted") == 1, f"expected deleted=1, got {resp}")
        print("[PASS] delete correct user")

        # ordering
        check_ordering(conn, USER_ID)
        print("[PASS] updated_at ordering DESC")

    except urllib.error.URLError as e:
        print(f"[FAIL] cannot reach API: {e}")
        return 2
    except Exception as e:
        print(f"[FAIL] api checks: {e}")
        return 2
    finally:
        conn.close()

    print("[OK] memory smoke tests completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
