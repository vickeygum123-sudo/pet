import asyncio
import base64
import hashlib
import hmac
import os
import random
import ssl
import time
import urllib.parse
import uuid
import json
import ast
import re
from typing import Dict
from pathlib import Path

import websockets
from dotenv import load_dotenv
import certifi
import httpx

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

TENCENT_APPID = os.getenv("TENCENT_APPID", "")
TENCENT_SECRET_ID = os.getenv("TENCENT_SECRET_ID", "")
TENCENT_SECRET_KEY = os.getenv("TENCENT_SECRET_KEY", "")

ENGINE_MODEL_TYPE = os.getenv("ENGINE_MODEL_TYPE", "16k_zh")
VOICE_FORMAT = os.getenv("VOICE_FORMAT", "1")  # 1: PCM
NEED_VAD = os.getenv("NEED_VAD", "1")
VAD_SILENCE_TIME = os.getenv("VAD_SILENCE_TIME", "800")
FILTER_PUNC = os.getenv("FILTER_PUNC", "1")
ALLOW_INSECURE_SSL = os.getenv("ALLOW_INSECURE_SSL", "0")
DEBUG_SIGN = os.getenv("DEBUG_SIGN", "0")
PRINT_RAW = os.getenv("PRINT_RAW", "0")

LOCAL_WS_HOST = os.getenv("LOCAL_WS_HOST", "0.0.0.0")
LOCAL_WS_PORT = int(os.getenv("LOCAL_WS_PORT", "8765"))
LLM_HTTP_URL = os.getenv("LLM_HTTP_URL", "http://127.0.0.1:8000/chat")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))


class ConfigError(Exception):
    pass


def _extract_final_text(message: str):
    # handle JSON, python-dict string, and simple regex
    try:
        payload = json.loads(message)
        if isinstance(payload, dict):
            result = payload.get("result")
            if isinstance(result, dict) and result.get("final", 0) == 1:
                return result.get("voice_text_str", "")
            if payload.get("slice_type") == 2 and "voice_text_str" in payload:
                return payload.get("voice_text_str", "")
    except Exception:
        pass
    try:
        payload = ast.literal_eval(message)
        if isinstance(payload, dict):
            if payload.get("slice_type") == 2 and "voice_text_str" in payload:
                return payload.get("voice_text_str", "")
    except Exception:
        pass
    try:
        m_text = re.search(r"[\"']voice_text_str[\"']\s*:\s*[\"']([^\"']*)[\"']", message)
        m_slice = re.search(r"[\"']slice_type[\"']\s*:\s*(\d+)", message)
        if m_text and m_slice and m_slice.group(1) == "2":
            return m_text.group(1)
    except Exception:
        pass
    return ""


def _require_env():
    if not TENCENT_APPID or not TENCENT_SECRET_ID or not TENCENT_SECRET_KEY:
        raise ConfigError(
            "Missing Tencent credentials. Set TENCENT_APPID, TENCENT_SECRET_ID, TENCENT_SECRET_KEY."
        )


def _build_query_params() -> Dict[str, str]:
    timestamp = int(time.time())
    expired = timestamp + 3600
    nonce = random.randint(1, 10**10)
    voice_id = uuid.uuid4().hex[:16]

    params = {
        "secretid": TENCENT_SECRET_ID,
        "timestamp": str(timestamp),
        "expired": str(expired),
        "nonce": str(nonce),
        "engine_model_type": ENGINE_MODEL_TYPE,
        "voice_id": voice_id,
        "voice_format": VOICE_FORMAT,
        "needvad": NEED_VAD,
        "vad_silence_time": VAD_SILENCE_TIME,
        "filter_punc": FILTER_PUNC,
    }
    return params


def _sign(params: Dict[str, str]) -> str:
    sorted_items = sorted(params.items())
    query = "&".join(f"{k}={v}" for k, v in sorted_items)
    base_url = f"asr.cloud.tencent.com/asr/v2/{TENCENT_APPID}"
    sign_str = f"{base_url}?{query}"
    if DEBUG_SIGN == "1":
        print("SIGN_BASE:", sign_str)
    digest = hmac.new(TENCENT_SECRET_KEY.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha1).digest()
    signature = base64.b64encode(digest).decode("utf-8")
    return signature


def _build_ws_url() -> str:
    params = _build_query_params()
    signature = _sign(params)
    params["signature"] = signature
    query = "&".join(
        f"{urllib.parse.quote(str(k), safe='')}={urllib.parse.quote(str(v), safe='')}"
        for k, v in sorted(params.items())
    )
    if DEBUG_SIGN == "1":
        print("WS_URL:", f"wss://asr.cloud.tencent.com/asr/v2/{TENCENT_APPID}?{query}")
    return f"wss://asr.cloud.tencent.com/asr/v2/{TENCENT_APPID}?{query}"


async def proxy_session(local_ws: websockets.WebSocketServerProtocol):
    _require_env()
    ws_url = _build_ws_url()

    ssl_context = ssl.create_default_context(cafile=certifi.where())
    if ALLOW_INSECURE_SSL == "1":
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

    async with websockets.connect(
        ws_url,
        ping_interval=10,
        ping_timeout=20,
        max_size=None,
        ssl=ssl_context,
    ) as remote_ws:
        async def forward_local_to_remote():
            try:
                async for message in local_ws:
                    await remote_ws.send(message)
            finally:
                await remote_ws.close()

        async def forward_remote_to_local():
            try:
                async for message in remote_ws:
                    await local_ws.send(message)
                    if isinstance(message, str):
                        if PRINT_RAW == "1":
                            print(message)
                        final_text = _extract_final_text(message)
                        if final_text:
                            print(final_text)
                            try:
                                async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
                                    payload = {
                                        "user_id": "device",
                                        "text": final_text,
                                        "session_id": None,
                                    }
                                    resp = await client.post(LLM_HTTP_URL, json=payload)
                                    if resp.status_code == 200:
                                        reply = resp.json().get("reply", "")
                                        if reply:
                                            await local_ws.send(reply)
                                    else:
                                        print(f"[LLM ERROR] {resp.status_code} {resp.text}")
                            except Exception as e:
                                print(f"[LLM EXCEPTION] {e}")
            except websockets.ConnectionClosed:
                pass
            finally:
                await local_ws.close()

        await asyncio.gather(forward_local_to_remote(), forward_remote_to_local())


async def main():
    print(f"Local WS listening on ws://{LOCAL_WS_HOST}:{LOCAL_WS_PORT}")
    async with websockets.serve(proxy_session, LOCAL_WS_HOST, LOCAL_WS_PORT, max_size=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
