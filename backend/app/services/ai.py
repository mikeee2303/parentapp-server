import json
import os

import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")


async def stream_completion(
    messages: list,
    timeout: int = 300,
) -> StreamingResponse:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.5},
    }

    async def generate():
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json=payload,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")

                        if content:
                            event = {
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None,
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                        if chunk.get("done"):
                            yield (
                                'data: {"choices":[{"index":0,"delta":{},'
                                '"finish_reason":"stop"}]}\n\n'
                            )
                            yield "data: [DONE]\n\n"

        except httpx.HTTPError as exc:
            # Der HTTP-Stream hat zu diesem Zeitpunkt eventuell schon begonnen.
            # Details des Fehlers erscheinen auch in den Backend-Logs.
            error_event = {"error": f"Ollama request failed: {exc}"}
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def get_completion(messages: list, timeout: int = 300) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.5},
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {exc}",
        ) from exc

