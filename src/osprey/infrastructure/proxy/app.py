"""FastAPI application that translates Anthropic Messages API to OpenAI Chat Completions.

Claude Code sends requests here (via ANTHROPIC_BASE_URL); the proxy translates
and forwards to the real OpenAI-compatible upstream provider.
"""

from __future__ import annotations

import json
import logging

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from osprey.infrastructure.proxy.translator import (
    _FINISH_REASON_MAP,
    _gen_id,
    anthropic_to_openai_request,
    format_sse,
    make_content_block_start,
    make_content_block_stop,
    make_message_delta,
    make_message_start,
    make_message_stop,
    make_text_delta,
    make_tool_input_delta,
    openai_to_anthropic_response,
)

logger = logging.getLogger("osprey.infrastructure.proxy")


def create_proxy_app(
    upstream_base_url: str,
    upstream_api_key: str | None = None,
) -> FastAPI:
    """Create the translation proxy FastAPI app.

    Args:
        upstream_base_url: OpenAI-compatible endpoint (e.g. https://aiapi-prod.stanford.edu/v1).
        upstream_api_key: API key for the upstream provider.
    """
    app = FastAPI(title="osprey-proxy", docs_url=None, redoc_url=None)

    @app.get("/health")
    async def health():
        return {"status": "ok", "upstream": upstream_base_url}

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        model = body.get("model", "")
        is_stream = body.get("stream", False)

        # Extract auth — prefer upstream_api_key, fall back to request headers
        api_key = upstream_api_key
        if not api_key:
            api_key = request.headers.get("x-api-key") or ""
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]

        # Translate request
        openai_body = anthropic_to_openai_request(body)

        # Build upstream URL and headers
        url = upstream_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        if is_stream:
            return StreamingResponse(
                _stream_proxy(url, headers, openai_body, model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            async with httpx.AsyncClient(timeout=300.0) as client:
                try:
                    resp = await client.post(url, json=openai_body, headers=headers)
                    resp.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    return _translate_error(exc.response)
                except httpx.RequestError as exc:
                    return JSONResponse(
                        {"type": "error", "error": {"type": "api_error", "message": str(exc)}},
                        status_code=502,
                    )

            anthropic_resp = openai_to_anthropic_response(resp.json(), model)
            return JSONResponse(anthropic_resp)

    return app


async def _stream_proxy(
    url: str,
    headers: dict,
    openai_body: dict,
    model: str,
):
    """Forward streaming request and translate OpenAI SSE to Anthropic SSE."""
    msg_id = _gen_id("msg_")
    block_index = 0
    in_text = False
    tool_blocks: dict[int, dict] = {}  # openai tool_call index → {block_index, id, name}
    output_tokens = 0

    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            async with client.stream("POST", url, json=openai_body, headers=headers) as resp:
                if resp.status_code != 200:
                    error_body = b""
                    async for chunk in resp.aiter_bytes():
                        error_body += chunk
                    yield format_sse(
                        "error",
                        {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": error_body.decode(errors="replace"),
                            },
                        },
                    )
                    return

                # Emit message_start
                yield make_message_start(model, msg_id)

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta", {})
                    finish = choice.get("finish_reason")

                    # Track usage if provided
                    usage = chunk.get("usage")
                    if usage and "completion_tokens" in usage:
                        output_tokens = usage["completion_tokens"]

                    # Text content
                    text = delta.get("content")
                    if text:
                        if not in_text:
                            yield make_content_block_start(block_index, "text")
                            in_text = True
                        yield make_text_delta(block_index, text)

                    # Tool calls
                    for tc_delta in delta.get("tool_calls") or []:
                        tc_idx = tc_delta.get("index", 0)
                        func = tc_delta.get("function", {})

                        if tc_idx not in tool_blocks:
                            # New tool call — close text block if open
                            if in_text:
                                yield make_content_block_stop(block_index)
                                block_index += 1
                                in_text = False

                            tool_id = tc_delta.get("id", _gen_id("toolu_"))
                            tool_name = func.get("name", "")
                            tool_blocks[tc_idx] = {
                                "block_index": block_index,
                                "id": tool_id,
                                "name": tool_name,
                            }
                            yield make_content_block_start(
                                block_index,
                                "tool_use",
                                tool_id=tool_id,
                                tool_name=tool_name,
                            )
                            block_index += 1

                        # Argument fragment
                        args_fragment = func.get("arguments", "")
                        if args_fragment:
                            tb = tool_blocks[tc_idx]
                            yield make_tool_input_delta(tb["block_index"], args_fragment)

                    # Finish reason
                    if finish:
                        # Close any open blocks
                        if in_text:
                            yield make_content_block_stop(block_index)
                            in_text = False
                        for tb in tool_blocks.values():
                            yield make_content_block_stop(tb["block_index"])

                        stop_reason = _FINISH_REASON_MAP.get(finish, "end_turn")
                        yield make_message_delta(stop_reason, output_tokens)
                        yield make_message_stop()
                        return

                # Stream ended without explicit finish_reason
                if in_text:
                    yield make_content_block_stop(block_index)
                for tb in tool_blocks.values():
                    yield make_content_block_stop(tb["block_index"])
                yield make_message_delta("end_turn", output_tokens)
                yield make_message_stop()

        except httpx.RequestError as exc:
            yield format_sse(
                "error",
                {
                    "type": "error",
                    "error": {"type": "api_error", "message": str(exc)},
                },
            )


def _translate_error(response: httpx.Response) -> JSONResponse:
    """Translate an upstream HTTP error to Anthropic error format."""
    try:
        body = response.json()
        message = body.get("error", {}).get("message", response.text)
    except Exception:
        message = response.text

    error_type = "api_error"
    if response.status_code == 401:
        error_type = "authentication_error"
    elif response.status_code == 429:
        error_type = "rate_limit_error"
    elif response.status_code == 404:
        error_type = "not_found_error"

    return JSONResponse(
        {"type": "error", "error": {"type": error_type, "message": message}},
        status_code=response.status_code,
    )
