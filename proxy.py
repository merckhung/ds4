import httpx
import json
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

app = FastAPI()

# Target port for local DS4 server
TARGET_URL = "http://127.0.0.1:1235"
# SECURITY: Move this to environment variable in production
# Rotate credentials periodically: DS4_AUTH_TOKEN=...
AUTH_TOKEN = os.environ.get("DS4_AUTH_TOKEN", "sk-lm-4j2UDSac:al9IOBz73vaUFdSbb0sE")

# Public model ID clients use. ds4-server accepts local-llm natively (and still
# accepts deepseek-v4-flash / deepseek-v4-pro as aliases). Keep both ends on
# local-llm so /v1/models and response bodies report one id.
PUBLIC_MODEL_ID = os.environ.get("DS4_PUBLIC_MODEL_ID", "local-llm")
BACKEND_MODEL = os.environ.get("DS4_BACKEND_MODEL", "local-llm")

# Thinking mode: kept ON but bounded. Unbounded reasoning at large context
# turns the first turn into a multi-minute pre-answer stall that outlives the
# client's patience (observed: 189k-token prefill + open-ended THINKING ->
# "client stream write failed"). We (1) strip prior reasoning blocks from the
# history so the prompt stops ballooning each turn, and (2) cap budget_tokens.
# Override the cap with DS4_THINKING_BUDGET (0 disables the cap entirely).
#
# Per DeepSeek official encoding:
# - Without tools: drop earlier reasoning blocks (done by strip_prior_reasoning)
# - With tools: preserve reasoning required by the tool chain
THINKING_BUDGET = int(os.environ.get("DS4_THINKING_BUDGET", "2048"))

# Auxiliary request routing: detect small helper requests (title, classification,
# routing, summarization) and route them with minimal thinking budget.
# These should not pay the full agent reasoning path.
AUXILIARY_PATTERNS = [
    "title", "summarize", "summarisation", "classification", "classify",
    "routing", "route", "extract", "extraction", "parse", "parsing",
    "short answer", "brief", "one line", "yes no", "yes-or-no",
]
AUXILIARY_BUDGET = int(os.environ.get("DS4_AUXILIARY_BUDGET", "128"))


def strip_prior_reasoning(messages):
    """Remove thinking / redacted_thinking content blocks from prior messages.

    Per DeepSeek official encoding:
    - Without tools: drop earlier reasoning blocks to save prompt space
    - With tools: preserve reasoning required by the tool chain

    Returns True if anything was stripped."""
    changed = False
    if not isinstance(messages, list):
        return changed

    # Check if this conversation has tool usage
    has_tools = any(msg.get("role") in ("tool", "function") or
                    "tool_calls" in msg for msg in messages)

    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue

        # With tools: preserve reasoning blocks (they're needed for multi-step tool chains)
        # Without tools: strip reasoning blocks to save prompt space
        if has_tools:
            # Keep everything when tools are used
            continue

        kept = [
            b for b in content
            if not (isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"))
        ]
        if len(kept) != len(content):
            msg["content"] = kept
            changed = True
    return changed


def is_auxiliary_request(data):
    """Detect if this is a small helper request that should use minimal thinking.
    Returns True if the request matches auxiliary patterns (title, classification,
    routing, summarization, etc.)."""
    messages = data.get("messages", [])
    if not messages:
        return False

    # Get the last user message
    last_user_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                last_user_content = content.lower()
                break
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        last_user_content = block.get("text", "").lower()
                        break

    # Check for auxiliary patterns
    for pattern in AUXILIARY_PATTERNS:
        if pattern in last_user_content:
            return True

    # Check for very short prompts (likely simple queries)
    if len(last_user_content.strip()) < 30:
        # Simple questions that don't need deep reasoning
        simple_questions = ["what is", "what's", "who is", "define", "explain briefly"]
        for q in simple_questions:
            if last_user_content.startswith(q):
                return True

    return False

# Custom HTTPX client with connection pooling
client = httpx.AsyncClient(base_url=TARGET_URL, timeout=600.0)


async def rewrite_model_stream(source, needle: bytes, repl: bytes):
    """Stream response bytes, replacing the backend model id with the public
    one. Carries a tail between chunks so a match split across a chunk
    boundary is still rewritten."""
    keep = len(needle) - 1
    carry = b""
    async for chunk in source:
        buf = (carry + chunk).replace(needle, repl)
        if keep and len(buf) > keep:
            carry, out = buf[-keep:], buf[:-keep]
            if out:
                yield out
        else:
            carry = buf
    if carry:
        yield carry

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy(path: str, request: Request):
    method = request.method
    headers = dict(request.headers)
    
    # Check for authorization
    auth_ok = False
    req_token = None
    
    # Check x-api-key (Anthropic format)
    if "x-api-key" in headers:
        req_token = headers["x-api-key"]
    # Check Authorization (OpenAI format)
    elif "authorization" in headers:
        auth_header = headers["authorization"]
        if auth_header.startswith("Bearer "):
            req_token = auth_header[7:]

    if req_token == AUTH_TOKEN:
        auth_ok = True

    # Allow OPTIONS pre-flight requests without authentication (CORS)
    if method == "OPTIONS":
        auth_ok = True

    if not auth_ok:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Set up headers to forward to local DS4
    forward_headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": headers.get("content-type", "application/json"),
        "Accept": headers.get("accept", "*/*"),
    }
    
    # Copy other standard/Anthropic-specific headers
    for k in ["accept-encoding", "user-agent", "anthropic-version"]:
        if k in headers:
            forward_headers[k] = headers[k]

    body = await request.body()
    
    # Intercept and modify the request body to disable thinking mode for speed
    if method == "POST" and "application/json" in headers.get("content-type", "").lower():
        try:
            data = json.loads(body)
            # (#1) Drop stale reasoning blocks from the conversation history so
            # the prompt doesn't grow unboundedly turn over turn.
            strip_prior_reasoning(data.get("messages"))
            # (#2) Keep thinking ON but bounded, unless the client explicitly
            # disabled it. A finite budget gives the reasoning benefit without a
            # multi-minute pre-answer stall.
            #
            # The server enforces this budget and will close thinking when the
            # limit is reached, then produce the visible answer.
            #
            # For auxiliary requests (title, classification, summarization, etc.),
            # use a minimal budget since these don't need deep reasoning.
            if THINKING_BUDGET > 0:
                th = data.get("thinking")
                explicitly_disabled = isinstance(th, dict) and th.get("type") == "disabled"

                # Determine budget based on request type
                if not explicitly_disabled:
                    if is_auxiliary_request(data):
                        budget = AUXILIARY_BUDGET
                    else:
                        budget = THINKING_BUDGET

                    # budget_tokens must be < max_tokens; leave room for the answer.
                    mt = data.get("max_tokens")
                    if isinstance(mt, int) and mt > 0:
                        budget = min(budget, max(256, mt - 256))

                    # Preserve client's thinking preference but add the budget
                    if isinstance(th, dict) and th.get("type") == "enabled":
                        th["budget_tokens"] = budget
                        data["thinking"] = th
                    else:
                        data["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # Sampling policy:
            # - DSML tool grammar requires constrained structural tokens; the
            #   server handles this via grammar-constrained decoding for tool
            #   blocks while keeping payload strings stochastic.
            # - For non-tool content, preserve the client's temperature/top_p
            #   settings. Only force greedy for tool-only requests.
            has_tools = "tools" in data
            if not has_tools:
                # Non-tool request: respect client sampling settings
                if "temperature" not in data:
                    data["temperature"] = 1.0
                if "top_p" not in data:
                    data["top_p"] = 1.0
            else:
                # Tool request: use greedy for structural DSML safety
                data["temperature"] = 0
                data.pop("top_p", None)
                data.pop("top_k", None)
            # Map the public model ID (or anything a client sends) to the
            # backend's real alias.
            if "model" in data:
                data["model"] = BACKEND_MODEL
            body = json.dumps(data).encode("utf-8")
        except Exception as e:
            # Fallback to raw body on parse errors
            pass

    params = request.query_params

    # Forward the request to DS4 using connection pool
    req = client.build_request(
        method,
        f"/{path}",
        content=body,
        headers=forward_headers,
        params=params,
    )
    
    try:
        resp = await client.send(req, stream=True)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Bad Gateway: {str(e)}")

    # Clean response headers
    response_headers = dict(resp.headers)
    for h in ["content-length", "transfer-encoding", "connection", "keep-alive"]:
        if h in response_headers:
            del response_headers[h]

    # Present a single model identity on the model-list endpoint. The backend
    # advertises multiple aliases (flash/pro) for the one loaded GGUF; collapse
    # them to just the public id so clients see exactly one model.
    if method == "GET" and path.rstrip("/") == "v1/models":
        raw = await resp.aread()
        try:
            data = json.loads(raw)
            entries = data.get("data") or []
            keep = entries[0] if entries else {"object": "model", "owned_by": "ds4.c"}
            keep = {**keep, "id": PUBLIC_MODEL_ID}
            data["data"] = [keep]
            out = json.dumps(data).encode("utf-8")
        except Exception:
            out = raw.replace(BACKEND_MODEL.encode(), PUBLIC_MODEL_ID.encode())
        return StreamingResponse(
            iter([out]),
            status_code=resp.status_code,
            headers=response_headers,
            media_type="application/json",
        )

    # Rewrite the backend model id to the public one in JSON / SSE bodies
    # (covers /v1/models, chat completions, and streamed chunks).
    content_type = resp.headers.get("content-type", "")
    if "json" in content_type or "event-stream" in content_type:
        stream = rewrite_model_stream(
            resp.aiter_raw(),
            BACKEND_MODEL.encode("utf-8"),
            PUBLIC_MODEL_ID.encode("utf-8"),
        )
    else:
        stream = resp.aiter_raw()

    return StreamingResponse(
        stream,
        status_code=resp.status_code,
        headers=response_headers,
        media_type=resp.headers.get("content-type"),
    )
