import json
import base64
import random
import asyncio
import mimetypes

import aiohttp
from pathlib import Path
from aiohttp import TCPConnector
from pydantic import ValidationError

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


# --- Multimodal content ----------------------------------------------------

_IMAGE, _VIDEO, _AUDIO, _DOC = "image", "video", "audio", "document"


def _media_kind(mime):
    if not mime:
        return None
    if mime.startswith("image/"):
        return _IMAGE
    if mime.startswith("video/"):
        return _VIDEO
    if mime.startswith("audio/"):
        return _AUDIO
    if mime == "application/pdf":
        return _DOC
    return None


def _media_block(item):
    """Build one OpenAI content block from a url, local file path, data: URI, or a
    generated Artifact (anything with .data_uri(), see alfred/gen.py)."""
    if hasattr(item, "data_uri"):
        key = "video_url" if getattr(item, "kind", _IMAGE) == _VIDEO else "image_url"
        return {"type": key, key: {"url": item.data_uri()}}
    if item.startswith("data:"):
        # Inline data URI — pass through directly; model server needs no download
        mime = item.split(";", 1)[0].split(":", 1)[1]
        key = "video_url" if mime.startswith("video") else "image_url"
        return {"type": key, key: {"url": item}}
    is_url = item.startswith(("http://", "https://"))
    probe = item.split("?", 1)[0].split("#", 1)[0]  # ignore query/fragment for detection
    mime = mimetypes.guess_type(probe)[0]
    kind = _media_kind(mime)

    if is_url:
        if kind is None:
            kind = _IMAGE  # extensionless/CDN URLs are assumed to be images
        if kind in (_AUDIO, _DOC):
            raise ValueError(f"{kind} media must be a local file, got URL: {item!r}")
        key = "image_url" if kind == _IMAGE else "video_url"
        return {"type": key, key: {"url": item}}

    if kind is None:
        raise ValueError(f"Unsupported or undetectable media type: {item!r}")
    path = Path(item)
    if not path.is_file():
        raise FileNotFoundError(f"Media file not found: {item}")
    b64 = base64.b64encode(path.read_bytes()).decode()

    if kind == _IMAGE:
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    if kind == _VIDEO:
        return {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{b64}"}}
    if kind == _AUDIO:
        fmt = (mimetypes.guess_extension(mime) or ".wav").lstrip(".")
        return {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}
    return {
        "type": "file",
        "file": {"filename": path.name, "file_data": f"data:{mime};base64,{b64}"},
    }


def _build_content(text, media=None, media_first=False):
    """Plain text, or an OpenAI content-block list when media is present."""
    if not media:
        return text
    text_block = {"type": "text", "text": text}
    media_blocks = [_media_block(m) for m in media]
    return media_blocks + [text_block] if media_first else [text_block] + media_blocks


def _messages_contain(messages, content_type):
    """Return whether any message contains an OpenAI multimodal block type."""
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == content_type
            for block in content
        ):
            return True
    return False


# --- Tools -----------------------------------------------------------------

class Tool:
    """Wraps a pydantic model into an OpenAI tool schema. The model's docstring
    is the tool description; its fields are the call parameters."""

    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.description = (model.__doc__ or "").strip()

    def describe(self):
        schema = self.model.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": schema.get("properties", {}),
                    "required": schema.get("required", []),
                },
            },
        }


class Tools:
    def __init__(self):
        self.tools = []

    def register(self, tool):
        self.tools.append(tool)

    def describe(self):
        return [t.describe() for t in self.tools]


# --- LLM server ------------------------------------------------------------

class LLMServer:
    """Generic OpenAI-compatible chat client (vLLM/Qwen, OpenAI, Perplexity, ...).
    Switch backends by changing base_url/model/api_key. `max_concurrent` is a
    single shared pool: every Task and Chat using this server draws from it."""

    def __init__(self, base_url, model, api_key=None, temperature=0.0,
                 max_tokens=1024, max_concurrent=64, timeout=300,
                 retries=3, backoff=0.5, default_extra=None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.retries = retries          # retries on transient transport errors
        self.backoff = backoff          # base seconds for exponential backoff
        self.default_extra = dict(default_extra or {})
        self._session = None
        self._sem = asyncio.Semaphore(max_concurrent)

    def build_content(self, text, media=None):
        """Format one user message. Backends may override multimodal ordering."""
        return _build_content(text, media)

    def prepare_extra(self, messages, extra=None):
        """Merge backend defaults with per-task request options."""
        merged = dict(self.default_extra)
        merged.update(extra or {})
        return merged

    def _get_session(self):
        if self._session is None or self._session.closed:
            connector = TCPConnector(limit=self.max_concurrent)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._session

    async def respond(self, messages, tools=None, guide=None, max_tokens=None, extra=None,
                      return_envelope=False, allow_redirects=True):
        """Send a chat request. `guide` (a pydantic model) enables structured
        JSON output via response_format; `tools` enables tool calling.

        `extra` is merged into the request body verbatim — provider-specific
        knobs that alfred does not interpret, e.g. grounded search options
        (`web_search_options` for OpenAI `*-search-preview` models,
        `search_recency_filter` / `search_domain_filter` for Perplexity sonar).
        A key whose value is None is *deleted* from the payload — use this to
        drop `temperature`, which OpenAI's search-preview models reject.

        Returns the assistant message dict. Grounding citations are surfaced on
        it: OpenAI puts them in `message["annotations"]` (already present);
        Perplexity returns them top-level, so `citations` / `search_results`
        are copied onto the message here. Set return_envelope=True to receive
        the full API response including usage, model and finish_reason instead.
        Set allow_redirects=False to reject HTTP redirects without resending."""
        # Opt-in envelope access preserves finish_reason/model/usage. Existing
        # callers still receive a message; redirect behavior is opt-in per call.
        token_limit = max_tokens or self.max_tokens
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        # OpenAI's o-series / gpt-5 reject `max_tokens` and require
        # `max_completion_tokens`; vLLM and Perplexity want plain `max_tokens`.
        # Gate on the endpoint, not the model name.
        if "api.openai.com" in self.base_url:
            payload["max_completion_tokens"] = token_limit
        else:
            payload["max_tokens"] = token_limit
        if guide is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": guide.__name__, "schema": guide.model_json_schema()},
            }
        if tools:
            payload["tools"] = tools
        request_extra = self.prepare_extra(messages, extra)
        if request_extra:
            for key, value in request_extra.items():
                if value is None:
                    payload.pop(key, None)
                else:
                    payload[key] = value

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.retries + 1):
            try:
                async with self._sem:
                    session = self._get_session()
                    async with session.post(url, json=payload, headers=headers,
                                            allow_redirects=allow_redirects) as resp:
                        body = await resp.text()
                        if 200 <= resp.status < 300:
                            data = json.loads(body)
                            if return_envelope:
                                return data
                            message = data["choices"][0]["message"]
                            for key in ("citations", "search_results"):
                                if key in data:
                                    message[key] = data[key]
                            return message
                        retry = resp.status in _RETRY_STATUS
                        err = RuntimeError(f"LLM request failed [{resp.status}]: {body}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                retry, err = True, e
            if attempt < self.retries and retry:
                await asyncio.sleep(self.backoff * 2 ** attempt + random.random() * 0.1)
                continue
            raise err

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()


class CosmosServer(LLMServer):
    """OpenAI-compatible Cosmos backend with its required video conventions.

    Task authors use the normal Alfred Task API. This backend keeps Cosmos media
    ordering and frame sampling out of task definitions.
    """

    def __init__(self, base_url, model="cosmos3-reasoner", video_fps=4.0, **kwargs):
        super().__init__(base_url, model=model, **kwargs)
        self.video_fps = video_fps

    def build_content(self, text, media=None):
        # Cosmos' tested chat template places media before the question text.
        return _build_content(text, media, media_first=True)

    def prepare_extra(self, messages, extra=None):
        merged = super().prepare_extra(messages, extra)
        if self.video_fps is not None and _messages_contain(messages, "video_url"):
            media_kwargs = dict(merged.get("media_io_kwargs") or {})
            video_kwargs = dict(media_kwargs.get("video") or {})
            video_kwargs.setdefault("fps", self.video_fps)
            media_kwargs["video"] = video_kwargs
            merged["media_io_kwargs"] = media_kwargs
        return merged


# --- Chat / Alfred (multi-turn, tool calling) ------------------------------

class Chat:
    """Mutable message history bound to a server."""

    def __init__(self, server, system=None):
        self.server = server
        self.messages = []
        if system:
            self.add_message("system", system)

    def add_message(self, role, content=None, tool_calls=None, tool_call_id=None, media=None):
        message = {"role": role, "content": self.server.build_content(content or "", media)}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        self.messages.append(message)

    async def respond(self, tools=None, extra=None):
        return await self.server.respond(self.messages, tools=tools, extra=extra)


class Alfred:
    """Multi-turn agent. Returns either content for the user or pending tool
    calls to execute; feed each result back via complete_tool_call."""

    def __init__(self, description, chat, tools):
        self.chat = chat
        self.tools = tools
        self.chat.add_message("system", description)
        self.pending = {}

    async def respond(self, query=None, media=None):
        if query is not None:
            self.chat.add_message("user", query, media=media)
        message = await self.chat.respond(tools=self.tools.describe())
        tool_calls = message.get("tool_calls")
        if tool_calls:
            self.chat.add_message("assistant", message.get("content") or "", tool_calls=tool_calls)
            self.pending = {tc["id"]: tc for tc in tool_calls}
            return {"content": None, "tool_calls": self.pending}
        self.chat.add_message("assistant", message.get("content") or "")
        return {"content": message.get("content"), "tool_calls": {}}

    async def complete_tool_call(self, id, result):
        """Record a tool result (JSON-encoded if not a string). Returns True
        once all pending tool calls for this turn are complete."""
        if not isinstance(result, str):
            result = json.dumps(result)
        self.chat.add_message("tool", result, tool_call_id=id)
        self.pending.pop(id, None)
        return len(self.pending) == 0


# --- Task (single-turn, structured output) ---------------------------------

class Task:
    """Single-turn structured extraction: instruction + guide (pydantic output
    model) -> validated model instance. Shares the server's concurrency pool.
    `repair` re-prompts the model with the validation error when its output is
    malformed JSON or fails the schema."""

    def __init__(self, instruction, guide, server, repair=1, max_tokens=None, extra=None):
        self.server = server
        self.instruction = instruction
        self.guide = guide
        self.repair = repair
        self.max_tokens = max_tokens   # per-task cap, overrides the server default
        self.extra = extra             # constant body knobs (e.g. grounded search options)

    async def do(self, data, media=None):
        if isinstance(data, dict):
            content = "".join(f"**{k}**\n{v}\n\n" for k, v in data.items())
        else:
            content = str(data)
        messages = [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": self.server.build_content(content, media)},
        ]
        for attempt in range(self.repair + 1):
            raw = (await self.server.respond(
                messages, guide=self.guide, max_tokens=self.max_tokens,
                extra=self.extra))["content"]
            try:
                # Some OpenAI-compatible gateways can return a successful
                # envelope with an empty/null assistant content.  Treat that
                # exactly like malformed structured output so the normal
                # schema-repair loop gets a chance to recover.
                if not isinstance(raw, str) or not raw.strip():
                    raise ValueError("assistant response has no JSON content")
                return self.guide.model_validate(json.loads(raw))
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                if attempt >= self.repair:
                    raise
                messages = messages + [
                    {"role": "assistant", "content": raw if isinstance(raw, str) else ""},
                    {"role": "user", "content":
                        f"That response was invalid: {e}. Reply with only JSON "
                        f"matching the required schema."},
                ]

    async def do_many(self, rows, media=None):
        """Run many inputs concurrently, capped by the server's shared pool.
        Returns a list aligned with `rows`; each item is the validated model or,
        if that row exhausted its retries, the Exception that was raised (so one
        bad row never sinks the batch)."""
        return await asyncio.gather(
            *(self.do(r, media=media) for r in rows), return_exceptions=True
        )
