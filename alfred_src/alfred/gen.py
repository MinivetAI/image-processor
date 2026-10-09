"""Generative image/video output against OpenAI-spec endpoints (vLLM-Omni).

Design and wire contracts: docs/gen_contract.md. One running deployment
(one port = one model+LoRA) = one GenServer; constant knobs are tuned once
on a GenTask; per call you vary only prompt / media / seed.
"""

import json
import time
import base64
import random
import asyncio
import mimetypes
import urllib.request

import aiohttp
from pathlib import Path
from aiohttp import TCPConnector

from .alfie import _RETRY_STATUS


# --- Artifact ----------------------------------------------------------------

class Artifact:
    """One generated output (image or video); exactly one of bytes_/url is set.
    Usable directly as a `media=` item on chat Tasks (via data_uri)."""

    def __init__(self, kind, mime, bytes_=None, url=None, meta=None):
        self.kind = kind          # "image" | "video"
        self.mime = mime
        self.bytes_ = bytes_
        self.url = url
        self.meta = meta or {}    # raw response item / final job object

    def data_uri(self):
        """Base64 data-URI from bytes, or the remote URL passed through."""
        if self.bytes_ is not None:
            return f"data:{self.mime};base64,{base64.b64encode(self.bytes_).decode()}"
        return self.url

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.bytes_
        if data is None:
            with urllib.request.urlopen(self.url) as resp:
                data = resp.read()
        path.write_bytes(data)
        return path


# --- request building --------------------------------------------------------

def _media_field(item):
    """Classify one media input. Returns (url, file_part) — exactly one set.
    URLs pass through as string fields; local paths and Artifacts upload as
    multipart file parts (filename, bytes, mime)."""
    if hasattr(item, "data_uri"):  # Artifact (duck-typed)
        if item.bytes_ is None:
            return item.url, None
        ext = mimetypes.guess_extension(item.mime) or ""
        return None, (f"artifact{ext}", item.bytes_, item.mime)
    item = str(item)
    if item.startswith(("http://", "https://")):
        return item, None
    path = Path(item)
    if not path.is_file():
        raise FileNotFoundError(f"Media file not found: {item}")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return None, (path.name, path.read_bytes(), mime)


def _form_fields(payload, model=None):
    """Flatten a payload dict into multipart string fields (None skipped)."""
    body = dict(payload)
    if model and "model" not in body:
        body["model"] = model
    fields = []
    for key, value in body.items():
        if value is None:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        fields.append((key, str(value)))
    return fields


def _image_artifacts(data):
    fmt = data.get("output_format") or "png"
    mime = f"image/{'jpeg' if fmt in ('jpg', 'jpeg') else fmt}"
    out = []
    for item in data.get("data", []):
        raw = base64.b64decode(item["b64_json"]) if item.get("b64_json") else None
        meta = {k: v for k, v in item.items() if k != "b64_json"}
        meta["size"] = data.get("size")
        out.append(Artifact("image", mime, bytes_=raw, url=item.get("url"), meta=meta))
    return out


# --- Gen server --------------------------------------------------------------

class GenServer:
    """One running gen deployment (which LoRA is loaded is a property of the
    server, not a request knob). Own pool sized to that GPU; a slot is held
    for a video job's whole submit->poll->download lifecycle. model=None omits
    the field — vLLM-Omni defaults to its loaded checkpoint."""

    def __init__(self, base_url, model=None, api_key=None, max_concurrent=2,
                 timeout=600, retries=3, backoff=0.5, poll_interval=5.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.timeout = timeout          # per request, and overall video-job deadline
        self.retries = retries
        self.backoff = backoff
        self.poll_interval = poll_interval
        self._session = None
        self._sem = asyncio.Semaphore(max_concurrent)

    def _get_session(self):
        if self._session is None or self._session.closed:
            connector = TCPConnector(limit=self.max_concurrent)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._session

    async def _http(self, method, path, json_body=None, form_fields=None, raw=False):
        """One endpoint call with transient-error retries (same envelope as
        LLMServer.respond). FormData cannot be re-sent, so multipart bodies are
        rebuilt from `form_fields` tuples on every attempt. Does NOT touch the
        semaphore — the public methods hold it around whole operations."""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = f"{self.base_url}{path}"
        for attempt in range(self.retries + 1):
            try:
                kwargs = {}
                if json_body is not None:
                    kwargs["json"] = json_body
                if form_fields is not None:
                    form = aiohttp.FormData()
                    for name, value in form_fields:
                        if isinstance(value, tuple):
                            filename, data, mime = value
                            form.add_field(name, data, filename=filename, content_type=mime)
                        else:
                            form.add_field(name, value)
                    kwargs["data"] = form
                session = self._get_session()
                async with session.request(method, url, headers=headers, **kwargs) as resp:
                    if resp.status < 400:
                        return await resp.read() if raw else json.loads(await resp.text())
                    body = await resp.text()
                    retry = resp.status in _RETRY_STATUS
                    err = RuntimeError(f"Gen request failed [{resp.status}] {path}: {body}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                retry, err = True, e
            if attempt < self.retries and retry:
                await asyncio.sleep(self.backoff * 2 ** attempt + random.random() * 0.1)
                continue
            raise err

    async def images(self, payload):
        """POST /images/generations (JSON). Returns list[Artifact]."""
        body = {k: v for k, v in payload.items() if v is not None}
        if self.model and "model" not in body:
            body["model"] = self.model
        async with self._sem:
            data = await self._http("POST", "/images/generations", json_body=body)
        return _image_artifacts(data)

    async def edits(self, payload, images):
        """POST /images/edits (multipart). `images` are the inputs in order:
        URLs go through as `url` fields, local paths / Artifacts upload as
        `image` file parts. Returns list[Artifact]."""
        fields = _form_fields(payload, model=self.model)
        for item in images:
            url, file_part = _media_field(item)
            fields.append(("url", url) if url else ("image", file_part))
        async with self._sem:
            data = await self._http("POST", "/images/edits", form_fields=fields)
        return _image_artifacts(data)

    async def videos(self, payload, first=None, last=None):
        """POST /videos, poll the job, download the MP4; returns one Artifact.
        `first` is the I2V reference, `last` additionally makes it FLF2V.
        A job that reports status "failed" is resubmitted within the retry
        budget; exceeding `timeout` while polling raises without resubmit.

        The semaphore (GPU slot) is released as soon as the poll confirms the
        GPU is done, before the download.  This lets the next job start on the
        GPU while the current job's output is being downloaded and uploaded to S3."""
        fields = _form_fields(payload, model=self.model)
        refs = ((first, "image_reference", "input_reference"),
                (last, "last_image_reference", "last_input_reference"))
        for item, url_name, file_name in refs:
            if item is None:
                continue
            url, file_part = _media_field(item)
            fields.append((url_name, url) if url else (file_name, file_part))

        # Phase 1: submit + poll — holds GPU slot via semaphore.
        job = err = None
        await self._sem.acquire()
        try:
            for attempt in range(self.retries + 1):
                job = await self._http("POST", "/videos", form_fields=fields)
                deadline = time.monotonic() + self.timeout
                while job.get("status") in ("queued", "in_progress"):
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"Video job {job.get('id')} still {job['status']} "
                            f"after {self.timeout}s")
                    await asyncio.sleep(self.poll_interval)
                    job = await self._http("GET", f"/videos/{job['id']}")
                if job.get("status") == "completed":
                    break  # GPU done — semaphore released in finally below
                err = RuntimeError(f"Video job failed: {job.get('error')}")
                if attempt < self.retries:
                    await asyncio.sleep(self.backoff * 2 ** attempt + random.random() * 0.1)
        finally:
            self._sem.release()  # GPU slot free — next job can start immediately

        if err is not None:
            raise err

        # Phase 2: return the download URL — caller (post-proc) downloads independently
        # so the GPU slot and this async slot are both free for the next job.
        download_url = f"{self.base_url}/videos/{job['id']}/content"
        return Artifact("video", job.get("media_type") or "video/mp4",
                        url=download_url, meta=job)

    async def vace(self, payload, reference_images=None, source_video=None, mask=None):
        """POST /videos (VACE variant), poll, download the MP4; returns one Artifact.
        reference_images: list of Artifact/URL — subject reference frames.
        source_video: list of Artifact/URL — control video frames (poses).
        mask: list of Artifact/URL — optional mask frames.
        Semaphore released at GPU-done (same pipeline pattern as videos())."""
        fields = _form_fields(payload, model=self.model)
        for items, field_name in (
            (reference_images, "vace_reference_image"),
            (source_video,     "vace_source_frame"),
            (mask,             "vace_mask_frame"),
        ):
            if not items:
                continue
            for item in items:
                _, file_part = _media_field(item)
                if file_part:
                    fields.append((field_name, file_part))

        job = err = None
        await self._sem.acquire()
        try:
            for attempt in range(self.retries + 1):
                job = await self._http("POST", "/videos", form_fields=fields)
                deadline = time.monotonic() + self.timeout
                while job.get("status") in ("queued", "in_progress"):
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"VACE job {job.get('id')} still {job['status']} "
                            f"after {self.timeout}s")
                    await asyncio.sleep(self.poll_interval)
                    job = await self._http("GET", f"/videos/{job['id']}")
                if job.get("status") == "completed":
                    break
                err = RuntimeError(f"VACE job failed: {job.get('error')}")
                if attempt < self.retries:
                    await asyncio.sleep(self.backoff * 2 ** attempt + random.random() * 0.1)
        finally:
            self._sem.release()

        if err is not None:
            raise err

        download_url = f"{self.base_url}/videos/{job['id']}/content"
        return Artifact("video", job.get("media_type") or "video/mp4",
                        url=download_url, meta=job)

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()


# --- Gen task ----------------------------------------------------------------

class GenTask:
    """Constant knobs tuned once (mirrors Task.instruction + extra); per call
    you vary only prompt / media / seed. kind picks the endpoint:
    "image" -> /images/generations, "edit" -> /images/edits, "video" -> /videos.
    Assume n=1 on this fleet — n>1 is unsupported or super-linear in time."""

    KINDS = ("image", "edit", "video")

    def __init__(self, server, kind, defaults=None):
        if kind not in self.KINDS:
            raise ValueError(f"kind must be one of {self.KINDS}, got {kind!r}")
        self.server = server
        self.kind = kind
        self.defaults = defaults or {}

    async def do(self, prompt, media=None, **overrides):
        """Returns list[Artifact] for image kinds, a single Artifact for video.
        media items may be URLs, local paths, or Artifacts:
          edit  -> the input image(s), in order
          video -> media[0] = first frame (I2V), media[1] = last frame (FLF2V)
        An override (or default) of None deletes the key from the request."""
        payload = {**self.defaults, **overrides, "prompt": prompt}
        media = media or []
        if self.kind == "image":
            return await self.server.images(payload)
        if self.kind == "edit":
            if not media:
                raise ValueError("edit requires at least one input image in media")
            return await self.server.edits(payload, media)
        first = media[0] if media else None
        last = media[1] if len(media) > 1 else None
        return await self.server.videos(payload, first=first, last=last)

    async def do_many(self, rows):
        """rows: prompt strings or dicts {"prompt": ..., "media": [...], **overrides}.
        Runs concurrently under the server's pool; a failed row is returned as
        its Exception, same contract as Task.do_many. Unlike Task.do_many,
        media is per-row — gen rows almost always differ in media."""
        async def one(row):
            if isinstance(row, dict):
                row = dict(row)
                return await self.do(row.pop("prompt"), media=row.pop("media", None), **row)
            return await self.do(row)
        return await asyncio.gather(*(one(r) for r in rows), return_exceptions=True)
