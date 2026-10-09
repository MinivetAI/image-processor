"""Fetch and downsize picker images before attaching them to an LLM request."""
import asyncio
import base64
import binascii
import io
import ipaddress
import socket
from pathlib import Path
from urllib.parse import unquote_to_bytes, urlsplit

import aiohttp
from PIL import Image, ImageOps


MAX_SOURCE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_WIDTH = 300
MAX_HEIGHT = 400
JPEG_QUALITY = 82


def _check_size(data: bytes) -> bytes:
    if not data or len(data) > MAX_SOURCE_BYTES:
        raise ValueError("Image is empty or exceeds the 20 MB source limit")
    return data


async def _public_host(host: str, port: int):
    try:
        results = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as error:
        raise ValueError(f"Could not resolve image host {host!r}") from error
    addresses = {entry[4][0] for entry in results}
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ValueError("Image URLs must resolve only to public IP addresses")


async def _download(source: str, session: aiohttp.ClientSession) -> bytes:
    parts = urlsplit(source)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Image source must be an HTTP(S) URL, local file, or image data URI")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    await _public_host(parts.hostname, port)
    async with session.get(source, allow_redirects=False) as response:
        peer = response.connection.transport.get_extra_info("peername") if response.connection else None
        if peer and not ipaddress.ip_address(peer[0]).is_global:
            raise ValueError("Image URL connected to a non-public IP address")
        if response.status != 200:
            raise ValueError(f"Image download failed with HTTP {response.status}")
        if response.content_length and response.content_length > MAX_SOURCE_BYTES:
            raise ValueError("Image exceeds the 20 MB source limit")
        data = bytearray()
        async for chunk in response.content.iter_chunked(64 * 1024):
            data.extend(chunk)
            if len(data) > MAX_SOURCE_BYTES:
                raise ValueError("Image exceeds the 20 MB source limit")
        return _check_size(bytes(data))


async def _read(source: str, session: aiohttp.ClientSession) -> bytes:
    if source.startswith("data:"):
        header, separator, payload = source.partition(",")
        if not separator or not header.startswith("data:image/"):
            raise ValueError("Only image data URIs are supported")
        try:
            data = base64.b64decode(payload, validate=True) if ";base64" in header else unquote_to_bytes(payload)
        except (binascii.Error, ValueError) as error:
            raise ValueError("Invalid image data URI") from error
        return _check_size(data)
    if source.startswith(("http://", "https://")):
        return await _download(source, session)
    path = Path(source)
    if not path.is_file():
        raise ValueError(f"Image file does not exist: {source}")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("Image exceeds the 20 MB source limit")
    return _check_size(path.read_bytes())


def _compress(data: bytes) -> str:
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("Image dimensions exceed the 40 megapixel limit")
            image = ImageOps.exif_transpose(source)
            image.seek(0)
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, "white")
                background.paste(rgba, mask=rgba.getchannel("A"))
                image = background
            else:
                image = image.convert("RGB")
            image.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.Resampling.LANCZOS)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=JPEG_QUALITY, optimize=True, progressive=True)
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Image could not be decoded") from error
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


async def prepare_images(sources: list[str]) -> list[str]:
    """Download, validate, resize and JPEG-compress images in their input order."""
    timeout = aiohttp.ClientTimeout(total=45, connect=10)
    connector = aiohttp.TCPConnector(limit=8)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        semaphore = asyncio.Semaphore(8)

        async def prepare(source):
            async with semaphore:
                return _compress(await _read(source, session))

        return await asyncio.gather(*(prepare(source) for source in sources))
