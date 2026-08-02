from __future__ import annotations

from dataclasses import dataclass
from email.message import Message
import ipaddress
from pathlib import Path
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


class DocumentUrlImportError(ValueError):
    pass


@dataclass(frozen=True)
class FetchedUrlDocument:
    content: bytes
    filename: str
    content_type: str
    final_url: str


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def fetch_url_document(
    url: str,
    *,
    max_bytes: int,
    filename_override: str | None = None,
    max_redirects: int = 3,
) -> FetchedUrlDocument:
    current = url.strip()
    opener = build_opener(_NoRedirect())
    for _ in range(max_redirects + 1):
        _validate_public_http_url(current)
        request = Request(
            current,
            headers={
                "User-Agent": "ResumeMind/1.0 provenance-import",
                "Accept": "application/pdf,application/msword,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,text/html,text/csv,application/x-ndjson,text/plain;q=0.8,*/*;q=0.2",
            },
        )
        try:
            response = opener.open(request, timeout=20)
        except HTTPError as exc:
            if 300 <= exc.code < 400 and exc.headers.get("Location"):
                current = urljoin(current, exc.headers["Location"])
                continue
            raise DocumentUrlImportError(f"URL 下载失败：HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise DocumentUrlImportError(f"URL 下载失败：{exc}") from exc
        try:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise DocumentUrlImportError("URL 文件超过上传大小限制")
            content = response.read(max_bytes + 1)
            if len(content) > max_bytes:
                raise DocumentUrlImportError("URL 文件超过上传大小限制")
            if not content:
                raise DocumentUrlImportError("URL 返回空内容")
            content_type = response.headers.get_content_type() or "application/octet-stream"
            filename = filename_override or _response_filename(response.headers, current, content_type)
            content, filename, content_type = _normalize_text_content(
                content,
                filename,
                content_type,
                response.headers.get_content_charset(),
            )
            return FetchedUrlDocument(content, filename, content_type, response.geturl() or current)
        finally:
            response.close()
    raise DocumentUrlImportError("URL 重定向次数超过限制")


def _validate_public_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise DocumentUrlImportError("仅支持 HTTP(S) URL")
    if parsed.username or parsed.password:
        raise DocumentUrlImportError("URL 不得包含用户名或密码")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise DocumentUrlImportError("URL 域名无法解析") from exc
    for address in {item[4][0] for item in addresses}:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise DocumentUrlImportError("URL 指向内网、本机或非公网地址，已拒绝访问")


def _response_filename(headers: Message, url: str, content_type: str) -> str:
    disposition = headers.get("Content-Disposition") or ""
    filename = None
    if "filename*=" in disposition:
        filename = disposition.split("filename*=", 1)[1].split(";", 1)[0].strip().strip('"')
        if "''" in filename:
            filename = filename.split("''", 1)[1]
        filename = unquote(filename)
    elif "filename=" in disposition:
        filename = disposition.split("filename=", 1)[1].split(";", 1)[0].strip().strip('"')
    filename = Path(filename or urlparse(url).path).name
    if filename and Path(filename).suffix:
        return filename
    suffixes = {
        "text/html": ".html",
        "text/csv": ".csv",
        "application/x-ndjson": ".jsonl",
        "application/pdf": ".pdf",
        "application/msword": ".doc",
        "application/vnd.ms-excel": ".xls",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "text/plain": ".txt",
    }
    return f"url-import{suffixes.get(content_type, '')}"


def _normalize_text_content(
    content: bytes,
    filename: str,
    content_type: str,
    charset: str | None,
) -> tuple[bytes, str, str]:
    if content_type not in {"text/html", "text/csv", "application/x-ndjson", "application/json", "text/plain"}:
        return content, filename, content_type
    encoding = charset or "utf-8"
    try:
        text = content.decode(encoding)
    except (LookupError, UnicodeDecodeError):
        try:
            text = content.decode("gb18030")
        except UnicodeDecodeError as exc:
            raise DocumentUrlImportError("URL 文本编码无法识别") from exc
    suffix = Path(filename).suffix.lower()
    if content_type == "text/html" and suffix not in {".html", ".htm"}:
        filename = f"{Path(filename).stem or 'url-import'}.html"
    elif content_type == "text/csv" and suffix != ".csv":
        filename = f"{Path(filename).stem or 'url-import'}.csv"
    elif content_type in {"application/x-ndjson", "application/json"} and suffix != ".jsonl":
        filename = f"{Path(filename).stem or 'url-import'}.jsonl"
        content_type = "application/x-ndjson"
    elif content_type == "text/plain" and not suffix:
        filename = f"{Path(filename).stem or 'url-import'}.txt"
    return text.encode("utf-8"), filename, content_type
