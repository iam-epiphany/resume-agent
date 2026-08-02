"""通用知识库上传脚本：把任意目录下可解析的文档上传到 ResumeMind 知识库。

用法（服务器部署后）:
    python scripts/upload_knowledge_base.py --base-url http://127.0.0.1:8000
    python scripts/upload_knowledge_base.py --base-url http://127.0.0.1:8000 --root docs

规则:
    - 收录扩展名: pdf/doc/docx/md/txt/html/htm
    - 自动排除: .git/.venv/node_modules/__pycache__/.idea/dist/outputs/data/.run-state
    - jpg/png 等图片无 OCR 解析能力，扫描时列出并跳过（建议先转成 PDF 再入库）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from uuid import uuid4
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 与 backend SUPPORTED_DOCUMENT_EXTENSIONS 保持一致：xls/xlsx/csv（表格场景）已不再收录
SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".md", ".txt", ".html", ".htm"}
EXCLUDED_DIRS = {
    ".git", ".venv", "node_modules", "__pycache__", ".idea", ".pytest_cache",
    "dist", "outputs", "data", ".run-state", ".github", "frontend", "backend",
    "scripts", "docs-guide", "项目",  # docs/项目 为源码归档，不进知识库
}


def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72, flush=True)


# 管理员 token（--admin-password 登录后填充），自动附加到后续所有请求。
_AUTH_HEADERS: dict[str, str] = {}


def request_json(
    method: str,
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    merged_headers = {**_AUTH_HEADERS, **(headers or {})}
    request = urllib.request.Request(url, data=data, headers=merged_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail}") from exc
    if not raw:
        return {}
    return json.loads(raw)


def admin_login(base_url: str, password: str, timeout: float) -> None:
    """登录管理员后台，获取 JWT 并附加到后续所有请求（/api/documents、/warmup 需认证）。"""
    global _AUTH_HEADERS
    print_header("管理员登录")
    payload = request_json(
        "POST",
        f"{base_url}/api/auth/login",
        data=json.dumps({"password": password}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    token = payload.get("token")
    if not token:
        raise RuntimeError("管理员登录失败：未返回 token（检查 ADMIN_PASSWORD 与 .env 配置）。")
    _AUTH_HEADERS = {"Authorization": f"Bearer {token}"}
    print("管理员登录成功")


def assert_service_available(base_url: str, timeout: float) -> None:
    print_header("检查服务状态")
    try:
        health = request_json("GET", f"{base_url}/api/health", timeout=timeout)
    except Exception as exc:
        raise RuntimeError(f"无法访问 ResumeMind 服务，请先启动系统：{exc}") from exc
    print(f"后端状态: {health.get('status')} build_id={health.get('build_id')}")


def warmup_models(base_url: str, timeout: float) -> None:
    print_header("模型预热")
    print("正在调用 /api/health/warmup。首次运行会下载并加载本地模型，耗时较长...")
    started = time.perf_counter()
    try:
        payload = request_json(
            "POST", f"{base_url}/api/health/warmup", data=b"", timeout=max(timeout, 600.0)
        )
    except Exception as exc:
        raise RuntimeError("模型预热失败。请检查模型文件与网络后重试。") from exc
    elapsed = time.perf_counter() - started
    print(f"模型预热完成: warmed={payload.get('warmed')} elapsed={elapsed:.1f}s")


def scan_documents(root: Path) -> tuple[list[Path], list[Path]]:
    """递归扫描 root，返回 (可上传文件列表, 图片文件列表)。"""
    files: list[Path] = []
    images: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        parts = relative.parts
        if any(part in EXCLUDED_DIRS or part.startswith("nginx") for part in parts):
            continue
        if (
            path.name.startswith("requirements")
            or path.name == "SKILL.md"
            or path.name == "PROGRESS.md"
            # 根 README 是系统说明（描述 ResumeMind 本身），会干扰“介绍你自己”等
            # 代词问题的检索（把“你”指代成系统），不能进入知识库
            or (path.name == "README.md" and path.parent == root)
        ):
            continue
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_EXTENSIONS:
            files.append(path)
        elif suffix in {".jpg", ".jpeg", ".png", ".gif", ".bmp"}:
            images.append(path)
    return files, images


def existing_documents_by_filename(base_url: str, timeout: float) -> dict[str, dict[str, Any]]:
    payload = request_json("GET", f"{base_url}/api/documents", timeout=timeout)
    documents = payload.get("documents") or []
    result: dict[str, dict[str, Any]] = {}
    for document in documents:
        filename = str(document.get("filename") or "")
        if filename and filename not in result:
            result[filename] = document
    return result


MANIFEST_PATH = PROJECT_ROOT / ".run-state" / "kb_manifest.json"


def load_manifest() -> dict[str, str]:
    """加载本地清单（filename -> sha256）。列表接口不返回文件哈希，靠它判断内容是否变化。"""
    if MANIFEST_PATH.is_file():
        try:
            payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in payload.items()}
        except (OSError, ValueError):
            return {}
    return {}


def save_manifest(manifest: dict[str, str]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


def upload_file(base_url: str, path: Path, timeout: float, *, overwrite_document_id: str | None = None, filename_override: str | None = None) -> dict[str, Any]:
    boundary = f"----ResumeMind{uuid4().hex}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    file_bytes = path.read_bytes()
    upload_name = filename_override or path.name
    parts = [
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{upload_name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        + file_bytes,
    ]
    if overwrite_document_id:
        parts.append(
            (
                f"\r\n--{boundary}\r\n"
                f'Content-Disposition: form-data; name="overwrite_document_id"\r\n\r\n'
                f"{overwrite_document_id}"
            ).encode("utf-8")
        )
    parts.append(f"\r\n--{boundary}--\r\n".encode("ascii"))
    body = b"".join(parts)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Idempotency-Key": f"knowledge-base-{hashlib.sha256(file_bytes).hexdigest()[:24]}",
    }
    return request_json(
        "POST",
        f"{base_url}/api/documents/upload",
        data=body,
        headers=headers,
        timeout=max(timeout, 300.0),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upload_files(base_url: str, files: list[Path], timeout: float, *, root: Path) -> dict[str, str]:
    """上传知识库文档。

    - 内容未变（本地清单 sha256 相同）→ 跳过
    - 内容已变 → 用 overwrite_document_id 覆盖更新并重新索引
    - 存在但索引失败/未完成 → 重新排队索引
    """
    print_header(f"上传知识库文档（共 {len(files)} 个文件）")
    existing = existing_documents_by_filename(base_url, timeout)
    manifest = load_manifest()
    tracked: dict[str, str] = {}
    uploaded = 0
    skipped = 0
    requeued = 0
    updated = 0

    total = len(files)
    for index, path in enumerate(files, start=1):
        # 知识库按文件名唯一：非根目录的 README.md 撞名时加父目录前缀（如 README_ResumeMind.md）
        filename_override = None
        if path.name == "README.md":
            relative_parts = path.relative_to(root).parts
            if len(relative_parts) > 1:
                parent_name = relative_parts[-2]
                filename_override = f"README_{parent_name}.md"
        upload_name = filename_override or path.name
        local_sha = file_sha256(path)
        existing_doc = existing.get(upload_name)
        if existing_doc:
            document_id = str(existing_doc["document_id"])
            tracked[upload_name] = document_id
            status = str(existing_doc.get("status") or "")
            unchanged = manifest.get(upload_name) == local_sha
            if status == "indexed" and unchanged:
                skipped += 1
                print(f"[{index:03d}/{total}] 内容未变化，跳过: {upload_name}")
                continue
            if status in {"index_failed", "uploaded", "index_queued", "indexing"} and unchanged:
                request_json("POST", f"{base_url}/api/documents/{document_id}/index", data=b"", timeout=timeout)
                requeued += 1
                print(f"[{index:03d}/{total}] 已存在，重新排队索引: {upload_name} status={status}")
                continue
            # 内容已变化 → 覆盖更新（目标正被索引时等待重试）
            response = None
            for attempt in range(4):
                try:
                    response = upload_file(base_url, path, timeout, overwrite_document_id=document_id, filename_override=filename_override)
                    break
                except RuntimeError as exc:
                    if "overwrite_target_busy" in str(exc) and attempt < 3:
                        print(f"    目标正在处理中，{15 * (attempt + 1)} 秒后重试...")
                        time.sleep(15 * (attempt + 1))
                        continue
                    raise
            document_id = str(response["document_id"])
            tracked[upload_name] = document_id
            updated += 1
            print(
                f"[{index:03d}/{total}] 内容已更新，覆盖重新上传: {upload_name} -> {document_id} "
                f"task={response.get('status')}/{response.get('stage')}",
                flush=True,
            )
            manifest[upload_name] = local_sha
            continue

        response = upload_file(base_url, path, timeout, filename_override=filename_override)
        document_id = str(response["document_id"])
        tracked[upload_name] = document_id
        uploaded += 1
        manifest[upload_name] = local_sha
        print(
            f"[{index:03d}/{total}] 上传完成: {upload_name} -> {document_id} "
            f"task={response.get('status')}/{response.get('stage')}",
            flush=True,
        )
    save_manifest(manifest)
    print(f"上传小结: 新上传={uploaded} 更新={updated} 跳过={skipped} 重新排队={requeued}")
    return tracked


def delete_documents(base_url: str, *, filenames: list[str] | None = None, purge: bool = False, timeout: float) -> int:
    """删除知识库文档：按文件名删除（--delete），或清空全部（--purge）。"""
    payload = request_json("GET", f"{base_url}/api/documents", timeout=timeout)
    documents = payload.get("documents") or []
    targets = [document for document in documents if purge or document.get("filename") in (filenames or [])]
    if purge:
        print_header(f"清空知识库（共 {len(targets)} 个文档）")
    else:
        print_header(f"删除文档（共 {len(targets)} 个匹配）")
    deleted = 0
    for document in targets:
        document_id = str(document["document_id"])
        for attempt in range(4):
            try:
                request_json("DELETE", f"{base_url}/api/documents/{document_id}", timeout=timeout)
                deleted += 1
                print(f"  已删除: {document.get('filename')} ({document_id})")
                break
            except RuntimeError as exc:
                if "索引" in str(exc) and attempt < 3:
                    print(f"    文档正在索引中，{15 * (attempt + 1)} 秒后重试...")
                    time.sleep(15 * (attempt + 1))
                    continue
                print(f"  删除失败: {document.get('filename')} ({document_id}): {str(exc)[:100]}")
                break
    if purge:
        save_manifest({})
    print(f"删除完成: {deleted} 个")
    return deleted


def wait_for_indexing(
    *,
    base_url: str,
    tracked_document_ids: dict[str, str],
    timeout_minutes: float,
    poll_interval: float,
    request_timeout: float,
) -> None:
    print_header("等待索引完成")
    deadline = time.time() + timeout_minutes * 60
    tracked_ids = set(tracked_document_ids.values())
    last_print = 0.0
    terminal_failures: list[dict[str, Any]] = []

    while time.time() < deadline:
        documents = (
            request_json("GET", f"{base_url}/api/documents", timeout=request_timeout).get("documents") or []
        )
        tracked_docs = [document for document in documents if document.get("document_id") in tracked_ids]
        counts = Counter(str(document.get("status") or "unknown") for document in tracked_docs)
        indexed = counts.get("indexed", 0)
        failed = counts.get("index_failed", 0) + counts.get("source_missing", 0)

        now = time.time()
        if now - last_print >= poll_interval:
            print(
                f"索引进度: indexed={indexed}/{len(tracked_ids)}, "
                f"queued={counts.get('index_queued', 0)}, indexing={counts.get('indexing', 0)}, "
                f"failed={failed}",
                flush=True,
            )
            last_print = now

        if indexed + failed >= len(tracked_ids):
            if failed:
                for document in tracked_docs:
                    if str(document.get("status")) in {"index_failed", "source_missing"}:
                        terminal_failures.append(document)
            print(f"索引完成: indexed={indexed} failed={failed}")
            break
        time.sleep(poll_interval)
    else:
        raise RuntimeError(
            f"索引超时（{timeout_minutes:.0f} 分钟）。当前 indexed={indexed}/{len(tracked_ids)} failed={failed}，"
            "请到网页查看失败文档的详细信息。"
        )

    if terminal_failures:
        print("\n失败文档：")
        for document in terminal_failures:
            print(
                f"  - {document.get('filename')} ({document.get('document_id')}): "
                f"{document.get('index_error') or document.get('status')}"
            )
        print("可修正后重跑本脚本（同名文件会自动重新排队索引）。")


def qdrant_collection_points(base_url: str, timeout: float) -> int | None:
    try:
        health = request_json("GET", f"{base_url}/api/health/ready", timeout=timeout)
    except Exception:
        return None
    collection = str(health.get("qdrant_collection") or "")
    if not collection:
        return None
    candidate_urls: list[str] = []
    qdrant_port = os.environ.get("RESUME_QDRANT_HTTP_PORT", "6333")
    if base_url.startswith("http://127.0.0.1") or base_url.startswith("http://localhost"):
        candidate_urls.append(f"http://127.0.0.1:{qdrant_port}/collections/{collection}")
    candidate_urls.append(f"http://qdrant:6333/collections/{collection}")
    for url in dict.fromkeys(candidate_urls):
        try:
            payload = request_json("GET", url, timeout=timeout)
            return int(payload.get("result", {}).get("points_count") or 0)
        except Exception:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="上传知识库文档到 ResumeMind")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="扫描根目录（默认项目根，即 docs/ 会被收录）")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--ingest-timeout-minutes", type=float, default=120.0)
    parser.add_argument("--admin-password", default=os.getenv("ADMIN_PASSWORD", ""),
                        help="管理员密码（默认取环境变量 ADMIN_PASSWORD）；知识库接口需管理员登录")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--delete", action="append", dest="delete_names", metavar="文件名",
                        help="按文件名删除知识库中的文档（可多次指定）")
    parser.add_argument("--purge", action="store_true", help="清空知识库中所有文档")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    root = args.root.resolve()

    print_header("ResumeMind 知识库管理")
    print(f"项目目录: {PROJECT_ROOT}")
    print(f"服务地址: {base_url}")
    assert_service_available(base_url, args.timeout)

    # 前台/后台权限分离：知识库管理接口需管理员登录（登录失败即终止）
    if not args.admin_password:
        raise RuntimeError(
            "缺少管理员密码：知识库接口已收进管理员后台，请用 --admin-password 或环境变量 ADMIN_PASSWORD 提供。"
        )
    admin_login(base_url, args.admin_password, args.timeout)

    # 删除 / 清空模式
    if args.purge or args.delete_names:
        delete_documents(
            base_url,
            filenames=args.delete_names,
            purge=args.purge,
            timeout=args.timeout,
        )
        return 0

    print(f"扫描根目录: {root}")
    files, images = scan_documents(root)
    if not files:
        raise RuntimeError(f"{root} 下未找到可上传的文档（支持 {sorted(SUPPORTED_EXTENSIONS)}）。")
    print(f"待上传文档: {len(files)} 个")
    for path in files:
        print(f"  - {path.relative_to(root)}")

    if images:
        print(f"\n发现 {len(images)} 个图片文件（无 OCR 解析能力，跳过）：")
        for path in images:
            print(f"  - {path.relative_to(root)}")
        print("建议：将图片内容转成 PDF 或文本后放入知识库。")

    if not args.skip_warmup:
        warmup_models(base_url, args.timeout)

    tracked = upload_files(base_url, files, args.timeout, root=root)
    wait_for_indexing(
        base_url=base_url,
        tracked_document_ids=tracked,
        timeout_minutes=args.ingest_timeout_minutes,
        poll_interval=args.poll_interval,
        request_timeout=args.timeout,
    )

    points = qdrant_collection_points(base_url, args.timeout)
    if points is not None:
        print(f"\nQdrant collection 向量数: {points}")
    print("\n知识库上传完成。可以在网页「智能问答」页开始提问。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)
