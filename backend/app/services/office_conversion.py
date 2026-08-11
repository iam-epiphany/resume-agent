from pathlib import Path
from dataclasses import dataclass
import shutil
import subprocess
import tempfile
from time import perf_counter

from backend.app.core.config import (
    OFFICE_CONVERSION_MAX_BYTES,
    OFFICE_CONVERSION_TIMEOUT_SECONDS,
)


class OfficeConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class OfficeConversionResult:
    path: Path
    backend: str
    elapsed_ms: float


def convert_with_libreoffice_detailed(
    file_path: Path, target_extension: str
) -> OfficeConversionResult:
    """Convert with an isolated profile so concurrent headless jobs cannot lock each other."""

    executable = _libreoffice_executable()
    target_extension = target_extension.lower().lstrip(".")
    output_dir = Path(tempfile.mkdtemp(prefix="resumemind-office-"))
    profile_dir = Path(tempfile.mkdtemp(prefix="resumemind-lo-profile-"))
    started = perf_counter()
    command = [
        executable,
        "--headless",
        "--nologo",
        "--nodefault",
        "--nofirststartwizard",
        f"-env:UserInstallation={profile_dir.resolve().as_uri()}",
        "--convert-to",
        target_extension,
        "--outdir",
        str(output_dir),
        str(file_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=OFFICE_CONVERSION_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise OfficeConversionError(f"LibreOffice conversion failed: {exc}") from exc
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        shutil.rmtree(output_dir, ignore_errors=True)
        raise OfficeConversionError(f"LibreOffice conversion failed: {detail}")

    expected = output_dir / f"{file_path.stem}.{target_extension}"
    if expected.exists():
        _validate_conversion_output(expected, output_dir)
        return OfficeConversionResult(expected, "libreoffice", round((perf_counter() - started) * 1000, 2))

    converted = sorted(output_dir.glob(f"*.{target_extension}"))
    if converted:
        _validate_conversion_output(converted[0], output_dir)
        return OfficeConversionResult(converted[0], "libreoffice", round((perf_counter() - started) * 1000, 2))
    shutil.rmtree(output_dir, ignore_errors=True)
    raise OfficeConversionError("LibreOffice conversion did not produce an output file")


def _libreoffice_executable() -> str:
    for candidate in ("soffice", "libreoffice"):
        path = shutil.which(candidate)
        if path:
            return path
    raise OfficeConversionError("LibreOffice executable not found")


def office_tool_status() -> dict[str, str | bool | None]:
    libreoffice = next((shutil.which(name) for name in ("soffice", "libreoffice") if shutil.which(name)), None)
    antiword = shutil.which("antiword")
    return {
        "libreoffice_ready": bool(libreoffice),
        "libreoffice_path": libreoffice,
        "libreoffice_version": _tool_version(libreoffice, "--version") if libreoffice else None,
        "antiword_ready": bool(antiword),
        "antiword_path": antiword,
        "antiword_version": _tool_version(antiword, "-h") if antiword else None,
    }


def cleanup_conversion_output(converted_path: Path) -> None:
    """Remove only temporary directories created by this module.

    The prefix and temp-root checks keep mocked or caller-owned paths safe.
    """

    parent = converted_path.resolve().parent
    temp_root = Path(tempfile.gettempdir()).resolve()
    if parent.name.startswith("resumemind-office-") and parent.is_relative_to(temp_root):
        shutil.rmtree(parent, ignore_errors=True)


def _validate_conversion_output(path: Path, output_dir: Path) -> None:
    size = path.stat().st_size
    if size == 0:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise OfficeConversionError("LibreOffice conversion produced an empty output file")
    if size > OFFICE_CONVERSION_MAX_BYTES:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise OfficeConversionError(
            f"LibreOffice conversion output exceeds {OFFICE_CONVERSION_MAX_BYTES} bytes"
        )


def _tool_version(executable: str, flag: str) -> str | None:
    try:
        completed = subprocess.run(
            [executable, flag], check=False, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (completed.stdout or completed.stderr or "").strip().splitlines()
    return text[0][:200] if text else None
