from __future__ import annotations

import bz2
import gzip
import hashlib
import os
import shutil
import urllib.request
import urllib.error
import ssl
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DownloadedFile:
    path: Path
    decompressed_path: Path


def _repo_root_from_this_file() -> Path:
    # This file is: research/code/distopt/datasets/download.py
    # repo root is 4 parents up.
    return Path(__file__).resolve().parents[4]


def get_data_dir(*, env_var: str = "DISTOPT_DATA_DIR") -> Path:
    """Return dataset cache directory.

    Precedence:
    1) `$DISTOPT_DATA_DIR`
    2) `<repo_root>/data`
    """

    env = os.environ.get(env_var)
    if env:
        return Path(env).expanduser().resolve()
    return _repo_root_from_this_file() / "data"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest_path: Path, *, sha256: str | None = None, timeout_s: int = 60) -> Path:
    """Download `url` to `dest_path` (idempotent).

    If `sha256` is provided and the file exists, the hash is checked.
    """

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        if sha256 is None:
            return dest_path
        if _sha256(dest_path) == sha256:
            return dest_path
        # Hash mismatch -> re-download.
        dest_path.unlink()

    verify_ssl = os.environ.get("DISTOPT_INSECURE_DOWNLOAD", "").strip() not in {"1", "true", "TRUE", "yes", "YES"}

    if verify_ssl:
        # Prefer certifi if available because some Python distributions (e.g. uv/venv)
        # may not be configured with a working system CA bundle.
        try:
            import certifi  # type: ignore

            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = ssl.create_default_context()
    else:
        ctx = ssl._create_unverified_context()  # noqa: SLF001

    req = urllib.request.Request(url, headers={"User-Agent": "distopt-dataset-downloader"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s, context=ctx) as r, dest_path.open("wb") as f:
            shutil.copyfileobj(r, f)
    except ssl.SSLCertVerificationError as e:
        raise RuntimeError(
            "SSL certificate verification failed while downloading a dataset. "
            "If you're in a controlled environment, you can opt out by setting "
            "DISTOPT_INSECURE_DOWNLOAD=1 (not recommended for untrusted networks)."
        ) from e
    except urllib.error.URLError as e:
        # urllib commonly wraps TLS failures into URLError(reason=SSLCertVerificationError).
        if isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
            raise RuntimeError(
                "SSL certificate verification failed while downloading a dataset. "
                "If you're in a controlled environment, you can opt out by setting "
                "DISTOPT_INSECURE_DOWNLOAD=1 (not recommended for untrusted networks)."
            ) from e
        raise

    if sha256 is not None:
        got = _sha256(dest_path)
        if got != sha256:
            raise RuntimeError(f"SHA256 mismatch for {dest_path}: expected {sha256}, got {got}")

    return dest_path


def decompress_if_needed(path: Path) -> Path:
    """Decompress `.bz2` / `.gz` to a sibling file and return the decompressed path."""

    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".bz2":
        out = path.with_suffix("")
        if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
            return out
        with bz2.open(path, "rb") as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return out

    if suffix == ".gz":
        out = path.with_suffix("")
        if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
            return out
        with gzip.open(path, "rb") as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return out

    return path


# Minimal registry for datasets referenced in the ACC-SONATA/MUDAG setup summary.
_LIBSVM_URLS: dict[str, str] = {
    # Ye et al. (MUDAG) uses a9a/w8a.
    "a9a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a",
    "w8a": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a",
    # Tian et al. (ACC-SONATA) mentions SUSY (logistic), but we use it for ridge-quadratic conversion.
    "SUSY": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY",
}


def download_libsvm_dataset(name: str, *, data_dir: Path | None = None) -> DownloadedFile:
    """Download a LIBSVM dataset and return the (possibly decompressed) path.

    The file is cached under `<data_dir>/libsvm/`.

    Notes:
    - Some LIBSVM datasets may be large (e.g., SUSY). Start with `a9a`.
    - If a dataset is hosted as a compressed file, this function will still work
      if `name` maps to the compressed URL; currently the registry points to the
      plain dataset endpoints.
    """

    if data_dir is None:
        data_dir = get_data_dir()

    if name not in _LIBSVM_URLS:
        raise ValueError(f"Unknown LIBSVM dataset {name!r}. Known: {sorted(_LIBSVM_URLS)}")

    url = _LIBSVM_URLS[name]
    dest = Path(data_dir) / "libsvm" / name
    downloaded = download(url, dest)
    decompressed = decompress_if_needed(downloaded)
    return DownloadedFile(path=downloaded, decompressed_path=decompressed)
