from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Tạo fingerprint tái lập để nhận diện chính xác phiên bản dữ liệu."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
