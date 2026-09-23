"""Content-addressed, immutable raw storage (FR-3/FR-6). The path is derived from the bytes,
so a stored object never changes and writing the same bytes twice is a no-op."""

import hashlib
import os
import tempfile

from ledgerline.config import settings


def put_object(data: bytes, suffix: str = "") -> tuple[str, str]:
    sha = hashlib.sha256(data).hexdigest()
    rel = f"objects/{sha[:2]}/{sha}{suffix}"
    dest = settings().data_dir / rel
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash never leaves a truncated file under a valid hash.
        fd, tmp = tempfile.mkstemp(dir=dest.parent)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
    return sha, rel


def read_object(path: str) -> bytes:
    return (settings().data_dir / path).read_bytes()
