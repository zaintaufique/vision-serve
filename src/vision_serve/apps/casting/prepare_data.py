"""Build the casting dataset manifest.

Reads data/raw/casting/ and records facts about every image.
Never copies, moves, modifies, or deletes a raw file.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, fields
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError

# Folder name on disk -> the label we use everywhere else.
LABEL_DIRS = {"def_front": "def_front", "ok_front": "ok_front"}

VALID_SUFFIXES = {".jpeg", ".jpg"}


@dataclass
class ImageRecord:
    """One row of the manifest."""

    path: str  # relative to the data root, never absolute
    label: str
    split: str = ""  # filled in later, by assign_splits()
    sha256: str = ""
    phash: str = ""
    dup_group: str = ""
    is_active: bool = True
    width: int = 0
    height: int = 0


def scan_files(data_root: Path) -> list[Path]:
    """Find every candidate image under the label folders."""
    found: list[Path] = []
    for label_dir in sorted(LABEL_DIRS):
        folder = data_root / label_dir
        if not folder.is_dir():
            raise FileNotFoundError(f"expected label folder: {folder}")
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in VALID_SUFFIXES:
                found.append(path)
    return found


def compute_sha256(path: Path) -> str:
    """Fingerprint of the exact bytes on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_image(path: Path, data_root: Path) -> ImageRecord:
    """Open one image, confirm it is readable, record its facts.

    Raises ValueError if the file is not a usable image.
    """
    try:
        with Image.open(path) as img:
            img.load()  # forces a full decode; a truncated file fails here
            width, height = img.size
            phash = str(imagehash.dhash(img))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"unreadable image: {path}") from exc

    return ImageRecord(
        path=str(path.relative_to(data_root)),
        label=LABEL_DIRS[path.parent.name],
        sha256=compute_sha256(path),
        phash=phash,
        width=width,
        height=height,
    )


def write_manifest(records: list[ImageRecord], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = [f.name for f in fields(ImageRecord)]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def main(data_root: Path, destination: Path) -> None:
    paths = scan_files(data_root)
    print(f"scanned {len(paths)} files")

    records: list[ImageRecord] = []
    failures: list[str] = []
    for path in paths:
        try:
            records.append(verify_image(path, data_root))
        except ValueError as exc:
            failures.append(str(exc))

    print(f"verified {len(records)} images, {len(failures)} failed")
    for message in failures:
        print(f"  FAILED {message}")

    sizes = {(r.width, r.height) for r in records}
    print(f"distinct image sizes: {sorted(sizes)}")

    counts: dict[str, int] = {}
    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1
    print(f"label counts: {counts}")

    write_manifest(records, destination)
    print(f"wrote {destination}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[4]
    main(
        data_root=repo_root / "data" / "raw" / "casting_512x512" / "casting_512x512",
        destination=Path(__file__).parent / "manifests" / "manifest.csv",
    )
