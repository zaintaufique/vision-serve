"""Build the casting dataset manifest.

Reads data/raw/casting/ and records facts about every image.
Never copies, moves, modifies, or deletes a raw file.

Two phases, run in order by main():
  build phase (scan -> verify -> hash) writes manifest.csv
  split phase (assign_splits)          writes manifest_split.csv

The split phase reads the manifest back, assigns train/val/test
stratified by label with a fixed seed, and writes a NEW file. It
never overwrites an existing split manifest.
"""

from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass, fields
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError

# Folder name on disk -> the label we use everywhere else.
LABEL_DIRS = {"def_front": "def_front", "ok_front": "ok_front"}
VALID_SUFFIXES = {".jpeg", ".jpg"}

# Split configuration. Test is the remainder, so it is never a separate
# ratio that could drift out of sync with the others.
SPLIT_SEED = 7
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


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


def read_manifest(source: Path) -> list[ImageRecord]:
    """Read a manifest CSV back into records, validating as it goes."""
    if not source.exists():
        raise FileNotFoundError(f"manifest not found: {source}")
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest is empty: {source}")

    records: list[ImageRecord] = []
    for row in rows:
        records.append(
            ImageRecord(
                path=row["path"],
                label=row["label"],
                split=row["split"],
                sha256=row["sha256"],
                phash=row["phash"],
                dup_group=row["dup_group"],
                # CSV stores everything as text, so the boolean came back
                # as the string "True"; convert it back to a real bool.
                is_active=row["is_active"] == "True",
                width=int(row["width"]),
                height=int(row["height"]),
            )
        )
    return records


def assign_splits(source: Path, destination: Path) -> None:
    """Fill the split column, stratified by label, reproducibly.

    Reads the raw manifest, assigns train/val/test within each label so
    the label ratio holds in every pile, and writes a NEW file. Refuses
    to overwrite an existing split manifest.
    """
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; delete it to regenerate the split")

    records = read_manifest(source)

    # Validate: labels known, and no split already assigned.
    for record in records:
        if record.label not in LABEL_DIRS:
            raise ValueError(f"unknown label {record.label!r} for {record.path}")
        if record.split:
            raise ValueError(f"split already set for {record.path}; expected empty")

    # Local generator, isolated from the global random module: nothing
    # else can perturb this sequence, so the shuffle is truly reproducible.
    rng = random.Random(SPLIT_SEED)

    for label in sorted(LABEL_DIRS):
        group = [r for r in records if r.label == label]
        group.sort(key=lambda r: r.path)  # pin order before shuffling
        rng.shuffle(group)

        n = len(group)
        n_train = round(TRAIN_RATIO * n)
        n_val = round(VAL_RATIO * n)
        # test is the remainder, so the three piles always sum to n.

        for i, record in enumerate(group):
            if i < n_train:
                record.split = "train"
            elif i < n_train + n_val:
                record.split = "val"
            else:
                record.split = "test"

    write_manifest(records, destination)

    # Report per-pile, per-label counts so the ratio is visible.
    print(f"wrote {destination}")
    for split in ("train", "val", "test"):
        by_label = {
            label: sum(1 for r in records if r.split == split and r.label == label)
            for label in sorted(LABEL_DIRS)
        }
        total = sum(by_label.values())
        print(f"  {split:5} {total:4}  {by_label}")


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

    split_destination = destination.parent / "manifest_split.csv"
    assign_splits(destination, split_destination)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[4]
    main(
        data_root=repo_root / "data" / "raw" / "casting_512x512" / "casting_512x512",
        destination=Path(__file__).parent / "manifests" / "manifest.csv",
    )
