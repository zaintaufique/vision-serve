"""One-off analysis: how similar are the images to each other?

Reads the manifest, computes every pairwise perceptual-hash distance,
and prints the distribution. Writes nothing. Run it, read it, record
the conclusion in DATASET.md, then the threshold goes into
prepare_data.py as a constant.
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

MANIFEST = Path(__file__).parent / "manifests" / "manifest.csv"


def load_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as handle:
        return list(csv.DictReader(handle))


def hamming(a: int, b: int) -> int:
    """Number of bit positions where two hashes differ (0-64)."""
    return (a ^ b).bit_count()


def main() -> None:
    rows = load_rows()
    hashes = [(row["path"], int(row["phash"], 16)) for row in rows]
    print(f"{len(hashes)} images")

    # --- exact byte duplicates, from sha256 ---
    by_sha: dict[str, list[str]] = {}
    for row in rows:
        by_sha.setdefault(row["sha256"], []).append(row["path"])
    exact = {k: v for k, v in by_sha.items() if len(v) > 1}
    print(f"exact byte-duplicate groups: {len(exact)}")
    for paths in list(exact.values())[:5]:
        print(f"  {paths}")

    # --- all pairwise perceptual distances ---
    counts = [0] * 65
    closest: list[tuple[int, str, str]] = []
    for (path_a, hash_a), (path_b, hash_b) in combinations(hashes, 2):
        distance = hamming(hash_a, hash_b)
        counts[distance] += 1
        if distance <= 8:
            closest.append((distance, path_a, path_b))

    total = sum(counts)
    print(f"\n{total} pairs compared\n")

    print("distance  pairs      histogram")
    peak = max(counts) or 1
    for distance, count in enumerate(counts):
        if count == 0:
            continue
        bar = "#" * max(1, round(60 * count / peak))
        print(f"{distance:>8}  {count:>9}  {bar}")

    print("\nclosest pairs (distance <= 8):")
    for distance, path_a, path_b in sorted(closest)[:30]:
        print(f"  d={distance:<3} {path_a}  <->  {path_b}")
    if not closest:
        print("  none")


if __name__ == "__main__":
    main()
