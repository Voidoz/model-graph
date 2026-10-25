"""Exact and simple near-duplicate detection for dataset examples.

Near-duplicates use a 64-bit SimHash over token shingles and a configurable Hamming-distance
threshold. This is intentionally simple and dependency-free: the goal is to stop obviously
repeated trace text from inflating a training or eval split, not to build a fuzzy matcher.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_WHITESPACE_RE = re.compile(r"\s+")
SHINGLE_SIZE = 3
SIMHASH_BITS = 64


def normalize_text(value: str) -> str:
    """Casefold and collapse whitespace so trivial formatting does not defeat exact dedupe."""
    return _WHITESPACE_RE.sub(" ", value.strip().lower())


def tokens(value: str) -> list[str]:
    return _TOKEN_RE.findall(normalize_text(value))


def shingles(value: str, *, size: int = SHINGLE_SIZE) -> list[str]:
    words = tokens(value)
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    return [" ".join(words[index : index + size]) for index in range(len(words) - size + 1)]


def simhash(value: str) -> int:
    """Return a deterministic 64-bit SimHash of the token shingles in ``value``."""
    vector = [0] * SIMHASH_BITS
    for shingle in shingles(value):
        digest = hashlib.sha256(shingle.encode("utf-8")).digest()
        bits = int.from_bytes(digest[:8], "big")
        for bit in range(SIMHASH_BITS):
            vector[bit] += 1 if bits & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(vector):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


@dataclass(frozen=True)
class DedupeResult:
    kept: list[int]
    exact_removed: int = 0
    near_removed: int = 0
    removed: list[dict[str, object]] = field(default_factory=list)

    @property
    def removed_count(self) -> int:
        return self.exact_removed + self.near_removed


def deduplicate(
    values: Sequence[str],
    *,
    mode: str = "exact+near",
    near_distance: int = 3,
) -> DedupeResult:
    """Deduplicate ``values`` deterministically, keeping the earliest item in each cluster.

    ``mode`` is ``none``, ``exact``, ``near``, or ``exact+near``. Exact duplicates are detected
    on normalized text. Near-duplicates are connected components of the pairwise SimHash
    distance graph, so a transitive chain cannot leave two near-identical items in the output.
    """
    if mode not in {"none", "exact", "near", "exact+near"}:
        raise ValueError(f"unknown dedupe mode {mode!r}")
    if near_distance < 0 or near_distance > SIMHASH_BITS:
        raise ValueError(f"near-duplicate distance must be between 0 and {SIMHASH_BITS}")
    if mode == "none" or not values:
        return DedupeResult(kept=list(range(len(values))))

    exact_enabled = "exact" in mode
    near_enabled = "near" in mode
    removed: list[dict[str, object]] = []
    exact_removed = 0
    candidate_indices: list[int] = []
    first_by_signature: dict[str, int] = {}
    for index, value in enumerate(values):
        if exact_enabled:
            signature = normalize_text(value)
            representative = first_by_signature.get(signature)
            if representative is not None:
                exact_removed += 1
                removed.append(
                    {
                        "index": index,
                        "reason": "exact",
                        "representative": representative,
                    }
                )
                continue
            first_by_signature[signature] = index
        candidate_indices.append(index)

    if not near_enabled or len(candidate_indices) < 2:
        return DedupeResult(
            kept=sorted(candidate_indices),
            exact_removed=exact_removed,
            near_removed=0,
            removed=removed,
        )

    hashes = {index: simhash(values[index]) for index in candidate_indices}
    parent = {index: index for index in candidate_indices}

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for position, left in enumerate(candidate_indices):
        for right in candidate_indices[position + 1 :]:
            if hamming_distance(hashes[left], hashes[right]) <= near_distance:
                union(left, right)

    clusters: dict[int, list[int]] = {}
    for index in candidate_indices:
        clusters.setdefault(find(index), []).append(index)
    near_removed = 0
    for members in clusters.values():
        members.sort()
        representative = members[0]
        for index in members[1:]:
            near_removed += 1
            removed.append(
                {
                    "index": index,
                    "reason": "near-duplicate",
                    "representative": representative,
                }
            )

    return DedupeResult(
        kept=[index for index in candidate_indices if find(index) == index],
        exact_removed=exact_removed,
        near_removed=near_removed,
        removed=sorted(removed, key=lambda item: int(str(item["index"]))),
    )
