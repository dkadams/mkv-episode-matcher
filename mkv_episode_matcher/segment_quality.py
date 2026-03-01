from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")

DEFAULT_CUE_TOKENS = {
    "music",
    "laugh",
    "laughs",
    "laughing",
    "applause",
    "clapping",
    "cheering",
    "upbeat",
    "sad",
    "mysterious",
    "theme",
    "sings",
    "singing",
    "sigh",
    "sighs",
    "groan",
    "groans",
    "gasp",
    "gasps",
    "crying",
    "whispering",
    "shouting",
}

LOW_INFO_METADATA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SegmentQuality:
    is_low_info: bool
    reasons: tuple[str, ...]
    token_count: int
    cue_ratio: float
    unique_token_ratio: float


def analyze_segment_quality(
    text: str,
    min_words: int,
    cue_ratio_threshold: float,
    cue_tokens: set[str] | None = None,
) -> SegmentQuality:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text or "")]
    token_count = len(tokens)
    unique_token_ratio = (len(set(tokens)) / token_count) if token_count else 0.0
    cue_tokens = cue_tokens or DEFAULT_CUE_TOKENS
    cue_count = sum(1 for token in tokens if token in cue_tokens)
    cue_ratio = (cue_count / token_count) if token_count else 0.0

    reasons: list[str] = []
    if token_count < int(min_words):
        reasons.append("short_text")
    if token_count > 0 and cue_ratio >= float(cue_ratio_threshold):
        reasons.append("cue_heavy")
    if token_count < 20 and unique_token_ratio < 0.40:
        reasons.append("low_diversity")

    return SegmentQuality(
        is_low_info=bool(reasons),
        reasons=tuple(reasons),
        token_count=token_count,
        cue_ratio=cue_ratio,
        unique_token_ratio=unique_token_ratio,
    )


def embeddings_sidecar_path(embeddings_path: Path) -> Path:
    return embeddings_path.with_suffix(".meta.json")


def write_low_info_sidecar(
    embeddings_path: Path,
    interval_quality: dict[int, SegmentQuality],
    min_words: int,
    cue_ratio_threshold: float,
):
    payload = {
        "schema_version": LOW_INFO_METADATA_SCHEMA_VERSION,
        "low_info_filter": {
            "min_words": int(min_words),
            "cue_ratio_threshold": float(cue_ratio_threshold),
        },
        "low_info_intervals": {
            str(interval_index): {
                "reasons": list(quality.reasons),
                "token_count": int(quality.token_count),
                "cue_ratio": float(quality.cue_ratio),
                "unique_token_ratio": float(quality.unique_token_ratio),
            }
            for interval_index, quality in sorted(interval_quality.items())
            if quality.is_low_info
        },
    }
    sidecar_path = embeddings_sidecar_path(embeddings_path)
    with sidecar_path.open("w", encoding="utf-8") as out:
        json.dump(payload, out, ensure_ascii=False, indent=2)


def load_low_info_intervals(embeddings_path: Path) -> set[int] | None:
    sidecar_path = embeddings_sidecar_path(embeddings_path)
    if not sidecar_path.exists():
        return None

    with sidecar_path.open("r", encoding="utf-8") as sidecar_in:
        payload = json.load(sidecar_in)
    intervals = payload.get("low_info_intervals", {})
    return {int(interval_index) for interval_index in intervals.keys()}
