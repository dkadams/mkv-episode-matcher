import json

from mkv_episode_matcher.segment_quality import (
    analyze_segment_quality,
    embeddings_sidecar_path,
    load_low_info_intervals,
    write_low_info_sidecar,
)


def test_analyze_segment_quality_flags_short_and_cue_heavy_text():
    quality = analyze_segment_quality(
        "(mysterious music) ha ha",
        min_words=8,
        cue_ratio_threshold=0.25,
    )
    assert quality.is_low_info is True
    assert "short_text" in quality.reasons
    assert "cue_heavy" in quality.reasons


def test_analyze_segment_quality_accepts_normal_dialog():
    quality = analyze_segment_quality(
        "I will call you tomorrow morning after we review the rehearsal script together.",
        min_words=8,
        cue_ratio_threshold=0.25,
    )
    assert quality.is_low_info is False
    assert quality.reasons == ()


def test_sidecar_round_trip(tmp_path):
    embeddings_path = tmp_path / "sample.npy"
    embeddings_path.write_bytes(b"npy-placeholder")
    quality = {
        1: analyze_segment_quality("(sad music)", 8, 0.25),
        2: analyze_segment_quality(
            "normal dialog with enough unique words to avoid filtering",
            8,
            0.25,
        ),
    }
    write_low_info_sidecar(
        embeddings_path,
        interval_quality=quality,
        min_words=8,
        cue_ratio_threshold=0.25,
    )

    sidecar = embeddings_sidecar_path(embeddings_path)
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert "1" in payload["low_info_intervals"]
    assert "2" not in payload["low_info_intervals"]

    assert load_low_info_intervals(embeddings_path) == {1}
