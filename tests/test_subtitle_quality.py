import pysubs2

from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.subtitle_quality import (
    SubtitleQualitySettings,
    candidate_quality_diagnostics,
    choose_best_attempt,
    evaluate_subtitle_signals,
    save_episode_quality,
    subtitle_file_entries,
)


def _write_srt(path, lines, end_minutes: float):
    subs = pysubs2.SSAFile()
    end_ms = int(end_minutes * 60_000)
    step = max(1, end_ms // max(1, len(lines)))
    for idx, line in enumerate(lines):
        start = idx * step
        end = min(end_ms, start + step)
        subs.append(pysubs2.SSAEvent(start=start, end=end, text=line))
    subs.save(str(path), encoding="utf-8", format_="srt")


def test_runtime_ratio_hard_fail_for_long_subtitle(tmp_path):
    srt = tmp_path / "S01E01.srt"
    _write_srt(srt, ["line a", "line b"], end_minutes=42.0)
    settings = SubtitleQualitySettings()
    metrics, lines = evaluate_subtitle_signals(
        srt,
        expected_runtime_minutes=22.0,
        settings=settings,
    )
    diagnostics = candidate_quality_diagnostics(
        episode_key=EpisodeKey(1, 1),
        candidate_metrics=metrics,
        candidate_lines=lines,
        neighbor_metrics_by_episode={},
        neighbor_lines_by_episode={},
        settings=settings,
        metadata_episode_mismatch=False,
    )
    assert metrics.runtime_hard_fail is True
    assert diagnostics["hard_fail"] is True
    assert "runtime_ratio_out_of_bounds" in diagnostics["hard_fail_reasons"]


def test_overlap_with_runtime_pass_neighbor_blames_runtime_fail_candidate(tmp_path):
    settings = SubtitleQualitySettings()
    candidate = tmp_path / "S01E20.srt"
    neighbor = tmp_path / "S01E21.srt"
    shared_lines = [f"shared line {idx}" for idx in range(1, 120)]
    _write_srt(candidate, shared_lines + ["candidate tail"], end_minutes=42.0)
    _write_srt(neighbor, shared_lines, end_minutes=21.0)
    cand_metrics, cand_lines = evaluate_subtitle_signals(
        candidate,
        expected_runtime_minutes=22.0,
        settings=settings,
    )
    nei_metrics, nei_lines = evaluate_subtitle_signals(
        neighbor,
        expected_runtime_minutes=22.0,
        settings=settings,
    )
    diagnostics = candidate_quality_diagnostics(
        episode_key=EpisodeKey(1, 20),
        candidate_metrics=cand_metrics,
        candidate_lines=cand_lines,
        neighbor_metrics_by_episode={EpisodeKey(1, 21): nei_metrics},
        neighbor_lines_by_episode={EpisodeKey(1, 21): nei_lines},
        settings=settings,
        metadata_episode_mismatch=False,
    )
    assert diagnostics["hard_fail"] is True
    assert "overlap_with_neighbor_runtime_pass" in diagnostics["hard_fail_reasons"]


def test_subtitle_file_entries_exclude_quarantined_by_default(tmp_path):
    subtitles_dir = tmp_path / "subtitles"
    subtitles_dir.mkdir(parents=True)
    _write_srt(subtitles_dir / "Show - S01E01.srt", ["one"], end_minutes=21.0)
    _write_srt(subtitles_dir / "Show - S01E02.srt", ["two"], end_minutes=21.0)
    save_episode_quality(
        subtitles_dir,
        EpisodeKey(1, 2),
        {
            "episode": "S01E02",
            "verdict": "quarantined",
            "attempts": [],
        },
    )

    kept = subtitle_file_entries(subtitles_dir, include_quarantined=False)
    kept_keys = {episode for episode, _ in kept}
    assert kept_keys == {EpisodeKey(1, 1)}

    with_quarantined = subtitle_file_entries(subtitles_dir, include_quarantined=True)
    all_keys = {episode for episode, _ in with_quarantined}
    assert all_keys == {EpisodeKey(1, 1), EpisodeKey(1, 2)}


def test_choose_best_attempt_prefers_passing_candidate():
    attempts = [
        {
            "rank": 1,
            "subtitle": {
                "download_count": 200,
                "new_download_count": 0,
                "votes": 30,
                "ratings": 8.0,
                "from_trusted": True,
            },
            "quality": {
                "hard_fail": True,
                "soft_penalty_total": 0.01,
            },
        },
        {
            "rank": 2,
            "subtitle": {
                "download_count": 20,
                "new_download_count": 0,
                "votes": 5,
                "ratings": 4.0,
                "from_trusted": False,
            },
            "quality": {
                "hard_fail": False,
                "soft_penalty_total": 0.20,
            },
        },
    ]
    selected = choose_best_attempt(attempts)
    assert selected is not None
    assert selected["rank"] == 2
    assert selected["selection_verdict"] == "pass"


def test_choose_best_attempt_returns_quarantined_when_all_fail():
    attempts = [
        {
            "rank": 1,
            "subtitle": {"download_count": 100, "new_download_count": 0, "votes": 10, "ratings": 7.0},
            "quality": {"hard_fail": True, "soft_penalty_total": 0.1},
        },
        {
            "rank": 2,
            "subtitle": {"download_count": 80, "new_download_count": 0, "votes": 8, "ratings": 6.0},
            "quality": {"hard_fail": True, "soft_penalty_total": 0.2},
        },
    ]
    selected = choose_best_attempt(attempts)
    assert selected is not None
    assert selected["selection_verdict"] == "quarantined"
