import json
from pathlib import Path
from types import SimpleNamespace

from mkv_episode_matcher.dataset_collector import _collect_series_dataset


class VideoInfo:
    def __init__(self, minutes):
        self.minutes = minutes


def test_collect_dataset_writes_aligned_and_misaligned_variants(monkeypatch, tmp_path):
    series_dir = tmp_path / "series"
    series_dir.mkdir()
    subtitles_dir = series_dir / ".mkv-episode-matcher" / "subtitles"
    subtitles_dir.mkdir(parents=True)
    (subtitles_dir / "Show - S01E01.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")
    video = series_dir / "Show.S01E01.mkv"
    video.write_text("")

    aligned_src = tmp_path / "aligned-src.json"
    aligned_src.write_text(json.dumps({"0": "aligned text"}))
    left_src = tmp_path / "left-src.json"
    left_src.write_text(json.dumps({"0": "misaligned text"}))

    class FakeMatcher:
        def __init__(self, _config, _series):
            pass

        @staticmethod
        def _collect_files(_paths):
            return [video]

        def get_transcriptions(self, _progress, videos, misalignment_policy=None, **_kwargs):
            info = {videos[0]: VideoInfo(22.0)}
            if misalignment_policy is None:
                # first pass for video info + aligned pass
                return info, {videos[0]: aligned_src}
            return info, {videos[0]: left_src}

    monkeypatch.setattr(
        "mkv_episode_matcher.dataset_collector.IndexedEpisodeMatcher",
        FakeMatcher,
    )
    monkeypatch.setattr(
        "mkv_episode_matcher.dataset_collector.get_specified_episodes",
        lambda _config, _series: [SimpleNamespace(key=lambda: (1, 1))],
    )

    output_dir = tmp_path / "dataset"
    args = SimpleNamespace(
        segment_duration=None,
        output_dir=str(output_dir),
        misalign_profiles=["left"],
        misalign_min_seconds=1.0,
        misalign_max_seconds=3.0,
        misalign_seed=7,
        misalign_per_segment=False,
        include_aligned=True,
        segments_per_minute=0.5,
    )
    config = SimpleNamespace(args=args)
    series = SimpleNamespace(
        dir=series_dir,
        name="Show",
        random_seed=12345,
        segment_duration=30,
        subtitles_dir=subtitles_dir,
    )

    _collect_series_dataset(config, series, [series_dir])

    assert (output_dir / "transcriptions" / "text").exists()
    assert (output_dir / "transcriptions" / "misaligned" / "left").exists()
    assert (output_dir / "subtitles" / "srt").exists()

    rows = [json.loads(line) for line in (output_dir / "manifest.jsonl").read_text().splitlines()]
    profiles = {row["variant_profile"] for row in rows}
    assert profiles == {"aligned", "left"}
