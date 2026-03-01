from argparse import Namespace
from configparser import ConfigParser

import pysubs2
import pytest
from pysubs2 import SSAEvent, SSAFile

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.subtitle_fixed_intervalizer import SubtitleFixedIntervalizer


class DummySeries:
    name = "Dummy"

    @staticmethod
    def get_episode_detail(_episode: EpisodeKey, keys=None):
        return {"runtime": None}


def test_intervalizer_uses_sliding_windows_and_true_overlap(tmp_path):
    input_path = tmp_path / "in.srt"
    output_path = tmp_path / "out.srt"

    subs = SSAFile()
    # Crosses the 25s stride boundary and should appear in both windows.
    subs.append(SSAEvent(start=29000, end=31000, text="boundary text"))
    subs.save(str(input_path), format_="srt", encoding="utf-8")

    config = Configuration(args=Namespace(), stored=ConfigParser())
    intervalizer = SubtitleFixedIntervalizer(
        config=config,
        series=DummySeries(),
        interval_seconds=30,
        subtitle_overlap_seconds=5,
    )
    intervalizer.execute(EpisodeKey(1, 1), input_path, output_path)

    out = pysubs2.load(str(output_path), format_="srt")
    assert len(out) == 2
    assert "boundary text" in out[0].plaintext
    assert "boundary text" in out[1].plaintext


def test_intervalizer_rejects_invalid_overlap():
    config = Configuration(args=Namespace(), stored=ConfigParser())
    with pytest.raises(ValueError):
        SubtitleFixedIntervalizer(
            config=config,
            series=DummySeries(),
            interval_seconds=30,
            subtitle_overlap_seconds=30,
        )
