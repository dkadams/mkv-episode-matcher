import math
from pathlib import Path
from typing import Iterable

import pysubs2
from loguru import logger
from pysubs2 import SSAFile, SSAEvent

from mkv_episode_matcher.config import Configuration
from mkv_episode_matcher.episode import EpisodeKey
from mkv_episode_matcher.series import Series


class SubtitleFixedIntervalizer:
    def __init__(self, config: Configuration, series: Series,
        interval_seconds: int):
        self.config = config
        self.series = series
        self.interval_seconds = interval_seconds

    def execute(self, episode: EpisodeKey, input: Path, output: Path):
        logger.info(f"Extracting embeddings for episode: {self.series.name},"
                    f" episode: {episode}")

        orig_subs = pysubs2.load(str(input), format_="srt")
        max_ts_seconds = orig_subs[-1].end / 1000
        details = self.series.get_episode_detail(episode, keys=["runtime"])
        if details["runtime"]:
            runtime_minutes = min(max_ts_seconds, int(details["runtime"]) * 60)
        else:
            runtime_minutes = max_ts_seconds

        interval_count = math.ceil(runtime_minutes / self.interval_seconds)
        indexed_intervals = self.get_indexed_intervals(interval_count)

        interval_subs = SSAFile()
        for index, interval in indexed_intervals:
            # There are no EOL characters in the transcribed text. So we
            # don't want any in the subtitle text, either.
            interval_text = " ".join(
                sub.plaintext.replace("\n", " ")
                for sub in orig_subs
                if sub.start in interval or sub.end in interval)

            # We still want to add an interval even if it's empty. That ensures
            # the indexes will be consistent between files
            event = SSAEvent(start=interval[0], end=interval[1],
                             text=interval_text)
            interval_subs.append(event)

        interval_subs.save(str(output), encoding="utf-8", format_="srt")

    def get_indexed_intervals(self, interval_count: int) -> Iterable[tuple[int, range]]:
        interval_ms = self.interval_seconds * 1000
        for i in range(interval_count):
            start = i * interval_ms
            yield i, range(start, start + interval_ms)
