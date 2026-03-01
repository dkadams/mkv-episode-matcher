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
        interval_seconds: int, subtitle_overlap_seconds: int = 0):
        self.config = config
        self.series = series
        self.interval_seconds = interval_seconds
        self.subtitle_overlap_seconds = subtitle_overlap_seconds
        self.stride_seconds = interval_seconds - subtitle_overlap_seconds
        if self.stride_seconds < 1:
            raise ValueError("subtitle overlap must be less than interval/window duration")

    def execute(self, episode: EpisodeKey, input: Path, output: Path):
        logger.info(f"Intervalizing subs for: {self.series.name},"
                    f" episode: {episode}")

        orig_subs = pysubs2.load(str(input), format_="srt")

        # The last sub in the file is not always a real subtitle. Sometimes it's
        # a tag for the transcriber.
        max_ts_ms = max(sub.end for sub in orig_subs)
        details = self.series.get_episode_detail(episode, keys=["runtime"])
        if details["runtime"]:
            runtime_ms = min(max_ts_ms, int(details["runtime"]) * 60 * 1000)
        else:
            runtime_ms = max_ts_ms

        indexed_intervals = self.get_indexed_intervals(runtime_ms)

        interval_subs = SSAFile()
        for index, start, end in indexed_intervals:
            interval_text = " ".join(sub.plaintext for sub in orig_subs
                                if sub.start < end and sub.end > start)

            # We still want to add an interval even if it's empty. That ensures
            # the indexes will be consistent between files
            event = SSAEvent(start=start, end=end,
                             text=interval_text)
            interval_subs.append(event)

        interval_subs.save(str(output), encoding="utf-8", format_="srt")

    def get_indexed_intervals(self, runtime_ms: int) -> Iterable[tuple[int, int, int]]:
        window_ms = self.interval_seconds * 1000
        stride_ms = self.stride_seconds * 1000
        interval_count = math.ceil(runtime_ms / stride_ms)
        for i in range(interval_count):
            start = i * stride_ms
            yield i, start, start + window_ms
