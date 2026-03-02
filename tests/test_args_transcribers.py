import pytest

from mkv_episode_matcher import transcribers
from mkv_episode_matcher.args import build_args_parser
from mkv_episode_matcher.transcribers import (
    FasterWhisperTranscriber,
    ParakeetMlxTranscriber,
    WhispercppTranscriber,
)


def test_match_defaults_to_parakeet_on_macos(monkeypatch):
    monkeypatch.setattr(transcribers.sys, "platform", "darwin")

    parser = build_args_parser()
    args = parser.parse_args(["match", "/series", "video.mkv"])

    assert args.transcriber is ParakeetMlxTranscriber


def test_match_defaults_to_whispercpp_on_non_macos(monkeypatch):
    monkeypatch.setattr(transcribers.sys, "platform", "linux")

    parser = build_args_parser()
    args = parser.parse_args(["match", "/series", "video.mkv"])

    assert args.transcriber is WhispercppTranscriber


def test_collect_dataset_defaults_to_parakeet_on_macos(monkeypatch):
    monkeypatch.setattr(transcribers.sys, "platform", "darwin")

    parser = build_args_parser()
    args = parser.parse_args(["collect-dataset", "/series", "--output-dir", "/tmp/out"])

    assert args.transcriber is ParakeetMlxTranscriber


def test_collect_dataset_defaults_to_whispercpp_on_non_macos(monkeypatch):
    monkeypatch.setattr(transcribers.sys, "platform", "linux")

    parser = build_args_parser()
    args = parser.parse_args(["collect-dataset", "/series", "--output-dir", "/tmp/out"])

    assert args.transcriber is WhispercppTranscriber


def test_match_rejects_removed_transcriber_flags():
    parser = build_args_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["match", "/series", "video.mkv", "--whispercpp-cli"])

    with pytest.raises(SystemExit):
        parser.parse_args(["match", "/series", "video.mkv", "--whisper"])

    with pytest.raises(SystemExit):
        parser.parse_args(["match", "/series", "video.mkv", "--parakeet-mlx-batch"])


def test_match_accepts_faster_whisper_flag():
    parser = build_args_parser()
    args = parser.parse_args(["match", "/series", "video.mkv", "--faster-whisper"])
    assert args.transcriber is FasterWhisperTranscriber


def test_collect_dataset_accepts_faster_whisper_flag():
    parser = build_args_parser()
    args = parser.parse_args(
        ["collect-dataset", "/series", "--output-dir", "/tmp/out", "--faster-whisper"]
    )
    assert args.transcriber is FasterWhisperTranscriber
