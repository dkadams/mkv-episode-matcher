from pathlib import Path

from mkv_episode_matcher.indexed_episode_matcher import IndexedEpisodeMatcher
from mkv_episode_matcher.transcribers import WhisperTranscriber, WhispercppCliTranscriber


def test_build_transcription_jobs_sorts_by_segment_count_desc():
    jobs = {
        Path("b.mkv"): [1],
        Path("a.mkv"): [1, 2, 3],
        Path("c.mkv"): [1, 2],
    }

    ordered = IndexedEpisodeMatcher._build_transcription_jobs(jobs)
    assert [path for path, _ in ordered] == [Path("a.mkv"), Path("c.mkv"), Path("b.mkv")]


def test_worker_count_for_subprocess_transcriber_uses_thread_env(monkeypatch):
    monkeypatch.setattr("mkv_episode_matcher.indexed_episode_matcher.multiprocessing.cpu_count", lambda: 16)
    monkeypatch.setenv("MEM_THREAD_WORKERS", "6")

    workers = IndexedEpisodeMatcher._transcription_worker_count(
        transcriber_type=WhispercppCliTranscriber,
        job_count=10,
    )
    assert workers == 6


def test_worker_count_for_python_transcriber_uses_process_env(monkeypatch):
    monkeypatch.setattr("mkv_episode_matcher.indexed_episode_matcher.multiprocessing.cpu_count", lambda: 16)
    monkeypatch.setenv("MEM_PROCESS_WORKERS", "3")

    class PythonTranscriber(WhisperTranscriber):
        pass

    workers = IndexedEpisodeMatcher._transcription_worker_count(
        transcriber_type=PythonTranscriber,
        job_count=10,
    )
    assert workers == 3
