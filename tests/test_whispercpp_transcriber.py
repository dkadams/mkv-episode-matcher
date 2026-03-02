from pathlib import Path
from types import SimpleNamespace

from mkv_episode_matcher.transcribers import WhispercppTranscriber


def test_whispercpp_transcribe_applies_runtime_env_args(monkeypatch, tmp_path):
    model_path = tmp_path / "ggml-small.en.bin"
    model_path.write_bytes(b"model")
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"audio")

    monkeypatch.setenv("WHISPERCPP_THREADS", "8")
    monkeypatch.setenv("WHISPERCPP_PROCESSORS", "2")
    monkeypatch.setenv("WHISPERCPP_EXTRA_ARGS", "-bo 1 -l en")

    seen: dict[str, list[str]] = {}

    def fake_run(cmd, capture_output, text, check):
        seen["cmd"] = cmd
        output_base = Path(cmd[cmd.index("-of") + 1])
        output_base.with_suffix(".txt").write_text("hello world", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mkv_episode_matcher.transcribers.subprocess.run", fake_run)

    transcriber = WhispercppTranscriber(str(model_path))
    text = transcriber.transcribe(audio_path)

    assert text == "hello world"
    assert seen["cmd"] == [
        "whisper-cli",
        "-m",
        str(model_path),
        "-f",
        str(audio_path),
        "-t",
        "8",
        "-p",
        "2",
        "-bo",
        "1",
        "-l",
        "en",
        "-otxt",
        "-of",
        seen["cmd"][-1],
    ]


def test_whispercpp_ignores_invalid_runtime_env_args(monkeypatch, tmp_path):
    model_path = tmp_path / "ggml-small.en.bin"
    model_path.write_bytes(b"model")
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"audio")

    monkeypatch.setenv("WHISPERCPP_THREADS", "nope")
    monkeypatch.setenv("WHISPERCPP_PROCESSORS", "0")
    monkeypatch.setenv("WHISPERCPP_EXTRA_ARGS", "\"")

    seen: dict[str, list[str]] = {}

    def fake_run(cmd, capture_output, text, check):
        seen["cmd"] = cmd
        output_base = Path(cmd[cmd.index("-of") + 1])
        output_base.with_suffix(".txt").write_text("ok", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mkv_episode_matcher.transcribers.subprocess.run", fake_run)

    transcriber = WhispercppTranscriber(str(model_path))
    text = transcriber.transcribe(audio_path)

    assert text == "ok"
    assert seen["cmd"] == [
        "whisper-cli",
        "-m",
        str(model_path),
        "-f",
        str(audio_path),
        "-t",
        "2",
        "-otxt",
        "-of",
        seen["cmd"][-1],
    ]


def test_whispercpp_uses_default_threads_when_env_unset(monkeypatch, tmp_path):
    model_path = tmp_path / "ggml-small.en.bin"
    model_path.write_bytes(b"model")
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"audio")

    monkeypatch.delenv("WHISPERCPP_THREADS", raising=False)
    monkeypatch.delenv("WHISPERCPP_PROCESSORS", raising=False)
    monkeypatch.delenv("WHISPERCPP_EXTRA_ARGS", raising=False)

    seen: dict[str, list[str]] = {}

    def fake_run(cmd, capture_output, text, check):
        seen["cmd"] = cmd
        output_base = Path(cmd[cmd.index("-of") + 1])
        output_base.with_suffix(".txt").write_text("ok", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mkv_episode_matcher.transcribers.subprocess.run", fake_run)

    transcriber = WhispercppTranscriber(str(model_path))
    text = transcriber.transcribe(audio_path)

    assert text == "ok"
    assert seen["cmd"] == [
        "whisper-cli",
        "-m",
        str(model_path),
        "-f",
        str(audio_path),
        "-t",
        "2",
        "-otxt",
        "-of",
        seen["cmd"][-1],
    ]


def test_whispercpp_transcribe_many_preserves_order(monkeypatch, tmp_path):
    model_path = tmp_path / "ggml-small.en.bin"
    model_path.write_bytes(b"model")
    audio_paths = [tmp_path / "chunk_a.wav", tmp_path / "chunk_b.wav"]
    for path in audio_paths:
        path.write_bytes(b"audio")

    seen_cmds: list[list[str]] = []

    def fake_run(cmd, capture_output, text, check):
        seen_cmds.append(cmd)
        input_path = Path(cmd[cmd.index("-f") + 1])
        output_base = Path(cmd[cmd.index("-of") + 1])
        output_base.with_suffix(".txt").write_text(f"text-{input_path.stem}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mkv_episode_matcher.transcribers.subprocess.run", fake_run)

    transcriber = WhispercppTranscriber(str(model_path))
    text = transcriber.transcribe_many(audio_paths)

    assert text == ["text-chunk_a", "text-chunk_b"]
    assert [Path(cmd[cmd.index("-f") + 1]) for cmd in seen_cmds] == audio_paths
