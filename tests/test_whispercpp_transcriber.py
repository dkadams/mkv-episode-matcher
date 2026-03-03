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
        "-t",
        "8",
        "-p",
        "2",
        "-bo",
        "1",
        "-l",
        "en",
        "-otxt",
        "-f",
        str(audio_path),
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
        "-t",
        "2",
        "-otxt",
        "-f",
        str(audio_path),
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
        "-t",
        "2",
        "-otxt",
        "-f",
        str(audio_path),
        "-of",
        seen["cmd"][-1],
    ]


def test_whispercpp_transcribe_many_preserves_order(monkeypatch, tmp_path):
    model_path = tmp_path / "ggml-small.en.bin"
    model_path.write_bytes(b"model")
    audio_paths = [tmp_path / "chunk_a.wav", tmp_path / "chunk_b.wav"]
    for path in audio_paths:
        path.write_bytes(b"audio")

    def fake_run(cmd, capture_output, text, check):
        f_positions = [i for i, arg in enumerate(cmd) if arg == "-f"]
        of_positions = [i for i, arg in enumerate(cmd) if arg == "-of"]
        for f_pos, of_pos in zip(f_positions, of_positions):
            input_path = Path(cmd[f_pos + 1])
            output_base = Path(cmd[of_pos + 1])
            output_base.with_suffix(".txt").write_text(f"text-{input_path.stem}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mkv_episode_matcher.transcribers.subprocess.run", fake_run)

    transcriber = WhispercppTranscriber(str(model_path))
    text = transcriber.transcribe_many(audio_paths)

    assert text == ["text-chunk_a", "text-chunk_b"]


def test_whispercpp_transcribe_many_uses_single_batched_command(monkeypatch, tmp_path):
    model_path = tmp_path / "ggml-small.en.bin"
    model_path.write_bytes(b"model")
    audio_paths = [tmp_path / f"chunk_{i}.wav" for i in range(3)]
    for path in audio_paths:
        path.write_bytes(b"audio")

    seen: dict[str, list[str]] = {}

    def fake_run(cmd, capture_output, text, check):
        seen["cmd"] = cmd
        f_positions = [i for i, arg in enumerate(cmd) if arg == "-f"]
        of_positions = [i for i, arg in enumerate(cmd) if arg == "-of"]
        assert len(f_positions) == len(audio_paths)
        assert len(of_positions) == len(audio_paths)
        for f_pos, of_pos in zip(f_positions, of_positions):
            input_path = Path(cmd[f_pos + 1])
            output_base = Path(cmd[of_pos + 1])
            output_base.with_suffix(".txt").write_text(f"text-{input_path.stem}", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mkv_episode_matcher.transcribers.subprocess.run", fake_run)

    transcriber = WhispercppTranscriber(str(model_path))
    texts = transcriber.transcribe_many(audio_paths)

    assert texts == ["text-chunk_0", "text-chunk_1", "text-chunk_2"]
    f_inputs = [Path(seen["cmd"][i + 1]) for i, arg in enumerate(seen["cmd"]) if arg == "-f"]
    assert f_inputs == audio_paths


def test_whispercpp_transcribe_many_raises_on_missing_output(monkeypatch, tmp_path):
    model_path = tmp_path / "ggml-small.en.bin"
    model_path.write_bytes(b"model")
    audio_paths = [tmp_path / "chunk_a.wav", tmp_path / "chunk_b.wav"]
    for path in audio_paths:
        path.write_bytes(b"audio")

    def fake_run(cmd, capture_output, text, check):
        f_positions = [i for i, arg in enumerate(cmd) if arg == "-f"]
        of_positions = [i for i, arg in enumerate(cmd) if arg == "-of"]
        # Write only the first transcript to simulate partial batch output.
        first_input = Path(cmd[f_positions[0] + 1])
        first_output_base = Path(cmd[of_positions[0] + 1])
        first_output_base.with_suffix(".txt").write_text(
            f"text-{first_input.stem}",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mkv_episode_matcher.transcribers.subprocess.run", fake_run)

    transcriber = WhispercppTranscriber(str(model_path))
    try:
        transcriber.transcribe_many(audio_paths)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "did not produce expected transcript files" in str(exc)


def test_whispercpp_transcribe_many_raises_on_non_zero_exit(monkeypatch, tmp_path):
    model_path = tmp_path / "ggml-small.en.bin"
    model_path.write_bytes(b"model")
    audio_path = tmp_path / "chunk.wav"
    audio_path.write_bytes(b"audio")

    def fake_run(cmd, capture_output, text, check):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr("mkv_episode_matcher.transcribers.subprocess.run", fake_run)

    transcriber = WhispercppTranscriber(str(model_path))
    try:
        transcriber.transcribe_many([audio_path])
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "whispercpp failed for batch" in str(exc)
