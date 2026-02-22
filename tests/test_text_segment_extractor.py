from mkv_episode_matcher.segment_transcriber import SegmentTranscriber


def test_normalize_transcript_dict_segments():
    payload = {
        "segments": [
            {"text": " Hello "},
            {"text": "from"},
            {"text": "segment transcriber"},
        ]
    }
    assert SegmentTranscriber._normalize_transcript(payload) == (
        "Hello from segment transcriber"
    )


def test_normalize_transcript_strips_blank_audio_marker():
    transcript = "alpha [BLANK_AUDIO] beta"
    assert SegmentTranscriber._normalize_transcript(transcript) == "alpha   beta"


def test_extract_text_prefers_segment_entries():
    payload = {
        "files": [
            {
                "path": "audio.wav",
                "text": "fallback text",
                "segments": [
                    {"text": "Hello"},
                    {"text": "from"},
                    {"text": "WhisperKit"},
                ],
            }
        ]
    }

    text = SegmentTranscriber._extract_text(payload)
    assert text == "Hello from WhisperKit"
