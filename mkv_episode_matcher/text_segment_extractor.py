import random
import math
import time

import torch
import whisper
from faster_whisper import WhisperModel
from loguru import logger

from mkv_episode_matcher.audio_chunk_extractor import AudioChunkExtractor

class WhisperTranscriber:
    def __init__(self, model_name):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path):
        fp16 = self.model.device != torch.device("cpu")
        result = whisper.transcribe(self.model, str(audio_path), fp16=fp16)
        return result["text"] if result else None

class FasterWhisperTranscriber:
    def __init__(self, model_name):
        self.model = WhisperModel(model_name)

    def transcribe(self, audio_path):
        logger.info(f"Transcribing {audio_path}")
        segments, info = self.model.transcribe(str(audio_path))
        text_segments = [segment.text for segment in segments]
        return " ".join(text_segments)

class TextSegmentExtractor:
    def __init__(self, model_name, transcriber):
        self.transcriber = transcriber(model_name)

    def get_random_segments(self, path, duration, count):
        total_duration = AudioChunkExtractor.get_video_duration(path)
        chunks_per_file = math.ceil(total_duration / duration)
        count = min(chunks_per_file, count)

        # TODO bias this towards the middle of the file
        chunk_indexes = random.sample(range(chunks_per_file), count)

        results = []
        total_extract_time = 0
        total_transcribe_time = 0
        with AudioChunkExtractor() as audio_extractor:
            for index in chunk_indexes:
                offset = index * duration

                before = time.time()
                chunk_path = audio_extractor.extract(path, offset, duration)
                total_extract_time += time.time() - before

                before = time.time()
                text = self.transcriber.transcribe(chunk_path)
                total_transcribe_time += time.time() - before

                if text:
                    results.append((index, text))
                else:
                    logger.warning(f"Failed to transcribe {chunk_path}")

        logger.info(f"Extracted {count} audio chunks "
                    f"in {total_extract_time:.2f}s, transcribed "
                    f"in {total_transcribe_time:.2f}s")
        return results

    def get_text_segments(self, path, duration=30, count=10):
        segments = self.get_random_segments(path, duration, count)
        return [(index, text) for index, text in segments]
