import random
import math
import time

import torch
import whisper
from loguru import logger

from mkv_episode_matcher.audio_chunk_extractor import AudioChunkExtractor

class TextSegmentExtractor:
    def __init__(self, model_name):
        self.whisper_model = whisper.load_model(model_name)

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
                fp16 = self.whisper_model.device != torch.device("cpu")
                result = whisper.transcribe(self.whisper_model, str(chunk_path),
                                            fp16=fp16)
                total_transcribe_time += time.time() - before

                results.append((index, result))

        logger.info(f"Extracted {count} audio chunks "
                    f"in {total_extract_time:.2f}s, transcribed "
                    f"in {total_transcribe_time:.2f}s")
        return results

    def get_text_segments(self, path, duration=30, count=10):
        segments = self.get_random_segments(path, duration, count)
        return [(index, segment['text']) for index, segment in segments]
