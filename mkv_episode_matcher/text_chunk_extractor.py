import random
import math

import whisper

from mkv_episode_matcher.audio_chunk_extractor import AudioChunkExtractor

class TextChunkExtractor:
    def __init__(self, chunk_seconds, chunk_count, model_name):
        self.chunk_seconds = chunk_seconds
        self.chunk_count = chunk_count
        self.audio_extractor = AudioChunkExtractor(chunk_seconds)

        self.whisper_model = whisper.load_model(model_name)


    def get_chunks(self, mkv_file):
        duration = self.audio_extractor.get_video_duration(mkv_file)
        chunks_per_file = math.ceil(duration / self.chunk_seconds)
        chunk_count = min(chunks_per_file, self.chunk_count)
        chunk_indexes = random.sample(range(chunks_per_file), chunk_count)

        chunks = []
        for index in chunk_indexes:
            offset = index * self.chunk_seconds
            audio_chunk = self.audio_extractor.extract_audio_chunk(mkv_file, offset)
            text_chunk = whisper.transcribe(self.whisper_model, audio_chunk)
            chunks.append(text_chunk)

        return chunks

    def get_text_chunks(self, mkv_file):
        return [chunk['text'] for chunk in self.get_chunks(mkv_file)]

    def get_text(self, mkv_file):
        return "\n".join(self.get_text_chunks(mkv_file))
