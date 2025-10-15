from mkv_episode_matcher.subtitle_index import FullEpisodeSubtitleIndex
from mkv_episode_matcher.text_chunk_extractor import TextChunkExtractor


class IndexedEpisodeMatcher:
    def __init__(self, show_name):
        self.index = FullEpisodeSubtitleIndex(show_name)
        self.text_extractor = TextChunkExtractor(30, 10, "small.en")

    def match_episodes(self, mkv_files_root):
        files = mkv_files_root.glob('**/*.py')

        for file in files.mkv_files_root:
            self.match_file(file)

    def match_file(self, file):
        text = self.text_extractor.get_text(file)
        candidates = self.index.full_episodes.query(query_texts=[text])
        return candidates
