
import chromadb
import pysubs2

from mkv_episode_matcher.utils import get_series_cache_path

class SubtitleIndex:
    def __init__(self, config, show_name):
        self.config = config
        self.show_name = show_name

        self.show_path = get_series_cache_path(self.show_name)
        show_db_file = self.show_path.with_suffix('.chromadb')

        self.chromadb = chromadb.PersistentClient(path=show_db_file)
        self.full_episodes = self.chromadb.get_or_create_collection(name="full-episodes",
                                                                    metadata={"hnsw:space": "cosine"})

    def upsert(self, path):
        sub_file = pysubs2.load(path)
        full_episode = "\n".join([line.plaintext for line in sub_file])

        self.full_episodes.upsert(ids=[str(path)], documents=[full_episode])

    def index_show(self):
        files = [f for f in self.show_path.iterdir() if f.is_file()]
        for file in files:
            self.upsert(file)


