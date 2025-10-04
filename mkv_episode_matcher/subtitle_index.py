
import chromadb
import pysubs2

from mkv_episode_matcher.__main__ import CONFIG_FILE
from mkv_episode_matcher.config import get_config
from mkv_episode_matcher.utils import get_series_cache_path

class SubtitleIndex:
    def __init__(self, show_name):
        self.config = get_config(CONFIG_FILE)
        self.show_name = show_name

        self.show_path = get_series_cache_path(self.show_name)
        show_db_file = self.show_path.with_suffix('.chromadb')

        self.chromadb = chromadb.PersistentClient(path=show_db_file)

    def upsert(self, path):
        full_episodes = self.chromadb.get_or_create_collection(name="full-episodes")

        sub_file = pysubs2.load(path)
        full_episode = "\n".join([line.plaintext for line in sub_file])
        print(full_episode)

        full_episodes.upsert(ids=[str(path)], documents=[full_episode])

    def index_show(self):
        files = [f for f in self.show_path.iterdir() if f.is_file()]
        for file in files:
            self.upsert(file)

