from pathlib import Path
from unittest.mock import patch

from mkv_episode_matcher.episode import EpisodeKey


class TestGuessitEpisodeParts:
    @patch("mkv_episode_matcher.episode.guessit", return_value={})
    def test_returns_empty_when_guessit_has_no_match(self, _mock_guessit):
        assert EpisodeKey._guessit_episode_parts(Path("unknown.mkv")) == []

    @patch("mkv_episode_matcher.episode.guessit", return_value={"season": 1})
    def test_returns_empty_when_episode_missing(self, _mock_guessit):
        assert EpisodeKey._guessit_episode_parts(Path("show.mkv")) == []

    @patch("mkv_episode_matcher.episode.guessit",
           return_value={"season": 2, "episode": 7})
    def test_single_episode_returns_single_tuple(self, _mock_guessit):
        assert EpisodeKey._guessit_episode_parts(Path("show.s02e07.mkv")) == [(2, 7)]

    @patch("mkv_episode_matcher.episode.guessit",
           return_value={"season": 3, "episode": [4, 5]})
    def test_multi_episode_returns_multiple_tuples(self, _mock_guessit):
        assert EpisodeKey._guessit_episode_parts(Path("show.s03e04-e05.mkv")) == [
            (3, 4),
            (3, 5),
        ]

    @patch("mkv_episode_matcher.episode.guessit",
           return_value={"season": 4, "episode": [(1, 2), 3]})
    def test_nested_episode_values_are_flattened(self, _mock_guessit):
        assert EpisodeKey._guessit_episode_parts(Path("show.s04e01-e03.mkv")) == [
            (4, 1),
            (4, 2),
            (4, 3),
        ]
