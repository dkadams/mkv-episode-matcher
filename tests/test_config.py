import argparse
from configparser import ConfigParser

from mkv_episode_matcher.config import (
    API_CONFIG_KEYS,
    CONFIG_FILE,
    Configuration,
    _get_config,
    get_config_file,
    read_config,
    store_api_config,
)


def _write_api_config(path, suffix=""):
    store_api_config(
        f"tmdb{suffix}",
        f"os_key{suffix}",
        f"os_agent{suffix}",
        f"os_user{suffix}",
        f"os_pass{suffix}",
        path,
    )


def test_read_config_returns_none_for_missing_file(tmp_path):
    missing = tmp_path / "missing.ini"
    assert read_config(missing) is None


def test_store_api_config_round_trip(tmp_path):
    config_file = tmp_path / "config.ini"
    _write_api_config(config_file)

    parsed = read_config(config_file)
    assert parsed is not None
    assert parsed.get("api", "tmdb_api_key") == "tmdb"
    assert parsed.get("api", "open_subtitles_password") == "os_pass"


def test_get_config_file_uses_arg_when_present(tmp_path):
    custom = tmp_path / "custom.ini"
    args = argparse.Namespace(config_file=custom)
    assert get_config_file(args) == custom


def test_get_config_file_falls_back_to_default():
    args = argparse.Namespace(config_file=None)
    assert get_config_file(args) == CONFIG_FILE


def test_get_config_merges_cli_overrides_using_underscore_names(tmp_path):
    config_file = tmp_path / "config.ini"
    _write_api_config(config_file, suffix="_stored")

    args = argparse.Namespace(
        tmdb_api_key="tmdb_cli",
        open_subtitles_api_key="os_key_cli",
        open_subtitles_user_agent="os_agent_cli",
        open_subtitles_username="os_user_cli",
        open_subtitles_password="os_pass_cli",
    )
    config = _get_config(config_file, args)

    for key in API_CONFIG_KEYS:
        assert config.stored.get("api", key).endswith("_cli")


def test_get_config_keeps_stored_values_when_cli_values_are_none(tmp_path):
    config_file = tmp_path / "config.ini"
    _write_api_config(config_file, suffix="_stored")

    args = argparse.Namespace(
        tmdb_api_key=None,
        open_subtitles_api_key=None,
        open_subtitles_user_agent=None,
        open_subtitles_username=None,
        open_subtitles_password=None,
    )
    config = _get_config(config_file, args)

    for key in API_CONFIG_KEYS:
        assert config.stored.get("api", key).endswith("_stored")


def test_get_config_ignores_unrelated_args(tmp_path):
    config_file = tmp_path / "config.ini"
    _write_api_config(config_file)

    args = argparse.Namespace(series_dirs=["/tmp/show"], verbose=True)
    config = _get_config(config_file, args)
    assert config.stored.get("api", "tmdb_api_key") == "tmdb"


def test_configuration_has_required_settings_true():
    parser = ConfigParser()
    parser["api"] = {key: "x" for key in API_CONFIG_KEYS}
    cfg = Configuration(args=argparse.Namespace(), stored=parser)
    assert cfg.has_required_settings() is True


def test_configuration_has_required_settings_false_when_missing():
    parser = ConfigParser()
    parser["api"] = {key: "x" for key in API_CONFIG_KEYS}
    parser["api"]["open_subtitles_password"] = ""
    cfg = Configuration(args=argparse.Namespace(), stored=parser)
    assert cfg.has_required_settings() is False
