import argparse
from configparser import ConfigParser

import mkv_episode_matcher.config as config_module
from mkv_episode_matcher.config import (
    API_CONFIG_KEYS,
    CONFIG_FILE,
    DEFAULT_LOG_DIR,
    Configuration,
    _get_config,
    get_config_file,
    read_config,
    resolve_log_dir,
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


def test_store_api_config_preserves_existing_logging_section(tmp_path):
    config_file = tmp_path / "config.ini"
    config_file.write_text("[logging]\nlog_dir = /tmp/logs\n", encoding="utf-8")

    _write_api_config(config_file)

    parsed = read_config(config_file)
    assert parsed is not None
    assert parsed.get("logging", "log_dir") == "/tmp/logs"
    assert parsed.get("api", "tmdb_api_key") == "tmdb"


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


def test_resolve_log_dir_uses_cli_override(tmp_path):
    config_file = tmp_path / "config.ini"
    config_file.write_text("[logging]\nlog_dir = /tmp/from-config\n", encoding="utf-8")

    args = argparse.Namespace(config_file=config_file, log_dir="/tmp/from-cli")
    assert str(resolve_log_dir(args)) == "/tmp/from-cli"


def test_resolve_log_dir_uses_config_when_cli_missing(tmp_path):
    config_file = tmp_path / "config.ini"
    config_file.write_text("[logging]\nlog_dir = /tmp/from-config\n", encoding="utf-8")

    args = argparse.Namespace(config_file=config_file, log_dir=None)
    assert str(resolve_log_dir(args)) == "/tmp/from-config"


def test_resolve_log_dir_falls_back_to_default(tmp_path):
    args = argparse.Namespace(config_file=tmp_path / "missing.ini", log_dir=None)
    assert resolve_log_dir(args) == DEFAULT_LOG_DIR


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


def test_edit_config_uses_existing_values_when_confirmed(tmp_path, monkeypatch):
    config_file = tmp_path / "config.ini"
    args = argparse.Namespace(config_file=config_file)

    parser = ConfigParser()
    parser["api"] = {
        "tmdb_api_key": "tmdb_existing",
        "open_subtitles_api_key": "os_key_existing",
        "open_subtitles_user_agent": "os_agent_existing",
        "open_subtitles_username": "os_user_existing",
        "open_subtitles_password": "os_pass_existing",
    }
    existing = Configuration(args=args, stored=parser)

    monkeypatch.setattr(config_module, "_get_config", lambda *_: existing)
    monkeypatch.setattr(config_module.Confirm, "ask", lambda *_, **__: True)
    monkeypatch.setattr(
        config_module.Prompt,
        "ask",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("Prompt.ask should not be called")),
    )
    monkeypatch.setattr(config_module.console, "print", lambda *_, **__: None)

    captured = {}

    def fake_store(*store_args):
        captured["args"] = store_args

    monkeypatch.setattr(config_module, "store_api_config", fake_store)

    config_module.edit_config(Configuration(args=args, stored=ConfigParser()))

    assert captured["args"] == (
        "tmdb_existing",
        "os_key_existing",
        "os_agent_existing",
        "os_user_existing",
        "os_pass_existing",
        config_file,
    )


def test_edit_config_prompts_for_missing_values(tmp_path, monkeypatch):
    config_file = tmp_path / "config.ini"
    args = argparse.Namespace(config_file=config_file)
    empty = Configuration(args=args, stored=ConfigParser())

    monkeypatch.setattr(config_module, "_get_config", lambda *_: empty)
    monkeypatch.setattr(config_module.console, "print", lambda *_, **__: None)

    asked_confirm = {"count": 0}

    def fake_confirm(*_, **__):
        asked_confirm["count"] += 1
        return False

    monkeypatch.setattr(config_module.Confirm, "ask", fake_confirm)

    prompted_values = iter(
        [
            "tmdb_prompted",
            "os_user_prompted",
            "os_pass_prompted",
            "os_agent_prompted",
            "os_key_prompted",
        ]
    )
    monkeypatch.setattr(config_module.Prompt, "ask", lambda *_, **__: next(prompted_values))

    captured = {}

    def fake_store(*store_args):
        captured["args"] = store_args

    monkeypatch.setattr(config_module, "store_api_config", fake_store)

    config_module.edit_config(Configuration(args=args, stored=ConfigParser()))

    assert asked_confirm["count"] == 0
    assert captured["args"] == (
        "tmdb_prompted",
        "os_key_prompted",
        "os_agent_prompted",
        "os_user_prompted",
        "os_pass_prompted",
        config_file,
    )
