import argparse
from configparser import ConfigParser
import os

import mkv_episode_matcher.config as config_module
from mkv_episode_matcher.config import (
    API_CONFIG_KEYS,
    CONFIG_FILE,
    DEFAULT_LOG_DIR,
    Configuration,
    _get_config,
    backup_config_file,
    get_config_file,
    MAX_CONFIG_BACKUPS,
    prune_config_backups,
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


def test_store_api_config_sets_logging_when_log_dir_provided(tmp_path):
    config_file = tmp_path / "config.ini"
    _write_api_config(config_file)

    store_api_config(
        "tmdb_new",
        "os_key_new",
        "os_agent_new",
        "os_user_new",
        "os_pass_new",
        config_file,
        log_dir="/tmp/persisted-logs",
    )

    parsed = read_config(config_file)
    assert parsed is not None
    assert parsed.get("logging", "log_dir") == "/tmp/persisted-logs"
    assert parsed.get("api", "tmdb_api_key") == "tmdb_new"


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


def test_backup_config_file_creates_timestamped_copy(tmp_path):
    config_file = tmp_path / "config.ini"
    config_file.write_text("[api]\ntmdb_api_key = test\n", encoding="utf-8")

    backup = backup_config_file(config_file)
    assert backup is not None
    assert backup.exists()
    assert backup.name.startswith("config.ini.bak.")
    assert backup.read_text(encoding="utf-8") == config_file.read_text(encoding="utf-8")


def test_backup_config_file_returns_none_when_source_missing(tmp_path):
    missing = tmp_path / "config.ini"
    assert backup_config_file(missing) is None


def test_prune_config_backups_keeps_max_10(tmp_path):
    config_file = tmp_path / "config.ini"
    config_file.write_text("[api]\n", encoding="utf-8")

    backups = []
    for i in range(MAX_CONFIG_BACKUPS + 2):
        backup = tmp_path / f"config.ini.bak.20200101T0000{i:02d}"
        backup.write_text(str(i), encoding="utf-8")
        os.utime(backup, (1000 + i, 1000 + i))
        backups.append(backup)

    prune_config_backups(config_file, MAX_CONFIG_BACKUPS)

    remaining = list(tmp_path.glob("config.ini.bak.*"))
    assert len(remaining) == MAX_CONFIG_BACKUPS
    assert backups[0] not in remaining
    assert backups[1] not in remaining


def test_edit_config_uses_existing_values_when_confirmed(tmp_path, monkeypatch):
    config_file = tmp_path / "config.ini"
    args = argparse.Namespace(config_file=config_file)
    _write_api_config(config_file, suffix="_existing")

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

    monkeypatch.setattr(
        config_module,
        "backup_config_file",
        lambda *_: (_ for _ in ()).throw(AssertionError("backup_config_file should not be called")),
    )
    monkeypatch.setattr(
        config_module,
        "store_api_config",
        lambda *_: (_ for _ in ()).throw(AssertionError("store_api_config should not be called")),
    )

    messages = []
    monkeypatch.setattr(config_module.console, "print", lambda *msg, **__: messages.append(msg))

    config_module.edit_config(Configuration(args=args, stored=ConfigParser()))
    assert any("No configuration changes detected." in part for msg in messages for part in msg)


def test_edit_config_aborts_when_backup_fails(tmp_path, monkeypatch):
    config_file = tmp_path / "config.ini"
    config_file.write_text("[api]\ntmdb_api_key = existing\n", encoding="utf-8")
    args = argparse.Namespace(config_file=config_file)

    parser = ConfigParser()
    parser["api"] = {key: "x" for key in API_CONFIG_KEYS}
    existing = Configuration(args=args, stored=parser)

    monkeypatch.setattr(config_module, "_get_config", lambda *_: existing)
    monkeypatch.setattr(config_module.Confirm, "ask", lambda *_, **__: False)
    monkeypatch.setattr(config_module.Prompt, "ask", lambda *_, **__: "changed")

    messages = []
    monkeypatch.setattr(config_module.console, "print", lambda *msg, **__: messages.append(msg))

    monkeypatch.setattr(
        config_module,
        "backup_config_file",
        lambda *_: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        config_module,
        "store_api_config",
        lambda *_: (_ for _ in ()).throw(AssertionError("store_api_config should not be called")),
    )

    config_module.edit_config(Configuration(args=args, stored=ConfigParser()))
    assert any("Failed to backup configuration" in part for msg in messages for part in msg)


def test_edit_config_skips_backup_and_write_when_unchanged(tmp_path, monkeypatch):
    config_file = tmp_path / "config.ini"
    args = argparse.Namespace(config_file=config_file)
    _write_api_config(config_file, suffix="_existing")

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

    messages = []
    monkeypatch.setattr(config_module.console, "print", lambda *msg, **__: messages.append(msg))
    monkeypatch.setattr(
        config_module,
        "backup_config_file",
        lambda *_: (_ for _ in ()).throw(AssertionError("backup_config_file should not be called")),
    )
    monkeypatch.setattr(
        config_module,
        "store_api_config",
        lambda *_: (_ for _ in ()).throw(AssertionError("store_api_config should not be called")),
    )

    config_module.edit_config(Configuration(args=args, stored=ConfigParser()))
    assert any("No configuration changes detected." in part for msg in messages for part in msg)


def test_edit_config_backs_up_and_writes_when_changed(tmp_path, monkeypatch):
    config_file = tmp_path / "config.ini"
    args = argparse.Namespace(config_file=config_file)
    _write_api_config(config_file, suffix="_existing")

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
    monkeypatch.setattr(config_module.Confirm, "ask", lambda *_, **__: False)
    prompted_values = iter(
        [
            "tmdb_changed",
            "os_user_changed",
            "os_pass_changed",
            "os_agent_changed",
            "os_key_changed",
        ]
    )
    monkeypatch.setattr(config_module.Prompt, "ask", lambda *_, **__: next(prompted_values))
    monkeypatch.setattr(config_module.console, "print", lambda *_, **__: None)

    backup_called = {"count": 0}

    def fake_backup(path):
        backup_called["count"] += 1
        return path.parent / "config.ini.bak.20260101T000000"

    monkeypatch.setattr(config_module, "backup_config_file", fake_backup)

    captured = {}

    def fake_store(*store_args, **store_kwargs):
        captured["args"] = store_args
        captured["kwargs"] = store_kwargs

    monkeypatch.setattr(config_module, "store_api_config", fake_store)

    config_module.edit_config(Configuration(args=args, stored=ConfigParser()))

    assert backup_called["count"] == 1
    assert captured["args"] == (
        "tmdb_changed",
        "os_key_changed",
        "os_agent_changed",
        "os_user_changed",
        "os_pass_changed",
        config_file,
    )
    assert captured["kwargs"] == {}


def test_edit_config_can_persist_log_dir_without_api_changes(tmp_path, monkeypatch):
    config_file = tmp_path / "config.ini"
    args = argparse.Namespace(config_file=config_file, set_log_dir="/tmp/new-logs")
    _write_api_config(config_file, suffix="_existing")

    parser = ConfigParser()
    parser["api"] = {
        "tmdb_api_key": "tmdb_existing",
        "open_subtitles_api_key": "os_key_existing",
        "open_subtitles_user_agent": "os_agent_existing",
        "open_subtitles_username": "os_user_existing",
        "open_subtitles_password": "os_pass_existing",
    }
    parser["logging"] = {"log_dir": "/tmp/old-logs"}
    existing = Configuration(args=args, stored=parser)

    monkeypatch.setattr(config_module, "_get_config", lambda *_: existing)
    monkeypatch.setattr(config_module.Confirm, "ask", lambda *_, **__: True)
    monkeypatch.setattr(
        config_module.Prompt,
        "ask",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("Prompt.ask should not be called")),
    )
    monkeypatch.setattr(config_module.console, "print", lambda *_, **__: None)
    monkeypatch.setattr(
        config_module,
        "backup_config_file",
        lambda path: path.parent / "config.ini.bak.20260101T000000",
    )

    captured = {}

    def fake_store(*store_args, **store_kwargs):
        captured["args"] = store_args
        captured["kwargs"] = store_kwargs

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
    assert captured["kwargs"] == {"log_dir": "/tmp/new-logs"}


def test_edit_config_no_file_with_all_cli_values_writes_non_interactively(tmp_path, monkeypatch):
    config_file = tmp_path / "config.ini"
    args = argparse.Namespace(
        config_file=config_file,
        set_log_dir=None,
        tmdb_api_key="tmdb_cli",
        open_subtitles_api_key="os_key_cli",
        open_subtitles_user_agent="os_agent_cli",
        open_subtitles_username="os_user_cli",
        open_subtitles_password="os_pass_cli",
    )

    monkeypatch.setattr(
        config_module.Confirm,
        "ask",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("Confirm.ask should not be called")),
    )
    monkeypatch.setattr(
        config_module.Prompt,
        "ask",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("Prompt.ask should not be called")),
    )
    monkeypatch.setattr(config_module.console, "print", lambda *_, **__: None)

    config_module.edit_config(Configuration(args=args, stored=ConfigParser()))
    parsed = read_config(config_file)

    assert parsed is not None
    assert parsed.get("api", "tmdb_api_key") == "tmdb_cli"
    assert parsed.get("api", "open_subtitles_password") == "os_pass_cli"


def test_edit_config_no_file_missing_values_invokes_interactive_with_cli_defaults(tmp_path, monkeypatch):
    config_file = tmp_path / "config.ini"
    args = argparse.Namespace(
        config_file=config_file,
        set_log_dir=None,
        tmdb_api_key="tmdb_cli",
        open_subtitles_api_key=None,
        open_subtitles_user_agent=None,
        open_subtitles_username=None,
        open_subtitles_password=None,
    )

    confirm_calls = {"count": 0}

    def fake_confirm(*_, **__):
        confirm_calls["count"] += 1
        return True

    monkeypatch.setattr(config_module.Confirm, "ask", fake_confirm)
    prompted_values = iter(["os_user_prompted", "os_pass_prompted", "os_agent_prompted", "os_key_prompted"])
    monkeypatch.setattr(config_module.Prompt, "ask", lambda *_, **__: next(prompted_values))
    monkeypatch.setattr(config_module.console, "print", lambda *_, **__: None)

    config_module.edit_config(Configuration(args=args, stored=ConfigParser()))
    parsed = read_config(config_file)

    assert confirm_calls["count"] == 1
    assert parsed is not None
    assert parsed.get("api", "tmdb_api_key") == "tmdb_cli"
    assert parsed.get("api", "open_subtitles_username") == "os_user_prompted"


def test_edit_config_existing_file_without_cli_edits_invokes_interactive(tmp_path, monkeypatch):
    config_file = tmp_path / "config.ini"
    _write_api_config(config_file, suffix="_existing")
    args = argparse.Namespace(
        config_file=config_file,
        set_log_dir=None,
        tmdb_api_key=None,
        open_subtitles_api_key=None,
        open_subtitles_user_agent=None,
        open_subtitles_username=None,
        open_subtitles_password=None,
    )

    confirm_calls = {"count": 0}
    monkeypatch.setattr(
        config_module.Confirm,
        "ask",
        lambda *_, **__: confirm_calls.__setitem__("count", confirm_calls["count"] + 1) or True,
    )
    monkeypatch.setattr(
        config_module.Prompt,
        "ask",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("Prompt.ask should not be called")),
    )
    monkeypatch.setattr(config_module.console, "print", lambda *_, **__: None)
    monkeypatch.setattr(
        config_module,
        "backup_config_file",
        lambda *_: (_ for _ in ()).throw(AssertionError("backup_config_file should not be called for no-op")),
    )

    config_module.edit_config(Configuration(args=args, stored=ConfigParser()))
    assert confirm_calls["count"] == len(API_CONFIG_KEYS)


def test_edit_config_existing_file_with_cli_edits_applies_targeted_update(tmp_path, monkeypatch):
    config_file = tmp_path / "config.ini"
    _write_api_config(config_file, suffix="_existing")
    args = argparse.Namespace(
        config_file=config_file,
        set_log_dir=None,
        tmdb_api_key="tmdb_changed",
        open_subtitles_api_key=None,
        open_subtitles_user_agent=None,
        open_subtitles_username=None,
        open_subtitles_password=None,
    )

    monkeypatch.setattr(
        config_module.Confirm,
        "ask",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("Confirm.ask should not be called")),
    )
    monkeypatch.setattr(
        config_module.Prompt,
        "ask",
        lambda *_, **__: (_ for _ in ()).throw(AssertionError("Prompt.ask should not be called")),
    )
    monkeypatch.setattr(config_module.console, "print", lambda *_, **__: None)

    config_module.edit_config(Configuration(args=args, stored=ConfigParser()))
    parsed = read_config(config_file)

    assert parsed is not None
    assert parsed.get("api", "tmdb_api_key") == "tmdb_changed"
    assert parsed.get("api", "open_subtitles_api_key") == "os_key_existing"


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
