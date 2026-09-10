"""Root resolution and the on-disk layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from vex import config, settings


def test_explicit_root_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.LEGACY_ENV_ROOT, raising=False)
    monkeypatch.setenv(config.ENV_ROOT, str(tmp_path / "from-env"))
    assert config.resolve_root(tmp_path / "explicit") == (tmp_path / "explicit").resolve()


def test_environment_variable_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.LEGACY_ENV_ROOT, raising=False)
    monkeypatch.setenv(config.ENV_ROOT, str(tmp_path / "from-env"))
    assert config.resolve_root() == (tmp_path / "from-env").resolve()


def test_default_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.ENV_ROOT, raising=False)
    monkeypatch.delenv(config.LEGACY_ENV_ROOT, raising=False)
    assert config.resolve_root() == (Path.home() / "Screenshots").resolve()


def test_the_version_1_root_variable_is_still_honoured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(config.ENV_ROOT, raising=False)
    monkeypatch.setenv(config.LEGACY_ENV_ROOT, str(tmp_path / "old-name"))
    assert config.resolve_root() == (tmp_path / "old-name").resolve()


def test_the_new_root_variable_wins_over_the_old_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config.ENV_ROOT, str(tmp_path / "new-name"))
    monkeypatch.setenv(config.LEGACY_ENV_ROOT, str(tmp_path / "old-name"))
    assert config.resolve_root() == (tmp_path / "new-name").resolve()


def test_ensure_layout_creates_every_directory(tmp_path: Path) -> None:
    root = config.ensure_layout(tmp_path / "root")
    for relative in config.LAYOUT_DIRS:
        assert (root / relative).is_dir()


def test_ensure_layout_is_idempotent(tmp_path: Path) -> None:
    config.ensure_layout(tmp_path / "root")
    config.ensure_layout(tmp_path / "root")
    assert (tmp_path / "root" / "inbox" / "mac").is_dir()


def test_relative_paths_use_forward_slashes(tmp_path: Path) -> None:
    root = config.ensure_layout(tmp_path / "root")
    target = root / "library" / "2026" / "07" / "a.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()

    relative = config.relative_path(root, target)
    assert relative == "library/2026/07/a.png"
    assert config.absolute_path(root, relative) == target


# ---------------------------------------------------------------------------
# The macOS drop zone, which lives outside the root
# ---------------------------------------------------------------------------


def test_mac_inbox_defaults_to_the_home_directory(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside the root on purpose, so it survives the drive being unplugged."""
    monkeypatch.delenv(settings.ENV_MAC_INBOX, raising=False)
    assert settings.mac_inbox(root) == Path.home() / "Screenshots" / "inbox" / "mac"


def test_mac_inbox_from_the_environment(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(settings.ENV_MAC_INBOX, "~/elsewhere/shots")
    assert settings.mac_inbox(root) == Path.home() / "elsewhere" / "shots"


def test_the_version_1_drop_zone_variable_is_still_honoured(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(settings.ENV_MAC_INBOX, raising=False)
    monkeypatch.setenv(settings.LEGACY_ENV_MAC_INBOX, "/from/the/old/name")
    assert settings.mac_inbox(root) == Path("/from/the/old/name")


def test_the_version_1_config_filename_is_still_read(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(settings.ENV_MAC_INBOX, raising=False)
    (root / settings.LEGACY_CONFIG_FILENAME).write_text(
        '[paths]\nmac_inbox = "/from/the/old/file"\n', encoding="utf-8"
    )
    assert settings.mac_inbox(root) == Path("/from/the/old/file")


def test_the_new_config_filename_wins(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(settings.ENV_MAC_INBOX, raising=False)
    (root / settings.LEGACY_CONFIG_FILENAME).write_text(
        '[paths]\nmac_inbox = "/from/the/old/file"\n', encoding="utf-8"
    )
    (root / settings.CONFIG_FILENAME).write_text(
        '[paths]\nmac_inbox = "/from/the/new/file"\n', encoding="utf-8"
    )
    assert settings.mac_inbox(root) == Path("/from/the/new/file")


def test_mac_inbox_from_the_config_file(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(settings.ENV_MAC_INBOX, raising=False)
    (root / settings.CONFIG_FILENAME).write_text(
        '[paths]\nmac_inbox = "~/Pictures/drops"\n', encoding="utf-8"
    )
    assert settings.mac_inbox(root) == Path.home() / "Pictures" / "drops"


def test_the_environment_beats_the_config_file(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (root / settings.CONFIG_FILENAME).write_text(
        '[paths]\nmac_inbox = "/from/file"\n', encoding="utf-8"
    )
    monkeypatch.setenv(settings.ENV_MAC_INBOX, "/from/env")
    assert settings.mac_inbox(root) == Path("/from/env")


def test_a_malformed_config_file_falls_back_to_the_default(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(settings.ENV_MAC_INBOX, raising=False)
    (root / settings.CONFIG_FILENAME).write_text("this is not toml [[[", encoding="utf-8")
    assert settings.mac_inbox(root) == settings.DEFAULT_MAC_INBOX
