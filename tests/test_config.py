from __future__ import annotations

import logging

from paperless_gemini_ocr.config import Config


def test_key_from_file(tmp_path, monkeypatch):
    key_file = tmp_path / "key"
    key_file.write_text("from-file\n")
    monkeypatch.setenv("PAPERLESS_GEMINI_OCR_API_KEY_FILE", str(key_file))
    assert Config.from_env().api_key == "from-file"


def test_unreadable_key_file_is_said_out_loud(tmp_path, monkeypatch, caplog):
    """Without a key the plugin stands aside, which looks like success."""
    monkeypatch.setenv("PAPERLESS_GEMINI_OCR_API_KEY_FILE", str(tmp_path / "missing"))
    with caplog.at_level(logging.WARNING, logger="paperless.parsing.gemini_ocr"):
        assert Config.from_env().api_key is None
    assert "inactive" in caplog.text


def test_key_file_with_broken_encoding_or_no_content(tmp_path, monkeypatch, caplog):
    key_file = tmp_path / "key"
    monkeypatch.setenv("PAPERLESS_GEMINI_OCR_API_KEY_FILE", str(key_file))
    with caplog.at_level(logging.WARNING, logger="paperless.parsing.gemini_ocr"):
        key_file.write_bytes(b"\xff\xfe\xfa")
        assert Config.from_env().api_key is None
        key_file.write_text("  \n")
        assert Config.from_env().api_key is None
    assert caplog.text.count("inactive") == 2


def test_key_is_not_part_of_the_printed_config():
    assert "secret-key" not in repr(Config(api_key="secret-key"))
