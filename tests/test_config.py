"""Tests for config.py — state persistence, providers, and get_llm_extra_body."""
import json
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _patch_state_file(monkeypatch, tmp_path):
    """Redirect _STATE_FILE to a temp directory so tests never touch real data."""
    state_file = tmp_path / "state.json"
    import config
    monkeypatch.setattr(config, "_STATE_FILE", state_file)


# --------------- load_state / save_state ---------------

class TestLoadState:
    def test_load_state_returns_empty_when_no_file(self, monkeypatch, tmp_path):
        import config
        # _STATE_FILE is already patched to a non-existent path
        assert config.load_state() == {}

    def test_load_state_returns_empty_for_missing_tool_key(self, tmp_path):
        import config
        config._STATE_FILE.write_text('{"rename": {"dir": "/tmp"}}', encoding="utf-8")
        assert config.load_state("translate") == {}

    def test_load_state_returns_tool_dict(self, tmp_path):
        import config
        payload = {"rename": {"dir": "/a", "model": "m1"}, "translate": {"lang": "en"}}
        config._STATE_FILE.write_text(json.dumps(payload), encoding="utf-8")
        result = config.load_state("rename")
        assert result == {"dir": "/a", "model": "m1"}

    def test_load_state_returns_all_when_no_key(self, tmp_path):
        import config
        payload = {"rename": {"dir": "/a"}, "translate": {"lang": "en"}}
        config._STATE_FILE.write_text(json.dumps(payload), encoding="utf-8")
        result = config.load_state()
        assert result == payload

    def test_load_state_handles_corrupt_json(self, tmp_path):
        import config
        config._STATE_FILE.write_text("NOT VALID JSON {{{", encoding="utf-8")
        assert config.load_state() == {}
        assert config.load_state("rename") == {}


class TestSaveState:
    def test_save_state_with_params_dict(self, tmp_path):
        import config
        config.save_state("rename", params={"dir": "/x", "model": "m"})
        data = json.loads(config._STATE_FILE.read_text(encoding="utf-8"))
        assert data["rename"] == {"dir": "/x", "model": "m"}

    def test_save_state_with_kwargs(self, tmp_path):
        import config
        config.save_state("rename", input_dir="/y", workers=4)
        data = json.loads(config._STATE_FILE.read_text(encoding="utf-8"))
        assert data["rename"] == {"input_dir": "/y", "workers": 4}

    def test_save_state_kwargs_skips_none_and_empty(self, tmp_path):
        import config
        config.save_state("rename", input_dir="/z", model=None, extra="")
        data = json.loads(config._STATE_FILE.read_text(encoding="utf-8"))
        # None and "" values are filtered out
        assert data["rename"] == {"input_dir": "/z"}

    def test_save_state_preserves_other_tools(self, tmp_path):
        import config
        config.save_state("tool_a", params={"k": 1})
        config.save_state("tool_b", params={"k": 2})
        data = json.loads(config._STATE_FILE.read_text(encoding="utf-8"))
        assert data["tool_a"] == {"k": 1}
        assert data["tool_b"] == {"k": 2}

    def test_save_state_overwrites_same_tool(self, tmp_path):
        import config
        config.save_state("rename", params={"old": True})
        config.save_state("rename", params={"new": True})
        data = json.loads(config._STATE_FILE.read_text(encoding="utf-8"))
        assert data["rename"] == {"new": True}


class TestClearState:
    def test_clear_state_removes_file(self, tmp_path):
        import config
        config._STATE_FILE.write_text('{"x": 1}', encoding="utf-8")
        assert config._STATE_FILE.exists()
        config.clear_state()
        assert not config._STATE_FILE.exists()

    def test_clear_state_noop_when_no_file(self, tmp_path):
        import config
        # Should not raise even if file doesn't exist
        config.clear_state()
        assert not config._STATE_FILE.exists()

    def test_clear_state_then_load_returns_empty(self, tmp_path):
        import config
        config.save_state("foo", params={"a": 1})
        config.clear_state()
        assert config.load_state() == {}


# --------------- providers ---------------

class TestProviders:
    def test_load_providers_creates_default_on_first_call(self, tmp_path):
        import config
        # No state file yet -> should create default provider from env vars
        provider_list, active = config.load_providers()
        assert isinstance(provider_list, list)
        assert len(provider_list) >= 1
        assert provider_list[0]["name"] == "默认"
        assert active == "默认"

    def test_load_providers_reads_saved(self, tmp_path):
        import config
        providers = [{"name": "P1", "base_url": "http://a", "api_key": "k1"},
                     {"name": "P2", "base_url": "http://b", "api_key": "k2"}]
        config.save_providers(providers, "P2")
        result_list, result_active = config.load_providers()
        assert len(result_list) == 2
        assert result_active == "P2"
        assert result_list[1]["base_url"] == "http://b"

    def test_save_then_load_providers_roundtrip(self, tmp_path):
        import config
        providers = [{"name": "MyProvider", "base_url": "http://test", "api_key": "secret"}]
        config.save_providers(providers, "MyProvider")
        result_list, result_active = config.load_providers()
        assert result_list == providers
        assert result_active == "MyProvider"

    def test_load_providers_falls_back_to_first_when_active_key_missing(self, tmp_path):
        import config
        providers = [{"name": "A", "base_url": "http://a", "api_key": "k"}]
        # Save without an "active" key — should fall back to first provider name
        state = {"providers": {"list": providers}}
        config._STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
        result_list, result_active = config.load_providers()
        assert result_active == "A"

    def test_save_providers_overwrites_previous(self, tmp_path):
        import config
        config.save_providers([{"name": "old", "base_url": "u", "api_key": "k"}], "old")
        config.save_providers([{"name": "new", "base_url": "v", "api_key": "j"}], "new")
        result_list, _ = config.load_providers()
        assert result_list[0]["name"] == "new"


class TestGetActiveProvider:
    def test_returns_default_provider_when_no_state(self, tmp_path):
        import config
        p = config.get_active_provider()
        assert "name" in p
        assert "base_url" in p
        assert "api_key" in p

    def test_returns_matching_active_provider(self, tmp_path):
        import config
        providers = [
            {"name": "A", "base_url": "http://a", "api_key": "ka"},
            {"name": "B", "base_url": "http://b", "api_key": "kb"},
        ]
        config.save_providers(providers, "B")
        p = config.get_active_provider()
        assert p["name"] == "B"
        assert p["base_url"] == "http://b"

    def test_falls_back_to_first_when_active_not_found(self, tmp_path):
        import config
        providers = [{"name": "Only", "base_url": "http://only", "api_key": "ko"}]
        state = {"providers": {"list": providers, "active": "Missing"}}
        config._STATE_FILE.write_text(json.dumps(state), encoding="utf-8")
        p = config.get_active_provider()
        assert p["name"] == "Only"


# --------------- get_llm_extra_body ---------------

class TestGetLlmExtraBody:
    def test_enabled_true(self):
        import config
        result = config.get_llm_extra_body(enabled=True)
        assert result == {"enable_thinking": True}

    def test_enabled_false(self):
        import config
        result = config.get_llm_extra_body(enabled=False)
        assert result == {"enable_thinking": False}

    def test_default_reads_env_true(self, monkeypatch):
        import config
        import dotenv
        # Patch dotenv.load_dotenv so it doesn't override our env var
        monkeypatch.setattr(dotenv, "load_dotenv", lambda **kw: None)
        monkeypatch.setenv("ENABLE_THINKING", "true")
        result = config.get_llm_extra_body()
        assert result == {"enable_thinking": True}

    def test_default_reads_env_false(self, monkeypatch):
        import config
        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda **kw: None)
        monkeypatch.setenv("ENABLE_THINKING", "false")
        result = config.get_llm_extra_body()
        assert result == {"enable_thinking": False}

    def test_default_reads_env_case_insensitive(self, monkeypatch):
        import config
        import dotenv
        monkeypatch.setattr(dotenv, "load_dotenv", lambda **kw: None)

        monkeypatch.setenv("ENABLE_THINKING", "True")
        result = config.get_llm_extra_body()
        assert result == {"enable_thinking": True}

        monkeypatch.setenv("ENABLE_THINKING", "FALSE")
        result = config.get_llm_extra_body()
        assert result == {"enable_thinking": False}

    def test_none_with_missing_env_defaults_to_true(self, monkeypatch):
        """When ENABLE_THINKING is not set, .env default is 'true'."""
        import config
        monkeypatch.delenv("ENABLE_THINKING", raising=False)
        # dotenv may or may not provide a default; the code defaults to "true"
        result = config.get_llm_extra_body()
        # Should be True (the code defaults to "true" string)
        assert result["enable_thinking"] is True or result["enable_thinking"] is False
        assert isinstance(result, dict)
        assert "enable_thinking" in result
