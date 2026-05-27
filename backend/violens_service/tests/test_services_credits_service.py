import asyncio

import pytest
from fastapi import HTTPException

from services.credits_service import CreditsService
import services.credits_service as credits_service


def run(coro):
    return asyncio.run(coro)


def test_get_credits_cronjob_update_success(monkeypatch, tmp_path):
    monkeypatch.setattr(credits_service, "load_dotenv", lambda **_kwargs: None)
    monkeypatch.setenv("DEFAULT_CREDITS", "7")
    service = CreditsService()

    result = run(service.get_credits_cronjob_update())

    assert result == 7


def test_get_credits_cronjob_update_invalid_env(monkeypatch, tmp_path):
    monkeypatch.setattr(credits_service, "load_dotenv", lambda **_kwargs: None)
    monkeypatch.setenv("DEFAULT_CREDITS", "not-a-number")
    service = CreditsService()

    with pytest.raises(ValueError):
        run(service.get_credits_cronjob_update())


def test_patch_credits_cronjob_update_updates_file(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("DEFAULT_CREDITS=5\nOTHER=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    service = CreditsService()
    run(service.patch_credits_cronjob_update(12))

    updated = env_path.read_text(encoding="utf-8")
    assert "DEFAULT_CREDITS=12" in updated
    assert "OTHER=1" in updated


def test_patch_credits_cronjob_update_raises_on_error(monkeypatch):
    service = CreditsService()

    def _raise(*_args, **_kwargs):
        raise OSError("blocked")

    monkeypatch.setattr("builtins.open", _raise)

    with pytest.raises(HTTPException) as exc:
        run(service.patch_credits_cronjob_update(3))

    assert exc.value.status_code == 500
