import asyncio

import pytest
from fastapi import HTTPException

import helpers.email_helper as email_helper


def run(coro):
    return asyncio.run(coro)


def test_send_registration_email_sends_message(monkeypatch):
    sent = []

    class FakeFastMail:
        def __init__(self, conf):
            self.conf = conf

        async def send_message(self, message):
            sent.append(message)

    monkeypatch.setattr(email_helper, "FastMail", FakeFastMail)

    message = run(email_helper.send_registration_email("user@example.com", "http://verify", object()))

    assert message.subject.startswith("Welcome to Violens")
    assert message.recipients[0].email == "user@example.com"
    assert sent[0] is message


def test_send_reset_password_email_error(monkeypatch):
    class FakeFastMail:
        def __init__(self, conf):
            self.conf = conf

        async def send_message(self, _message):
            raise Exception("smtp down")

    monkeypatch.setattr(email_helper, "FastMail", FakeFastMail)

    with pytest.raises(HTTPException) as exc:
        run(email_helper.send_reset_password_email("user@example.com", "http://reset", object()))

    assert exc.value.status_code == 500


def test_send_dataset_approval_email(monkeypatch):
    sent = []

    class FakeFastMail:
        def __init__(self, conf):
            self.conf = conf

        async def send_message(self, message):
            sent.append(message)

    monkeypatch.setattr(email_helper, "FastMail", FakeFastMail)

    run(email_helper.send_dataset_approval_mail("user@example.com", "dataset", "ok", object()))

    assert len(sent) == 1


def test_send_dataset_rejection_email(monkeypatch):
    sent = []

    class FakeFastMail:
        def __init__(self, conf):
            self.conf = conf

        async def send_message(self, message):
            sent.append(message)

    monkeypatch.setattr(email_helper, "FastMail", FakeFastMail)

    run(email_helper.send_dataset_rejection_mail("user@example.com", "dataset", "no", object()))

    assert len(sent) == 1


def test_send_user_ban_email(monkeypatch):
    sent = []

    class FakeFastMail:
        def __init__(self, conf):
            self.conf = conf

        async def send_message(self, message):
            sent.append(message)

    monkeypatch.setattr(email_helper, "FastMail", FakeFastMail)

    run(email_helper.send_user_ban_email("user@example.com", "reason", object()))

    assert len(sent) == 1
