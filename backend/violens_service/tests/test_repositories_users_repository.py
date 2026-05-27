import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.exc import SQLAlchemyError

import repositories.users_repository as users_repository
from repositories.users_repository import UsersRepository
from schemas.users_schema import CreateUserDto


def run(coro):
    return asyncio.run(coro)


class DummyScalars:
    def __init__(self, items):
        self._items = items

    def all(self):
        return list(self._items)

    def first(self):
        return self._items[0] if self._items else None


class DummyResult:
    def __init__(self, items=None, scalar_value=None, all_value=None):
        self._items = items or []
        self._scalar_value = scalar_value
        self._all_value = all_value

    def scalars(self):
        return DummyScalars(self._items if self._all_value is None else self._all_value)

    def scalar_one(self):
        return self._scalar_value

    def all(self):
        return self._all_value or []


def test_get_by_id_and_email():
    user = SimpleNamespace(id=1, email="user@example.com")
    db = Mock()
    db.execute = AsyncMock(return_value=DummyResult([user]))
    repo = UsersRepository(db)

    assert run(repo.get_by_id(1)) == user
    assert run(repo.get_by_email("user@example.com")) == user


def test_create_user_success(monkeypatch):
    monkeypatch.setattr(users_repository, "load_dotenv", lambda: None)
    monkeypatch.setenv("DEFAULT_CREDITS", "5")
    db = Mock()
    db.commit = AsyncMock(return_value=None)
    db.refresh = AsyncMock(return_value=None)
    repo = UsersRepository(db)

    dto = CreateUserDto(email="user@example.com", password="pass", is_admin=False)
    user = run(repo.create_user(dto, "hashed"))

    assert user.email == "user@example.com"
    assert user.credits == 5


def test_create_user_rolls_back_on_error(monkeypatch):
    monkeypatch.setattr(users_repository, "load_dotenv", lambda: None)
    monkeypatch.setenv("DEFAULT_CREDITS", "5")
    db = Mock()
    db.commit = AsyncMock(side_effect=SQLAlchemyError("boom"))
    db.rollback = AsyncMock(return_value=None)
    repo = UsersRepository(db)

    dto = CreateUserDto(email="user@example.com", password="pass", is_admin=False)

    with pytest.raises(SQLAlchemyError):
        run(repo.create_user(dto, "hashed"))

    db.rollback.assert_awaited_once()


def test_get_users_paged_and_exclude_banned():
    users = [SimpleNamespace(id=1)]
    db = Mock()
    db.execute = AsyncMock(return_value=DummyResult(users))
    repo = UsersRepository(db)

    assert run(repo.get_all_exclude_banned()) == users
    assert run(repo.get_users_paged("user", page=1, page_size=10)) == users


def test_save_changes_and_add_user():
    user = SimpleNamespace(id=1)
    db = Mock()
    db.commit = AsyncMock(return_value=None)
    db.refresh = AsyncMock(return_value=None)
    repo = UsersRepository(db)

    run(repo.save_changes(user))
    run(repo.add_user(user))


def test_update_users_credits_empty_list():
    db = Mock()
    repo = UsersRepository(db)

    assert run(repo.update_users_credits([], credits_to_update=5)) == []


def test_update_users_credits_updates(monkeypatch):
    users = [SimpleNamespace(id=1, credits=10), SimpleNamespace(id=2, credits=1)]
    db = Mock()
    db.execute = AsyncMock(return_value=None)
    db.commit = AsyncMock(return_value=None)
    db.refresh = AsyncMock(return_value=None)
    repo = UsersRepository(db)

    result = run(repo.update_users_credits(users, credits_to_update=2))

    assert result == users
    assert db.refresh.await_count == 2


class HashableUser(SimpleNamespace):
    __hash__ = object.__hash__


def test_count_users_and_most_active():
    db = Mock()
    db.execute = AsyncMock(return_value=DummyResult(scalar_value=4, all_value=[(HashableUser(id=1), 10)]))
    repo = UsersRepository(db)

    assert run(repo.count_active_users()) == 4
    assert run(repo.count_inactive_users()) == 4
    assert run(repo.count_banned_users()) == 4

    most_active = run(repo.get_most_active_users())
    assert list(most_active.values())[0] == 10
