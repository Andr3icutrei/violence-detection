from core.database import _build_async_database_url


def test_build_async_database_url_postgres_psycopg2():
    url = _build_async_database_url("postgresql+psycopg2://user:pass@host/db")
    assert url == "postgresql+asyncpg://user:pass@host/db"


def test_build_async_database_url_postgres_default():
    url = _build_async_database_url("postgresql://user:pass@host/db")
    assert url == "postgresql+asyncpg://user:pass@host/db"


def test_build_async_database_url_sqlite():
    url = _build_async_database_url("sqlite:///app.db")
    assert url == "sqlite+aiosqlite:///app.db"


def test_build_async_database_url_passthrough():
    url = _build_async_database_url("sqlite+aiosqlite:///app.db")
    assert url == "sqlite+aiosqlite:///app.db"

