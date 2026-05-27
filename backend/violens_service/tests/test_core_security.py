import pytest

pytest.importorskip("bcrypt")

from core.security import get_password_hash, verify_password


def test_password_hash_and_verify():
    hashed = get_password_hash("secret")

    assert verify_password("secret", hashed) is True
    assert verify_password("wrong", hashed) is False


def test_password_hash_truncates_to_72_chars():
    base = "a" * 72
    password_a = base + "x"
    password_b = base + "y"

    hashed = get_password_hash(password_a)

    assert verify_password(password_b, hashed) is True
