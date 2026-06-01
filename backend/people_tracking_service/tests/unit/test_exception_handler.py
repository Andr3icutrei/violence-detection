import asyncio
from fastapi import HTTPException
from starlette.requests import Request

from exception_handling.exception_handler import global_exception_handler


def run_async(coro):
    return asyncio.run(coro)


def test_global_exception_handler_http_exception():
    scope = {"type": "http", "method": "GET", "path": "/", "headers": []}
    request = Request(scope)
    exc = HTTPException(status_code=400, detail="bad")

    response = run_async(global_exception_handler(request, exc))

    assert response.status_code == 400
    assert response.body == b'{"detail":"bad"}'


def test_global_exception_handler_generic_exception():
    scope = {"type": "http", "method": "GET", "path": "/", "headers": []}
    request = Request(scope)

    response = run_async(global_exception_handler(request, RuntimeError("boom")))

    assert response.status_code == 500
    assert b"Internal Server Error" in response.body
