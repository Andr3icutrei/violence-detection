import asyncio
import json
from types import SimpleNamespace

from fastapi import HTTPException

from exception_handling.exception_handler import global_exception_handler


def run(coro):
    return asyncio.run(coro)


def test_global_exception_handler_http_exception():
    response = run(global_exception_handler(SimpleNamespace(), HTTPException(status_code=404, detail="Not found")))

    assert response.status_code == 404
    assert json.loads(response.body.decode("utf-8")) == {"detail": "Not found"}


def test_global_exception_handler_generic_exception():
    response = run(global_exception_handler(SimpleNamespace(), RuntimeError("boom")))

    assert response.status_code == 500
    body = json.loads(response.body.decode("utf-8"))
    assert body["error"] == "Internal Server Error"

