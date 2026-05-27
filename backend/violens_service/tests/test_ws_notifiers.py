import asyncio

from starlette.websockets import WebSocketState

from api.routers.datasets_ws_router import DatasetUpdatedWs
from api.routers.users_ws_router import UserUpdatedWs


def run(coro):
    return asyncio.run(coro)


class DummyWebSocket:
    def __init__(self, state: WebSocketState, should_fail: bool = False) -> None:
        self.client_state = state
        self.accepted = False
        self.should_fail = should_fail
        self.sent = []

    async def accept(self):
        self.accepted = True

    async def send_json(self, payload):
        if self.should_fail:
            raise RuntimeError("send failed")
        self.sent.append(payload)


def test_user_ws_broadcast_removes_disconnected():
    ws = UserUpdatedWs()
    connected = DummyWebSocket(WebSocketState.CONNECTED)
    disconnected = DummyWebSocket(WebSocketState.DISCONNECTED)

    run(ws.connect(connected))
    run(ws.connect(disconnected))

    run(ws.broadcast_user_updated(5))

    assert connected.sent[0]["type"] == "user_updated"
    assert disconnected not in ws._connections


def test_dataset_ws_broadcast_removes_failed():
    ws = DatasetUpdatedWs()
    connected = DummyWebSocket(WebSocketState.CONNECTED)
    failing = DummyWebSocket(WebSocketState.CONNECTED, should_fail=True)

    run(ws.connect(connected))
    run(ws.connect(failing))

    run(ws.broadcast_dataset_updated(3))

    assert connected.sent[0]["type"] == "dataset_updated"
    assert failing not in ws._connections

