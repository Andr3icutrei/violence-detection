from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from starlette.websockets import WebSocketState

router = APIRouter(prefix="/users_ws", tags=["WebSocket"])

class UserUpdatedWs:
    def __init__(self) -> None:
        self._connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self._connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast_user_updated(self, user_id: int):
        payload = {
            "type": "user_updated",
            "data": {
                "userId": user_id,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            },
        }

        disconnected: List[WebSocket] = []
        for connection in self._connections:
            try:
                if connection.client_state == WebSocketState.CONNECTED:
                    await connection.send_json(payload)
                else:
                    disconnected.append(connection)
            except Exception:
                disconnected.append(connection)

        for connection in disconnected:
            await self.disconnect(connection)

user_role_ws = UserUpdatedWs()

@router.websocket("/user-updated")
async def user_role_updates_socket(websocket: WebSocket):
    await user_role_ws.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await user_role_ws.disconnect(websocket)
    except Exception:
        await user_role_ws.disconnect(websocket)
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)