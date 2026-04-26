from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter
from starlette import status
from starlette.websockets import WebSocket, WebSocketState, WebSocketDisconnect

router = APIRouter(prefix="/datasets_ws", tags=["Datasets WebSocket"])

class DatasetUpdatedWs:
    def __init__(self):
        self._connections = []

    async def connect(self, websocket: WebSocket)-> None:
        await websocket.accept()
        self._connections.append(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast_dataset_updated(self, dataset_id: int):
        payload = {
            "type": "dataset_updated",
            "data": {
                "datasetId": dataset_id,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
            }
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

dataset_updates_ws = DatasetUpdatedWs()

@router.websocket("/dataset_updated")
async def dataset_updated(websocket: WebSocket):
    await dataset_updates_ws.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await dataset_updates_ws.disconnect(websocket)
    except Exception:
        await dataset_updates_ws.disconnect(websocket)
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)