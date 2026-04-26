from typing import Protocol


class UserEventsNotifier(Protocol):
    async def broadcast_user_updated(self, user_id: int) -> None: ...
