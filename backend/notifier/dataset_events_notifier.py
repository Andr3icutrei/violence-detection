from typing import Protocol


class DatasetEventsNotifier(Protocol):
    async def broadcast_dataset_updated(self, dataset_id: int) -> None: ...
