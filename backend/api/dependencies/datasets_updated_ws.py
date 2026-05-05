from fastapi import Depends

from api.routers.datasets_ws_router import DatasetUpdatedWs, get_datasets_updated


def get_datasets_updated_ws(
    datasets_updated_ws: DatasetUpdatedWs = Depends(get_datasets_updated),
) -> DatasetUpdatedWs:
    return datasets_updated_ws

