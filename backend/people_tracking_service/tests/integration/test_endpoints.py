import os

from fastapi import status

from core.dependencies.people_tracking_service import get_classification_service


class StubPeopleTrackingService:
    def __init__(self, overlay_path, people_count, temp_path):
        self._overlay_path = overlay_path
        self._people_count = people_count
        self._temp_path = temp_path

    async def people_tracking(self, _video_path):
        return self._overlay_path, self._people_count, self._temp_path


def test_root_health(client):
    response = client.get("/")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "Violence People tracking is running"}


def test_people_tracking_response(client, app, tmp_path):
    overlay_path = str(tmp_path / "overlay.mp4")
    temp_path = str(tmp_path / "temp.mp4")
    with open(overlay_path, "wb") as handle:
        handle.write(b"overlay")
    with open(temp_path, "wb") as handle:
        handle.write(b"temp")

    app.dependency_overrides[get_classification_service] = lambda: StubPeopleTrackingService(
        overlay_path=overlay_path,
        people_count=5,
        temp_path=temp_path,
    )

    response = client.get("/people_tracking", params={"video_path": "remote.mp4"})

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"video_path": overlay_path, "people_tracked": "5"}


def test_people_tracking_stream_response(client, app, tmp_path):
    overlay_path = str(tmp_path / "overlay.avi")
    temp_path = str(tmp_path / "temp.mp4")
    with open(overlay_path, "wb") as handle:
        handle.write(b"overlay")
    with open(temp_path, "wb") as handle:
        handle.write(b"temp")

    app.dependency_overrides[get_classification_service] = lambda: StubPeopleTrackingService(
        overlay_path=overlay_path,
        people_count=2,
        temp_path=temp_path,
    )

    response = client.get("/people_tracking/stream", params={"video_path": "remote.mp4"})

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["X-Tracked-People-Count"] == "2"
    assert response.headers["content-type"].startswith("video/x-msvideo")
    assert not os.path.exists(overlay_path)
    assert not os.path.exists(temp_path)

