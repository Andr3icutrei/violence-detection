import os

from api.dependencies import get_videos_service
from tests.fakes import FakeVideosService


def test_get_videos_paged(client):
    response = client.get("/videos/get_videos_paged")
    assert response.status_code == 200
    assert response.json()[0]["name"] == "video.mp4"


def test_exists_video(client):
    response = client.get("/videos/exists_video/exists")
    assert response.status_code == 200


def test_missing_video_returns_404(client):
    response = client.get("/videos/exists_video/missing")
    assert response.status_code == 404


def test_classify_video_gradcam_returns_file(client, app, tmp_path):
    service = FakeVideosService(temp_dir=str(tmp_path))
    app.dependency_overrides[get_videos_service] = lambda: service

    response = client.post("/videos/classify_video_gradcam/1")
    assert response.status_code == 200
    assert response.headers["x-predicted-label"] == "violent"
    assert response.headers["content-type"].startswith("video/")
    assert service.last_temp_path is not None
    assert not os.path.exists(service.last_temp_path)


def test_people_tracking_returns_file(client, app, tmp_path):
    service = FakeVideosService(temp_dir=str(tmp_path))
    app.dependency_overrides[get_videos_service] = lambda: service

    response = client.post("/videos/people_tracking/1")
    assert response.status_code == 200
    assert response.headers["x-tracked-people-count"] == "3"
    assert service.last_temp_path is not None
    assert not os.path.exists(service.last_temp_path)

