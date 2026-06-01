import os

from core.routers.people_tracking_router import _cleanup_temp_file, _media_type_for_path


def test_media_type_for_path():
    assert _media_type_for_path("video.avi") == "video/x-msvideo"
    assert _media_type_for_path("video.mp4") == "video/mp4"
    assert _media_type_for_path("VIDEO.MP4") == "video/mp4"


def test_cleanup_temp_file_removes_existing(tmp_path):
    target = tmp_path / "temp.mp4"
    target.write_bytes(b"data")

    _cleanup_temp_file(str(target))

    assert not target.exists()


def test_cleanup_temp_file_missing(tmp_path):
    target = tmp_path / "missing.mp4"
    _cleanup_temp_file(str(target))
    assert not target.exists()

