import os

import pytest

from api.routers.videos_router import _cleanup_temp_file, _format_score, _media_type_for_path


def test_format_score_rounds_to_two_decimals():
    assert _format_score(0.876) == "0.88"


def test_format_score_raises_for_invalid_value():
    with pytest.raises(Exception):
        _format_score("bad")


def test_media_type_for_path():
    assert _media_type_for_path("video.avi") == "video/x-msvideo"
    assert _media_type_for_path("video.mp4") == "video/mp4"


def test_cleanup_temp_file(tmp_path):
    file_path = tmp_path / "temp.txt"
    file_path.write_text("data", encoding="utf-8")
    _cleanup_temp_file(str(file_path))
    assert not os.path.exists(file_path)

