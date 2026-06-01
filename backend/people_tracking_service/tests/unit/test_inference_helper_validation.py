import pytest

from helpers.inference_helper import write_overlay_video


def test_write_overlay_video_requires_overlays():
    with pytest.raises(ValueError):
        write_overlay_video([], "video.mp4")


def test_write_overlay_video_invalid_frame():
    with pytest.raises(ValueError):
        write_overlay_video([None], "video.mp4")

