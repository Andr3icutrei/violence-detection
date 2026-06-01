import os
import numpy as np
import pytest
from fastapi import HTTPException

from helpers import inference_helper


class DummyWriter:
    def __init__(self, output_path):
        self._output_path = output_path
        self._opened = True

    def isOpened(self):
        return self._opened

    def write(self, _frame):
        with open(self._output_path, "ab") as handle:
            handle.write(b"frame")

    def release(self):
        self._opened = False


class DummyCodecWriter(DummyWriter):
    def __init__(self, output_path, opened=True):
        super().__init__(output_path)
        self._opened = opened


class DummyCapture:
    def __init__(self, frames, opened=True, fps=24.0, width=320, height=240, frame_count=0):
        self._frames = list(frames)
        self._opened = opened
        self._fps = fps
        self._width = width
        self._height = height
        self._frame_count = frame_count
        self._index = 0

    def isOpened(self):
        return self._opened

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def get(self, prop_id):
        if prop_id == 5:  # CAP_PROP_FPS
            return self._fps
        if prop_id == 3:  # CAP_PROP_FRAME_WIDTH
            return self._width
        if prop_id == 4:  # CAP_PROP_FRAME_HEIGHT
            return self._height
        if prop_id == 7:  # CAP_PROP_FRAME_COUNT
            return self._frame_count
        return 0

    def release(self):
        self._opened = False


class DummyCV2:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FRAME_COUNT = 7
    COLOR_RGB2BGR = 1
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self, capture):
        self._capture = capture

    def VideoCapture(self, _path):
        return self._capture

    def resize(self, frame, size):
        width, height = size
        return np.zeros((height, width, 3), dtype=frame.dtype)

    def cvtColor(self, frame, _code):
        return frame


class DummyCV2WithWriter(DummyCV2):
    def __init__(self, capture, writer_factory):
        super().__init__(capture)
        self._writer_factory = writer_factory

    def VideoWriter(self, output_path, _fourcc, _fps, _size):
        return self._writer_factory(output_path)

    def VideoWriter_fourcc(self, *_args):
        return 0

    def rectangle(self, *_args, **_kwargs):
        return None

    def putText(self, *_args, **_kwargs):
        return None


class DummyTensor:
    def __init__(self, data):
        self._data = data

    def int(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._data

    def numpy(self):
        return self._data


class DummyBoxes:
    def __init__(self):
        self.id = DummyTensor([1])
        self.xyxy = DummyTensor([[0, 0, 10, 10]])


class DummyResult:
    def __init__(self):
        self.boxes = DummyBoxes()


class DummyModel:
    def __init__(self):
        self.predictor = None

    def track(self, **_kwargs):
        return [DummyResult()]


def test_create_temp_writer_with_forced_codec(tmp_path, monkeypatch):
    output_path = str(tmp_path / "forced.avi")

    def _writer_factory(path):
        return DummyCodecWriter(path, opened=True)

    capture = DummyCapture(frames=[], opened=True)
    dummy_cv2 = DummyCV2WithWriter(capture, _writer_factory)

    monkeypatch.setattr(inference_helper, "cv2", dummy_cv2)
    monkeypatch.setenv("VIDEO_OUTPUT_CODEC", "MJPG")
    monkeypatch.setenv("VIDEO_OUTPUT_EXT", ".avi")

    path, writer = inference_helper._create_temp_writer(24.0, (320, 240))

    assert path.endswith(".avi")
    assert writer.isOpened()


def test_write_overlay_video_raises_when_writer_closed(tmp_path, monkeypatch):
    overlay = np.zeros((240, 320, 3), dtype=np.uint8)
    capture = DummyCapture(frames=[overlay], opened=True)

    def _writer_factory(path):
        return DummyCodecWriter(path, opened=False)

    dummy_cv2 = DummyCV2WithWriter(capture, _writer_factory)

    monkeypatch.setattr(inference_helper, "cv2", dummy_cv2)
    monkeypatch.setattr(
        inference_helper,
        "_create_temp_writer",
        lambda _fps, _size: (str(tmp_path / "out.mp4"), DummyCodecWriter(str(tmp_path / "out.mp4"), opened=False)),
    )

    with pytest.raises(ValueError):
        inference_helper.write_overlay_video([overlay], "source.mp4")


def test_run_people_tracking_success_fallback(tmp_path, monkeypatch):
    frames = [np.zeros((240, 320, 3), dtype=np.uint8)]
    capture = DummyCapture(frames=frames, opened=True, fps=30.0, width=320, height=240)

    tracked_path = str(tmp_path / "tracked.avi")

    def _writer_factory(path):
        return DummyCodecWriter(path, opened=True)

    dummy_cv2 = DummyCV2WithWriter(capture, _writer_factory)

    class DummyBaseTrack:
        @staticmethod
        def reset_id():
            return None

    monkeypatch.setattr(inference_helper, "cv2", dummy_cv2)
    monkeypatch.setattr(inference_helper, "BaseTrack", DummyBaseTrack)
    monkeypatch.setattr(inference_helper, "_create_temp_writer", lambda _fps, _size: (tracked_path, DummyCodecWriter(tracked_path, opened=True)))

    def _raise_ffmpeg(*_args, **_kwargs):
        raise RuntimeError("ffmpeg missing")

    import subprocess

    monkeypatch.setattr(subprocess, "run", _raise_ffmpeg)

    result_path, count = inference_helper.run_people_tracking("video.mp4", DummyModel())

    assert result_path == tracked_path
    assert count == 1
    assert os.path.exists(result_path)


def test_run_people_tracking_video_not_opened(monkeypatch):
    capture = DummyCapture(frames=[], opened=False)
    dummy_cv2 = DummyCV2(capture)

    class DummyBaseTrack:
        @staticmethod
        def reset_id():
            return None

    monkeypatch.setattr(inference_helper, "cv2", dummy_cv2)
    monkeypatch.setattr(inference_helper, "BaseTrack", DummyBaseTrack)

    class DummyModel:
        predictor = None

    with pytest.raises(HTTPException) as excinfo:
        inference_helper.run_people_tracking("missing.mp4", DummyModel())

    assert excinfo.value.status_code == 500
