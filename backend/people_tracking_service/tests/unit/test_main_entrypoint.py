import runpy

import uvicorn
import main


def test_main_invokes_uvicorn(monkeypatch):
    called = {}

    def _fake_run(app, host, port, ssl_keyfile, ssl_certfile):
        called["app"] = app
        called["host"] = host
        called["port"] = port
        called["ssl_keyfile"] = ssl_keyfile
        called["ssl_certfile"] = ssl_certfile

    monkeypatch.setattr(uvicorn, "run", _fake_run)

    runpy.run_path(main.__file__, run_name="__main__")

    assert called["app"].title == "Violence Inference API"
    assert called["host"] == "0.0.0.0"
    assert called["port"] == 8001
    assert called["ssl_keyfile"] == "./key.pem"
    assert called["ssl_certfile"] == "./cert.pem"

