def test_users_ws_connect(client):
    with client.websocket_connect("/users_ws/user-updated") as websocket:
        websocket.send_text("ping")


def test_datasets_ws_connect(client):
    with client.websocket_connect("/datasets_ws/dataset_updated") as websocket:
        websocket.send_text("ping")

