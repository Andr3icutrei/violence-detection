def test_get_inference_actions_for_dataset(client):
    response = client.get("/inference_actions/get_inference_actions_for_dataset/1")
    assert response.status_code == 200
    assert response.json()[0]["credits"] == 5


def test_get_inference_actions_stats(client):
    response = client.get("/inference_actions/get_inference_actions_stats")
    assert response.status_code == 200
    assert response.json()[0]["credits"] == 3


def test_update_inference_actions(client):
    payload = {"actions": [{"id": 1, "new_credits": 10}]}
    response = client.patch("/inference_actions/update_credits_inference_actions", json=payload)
    assert response.status_code == 200

