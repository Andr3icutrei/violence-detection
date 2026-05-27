def test_get_inference_history_stats(client):
    response = client.get("/inference_history/get_inference_history_stats", params={"year": 2024, "month": 1})
    assert response.status_code == 200
    assert len(response.json()["classification_runs"]) == 1

