def test_get_credits_cronjob_update(client):
    response = client.get("/credits/get_credits_cronjob_update")
    assert response.status_code == 200
    assert response.json()["default_credits"] == 10


def test_patch_credits_cronjob_update(client):
    response = client.patch("/credits/patch_credits_cronjob_update", params={"new_credits": 15})
    assert response.status_code == 200

