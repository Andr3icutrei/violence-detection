def test_login_sets_cookie(client):
    response = client.post(
        "/auth/login",
        json={"email": "user@example.com", "password": "secret"},
    )
    assert response.status_code == 201
    assert "set-cookie" in response.headers
    assert response.json()["email"] == "user@example.com"


def test_google_login_sets_cookie(client):
    response = client.post(
        "/auth/google-login",
        json={"tokenId": "fake-token"},
    )
    assert response.status_code == 200
    assert "set-cookie" in response.headers


def test_logout_clears_cookie(client):
    response = client.post("/auth/logout")
    assert response.status_code == 200
    assert "set-cookie" in response.headers


def test_me_returns_current_user(client):
    response = client.get("/auth/me")
    assert response.status_code == 200
    assert response.json()["email"] == "user@example.com"

