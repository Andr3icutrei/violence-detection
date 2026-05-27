def test_create_user(client):
    response = client.post(
        "/users/create",
        json={"email": "user@example.com", "password": "secret"},
    )
    assert response.status_code == 201
    assert response.json()["email"] == "user@example.com"


def test_verify_account(client):
    response = client.patch("/users/verify_account", params={"token": "token"})
    assert response.status_code == 200


def test_reset_password(client):
    response = client.patch(
        "/users/reset_password",
        params={"token": "token", "newPassword": "newpass"},
    )
    assert response.status_code == 200


def test_request_reset_password(client):
    response = client.get("/users/request_reset_password", params={"email": "user@example.com"})
    assert response.status_code == 200
    assert "message" in response.json()


def test_verify_reset_password_token(client):
    response = client.get("/users/verify_reset_password_token", params={"token": "token"})
    assert response.status_code == 200
    assert response.json()["valid"] is True


def test_resend_verification_email(client):
    response = client.get("/users/resend_verification_email", params={"token": "token"})
    assert response.status_code == 200


def test_get_topbar_information(client):
    response = client.get("/users/topbar_information")
    assert response.status_code == 200
    assert response.json()["email"] == "user@example.com"


def test_get_all_users(client):
    response = client.get("/users/get_all_users")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_update_user_role(client):
    response = client.patch(
        "/users/update_user_role",
        params={"user_id": 1, "is_admin": True},
    )
    assert response.status_code == 200


def test_ban_user(client):
    response = client.patch(
        "/users/ban_user/1",
        json={"ban_reason": "spam"},
    )
    assert response.status_code == 200


def test_get_users_stats(client):
    response = client.get("/users/get_users_stats")
    assert response.status_code == 200
    assert response.json()["active_users"] == 3

