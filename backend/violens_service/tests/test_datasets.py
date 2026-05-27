def test_get_accepted_datasets(client):
    response = client.get("/datasets/get_accepted_datasets")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_create_unofficial_dataset(client):
    files = [
        ("videos", ("video1.mp4", b"data", "video/mp4")),
        ("inference_model", ("model.onnx", b"model", "application/octet-stream")),
    ]
    data = {"name": "Dataset A", "inference_model_name": "model.onnx"}
    response = client.post("/datasets/create_unofficial_dataset", data=data, files=files)
    assert response.status_code == 200


def test_get_datasets(client):
    response = client.get("/datasets/get_datasets", params={"page": 1, "page_size": 10})
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_get_dataset_videos(client):
    response = client.get("/datasets/get_dataset_videos/1")
    assert response.status_code == 200
    assert response.json()["id"] == 1


def test_review_dataset(client):
    payload = {
        "is_approved": True,
        "videos": [{"video_id": 1, "is_violent": False}],
        "review_comment": "ok",
        "excluded_video_ids": [],
    }
    response = client.patch("/datasets/review_dataset/1", json=payload)
    assert response.status_code == 200


def test_delete_dataset(client):
    response = client.delete("/datasets/delete_dataset/1")
    assert response.status_code == 200


def test_edit_dataset(client):
    payload = {
        "videos": [{"video_id": 1, "is_violent": False}],
        "excluded_video_ids": [],
    }
    response = client.patch("/datasets/edit_dataset/1", json=payload)
    assert response.status_code == 200


def test_validate_dataset_model(client):
    payload = {
        "videos": [{"video_id": 1, "is_violent": False}],
        "excluded_video_ids": [],
    }
    response = client.post("/datasets/validate_dataset_model/1", json=payload)
    assert response.status_code == 200
    assert response.json()["accuracy"] == 0.9


def test_get_datasets_stats(client):
    response = client.get("/datasets/get_datasets_stats")
    assert response.status_code == 200
    assert response.json()["official_datasets_count"] == 2

