# VIOLENS Platform

VIOLENS is a video-intelligence platform for violence detection, dataset review, and model-backed inference. It combines a web dashboard with a microservice backend to manage videos, datasets, users, credits, and AI inference flows end to end.

## Core features

- **3D CNN violence detection** on video clips through an ONNX-based pipeline.
- **Explainable inference** with occlusion/Grad-CAM-style heatmaps over the processed video.
- **People tracking** as a separate inference action with tracked-person counts returned in the response headers.
- **Dataset management** for accepted, unofficial, pending, approved, and rejected datasets.
- **Dataset review workflow** with approve/reject/edit actions, per-video labels, excluded videos, and admin notes.
- **Model integration per dataset**: each dataset can carry its own inference model, which is validated before upload and reused during inference.
- **Dataset-model validation** to evaluate the model against reviewed videos and return accuracy/confusion metrics.
- **User management** with registration, login, Google login, account verification, password reset, banning, and role updates.
- **Credit-based usage** for inference actions, including admin-adjustable credit values and daily credit refresh support.
- **Live updates** through WebSockets for user and dataset changes.
- **Analytics dashboards** for users, datasets, inference actions, credits, and storage usage.
- **Multilingual frontend** with translation support and chart visualizations.

## Tech stack

### Frontend

- Angular 21
- TypeScript
- RxJS
- Angular forms and router
- `@ngx-translate/core`
- `@swimlane/ngx-charts`
- `@abacritt/angularx-social-login`

### Backend

- Python
- FastAPI
- SQLAlchemy + Alembic
- asyncpg / PostgreSQL
- Pydantic
- APScheduler
- WebSockets
- OpenCV
- PyTorch / TorchVision
- ONNX Runtime GPU
- Ultralytics YOLO
- boto3 / aioboto3 for object storage
- ffmpeg for browser-compatible video output

## Platform architecture

VIOLENS is split into three backend services:

1. **Main API** (`violens_service`)  
   Handles authentication, users, datasets, credits, inference history, video listing, and WebSocket notifications.

2. **Classification service** (`classification_service`)  
   Loads the 3D CNN ONNX model, preprocesses video frames, runs classification, and generates explainability overlays.

3. **People tracking service** (`people_tracking_service`)  
   Runs the tracking pipeline and returns an annotated video plus the total tracked people count.

The main API orchestrates both inference services and stores the result history, while datasets and inference models are stored in object storage and linked through the database.

## 3D CNN pipeline

The violence detector is built around a **3D CNN ONNX model**. The model is validated before upload to ensure:

- 5D video input shape (`N, C, T, H, W`)
- binary output with 2 classes
- compatible ONNX inference execution

At inference time, the pipeline:

1. loads the ONNX model,
2. preprocesses the input video,
3. extracts spatio-temporal tensors,
4. runs binary classification,
5. produces confidence scores,
6. generates an occlusion heatmap,
7. overlays the heatmap on the output video.

This makes the platform useful not only for prediction, but also for understanding why a clip was classified as violent or non-violent.

## Dataset and model integration

Datasets are first-class objects in the platform. A dataset can include:

- uploaded videos,
- a review status,
- an official/unofficial flag,
- an associated inference model,
- review comments and video-level labels.

During dataset creation, the user can upload both videos and an ONNX inference model. The platform stores the model separately, links it to the dataset, and validates it before accepting the dataset payload.

For admins, the review flow supports:

- opening a dataset with its videos,
- marking each video as violent or non-violent,
- excluding videos from the final dataset,
- validating the selected labels against the attached model,
- approving or rejecting the dataset,
- cleaning up rejected assets and unused models automatically.

This is the main idea behind VIOLENS: **dataset quality, model quality, and inference history are all connected in one workflow**.

## Extra platform features

- Secure auth with HTTP-only JWT cookies.
- Google login support.
- Account verification and password recovery.
- Admin-only pages for users, datasets, statistics, and credit management.
- Inference history tracking per user and per video.
- Dataset and user updates pushed live to the UI.
- Presigned video URLs for secure media access.
- Support for HTTPS on the backend services.

## Docker setup

The backend already includes Dockerfiles for all services and a `docker-compose.yml` in `backend/`.

### Requirements

- Docker Desktop or Docker Engine
- Docker Compose
- NVIDIA GPU support for the classification and tracking services

### Run the backend

From the `backend` directory:

```bash
docker compose up --build
```

This starts:

- `violens-service` on `https://localhost:8000`
- `people-tracking-service` on `https://localhost:8001`
- `classification-service` on `https://localhost:8002`

### Environment configuration

Each service reads its own `.env` file. Before starting the stack, configure the required values, especially:

- `DATABASE_URL`
- `CLASSIFICATION_SERVICE_URL`
- `PEOPLE_TRACKING_SERVICE_URL`
- `YOLO_MODEL_PATH`
- `DEFAULT_CREDITS`
- `GOOGLE_AUTH_CLIENT_ID`
- `BUCKET_NAME_DATASETS`
- `BUCKET_NAME_MODELS`
- `SECRET_AWS_KEY`
- mail credentials for dataset/user emails

If you deploy the frontend elsewhere, also update the frontend environment URLs so they point to the correct backend domain.

## Frontend

The frontend is an Angular application. Install dependencies and run it with:

```bash
cd frontend
npm install
npm start
```

## Main user areas

- **Dashboard** for browsing videos and starting inference.
- **Inference page** with action selection, result video playback, confidence values, and tracked-person statistics.
- **Datasets page** for browsing accepted datasets.
- **Review datasets page** for admin moderation and model validation.
- **Users page** for admin user management.
- **Stats page** for charts and platform metrics.
- **Portal** for login, registration, verification, and password reset.
