import os
from typing import Any, List
import tempfile
from contextlib import asynccontextmanager

import aiofiles
import aioboto3
from dotenv import load_dotenv
from fastapi import UploadFile, HTTPException
from starlette import status

from helpers.model_validation_helper import validate_3dcnn_onnx

load_dotenv()

ACCOUNT_ID = os.getenv("ACCOUNT_ID")
ACCESS_KEY = os.getenv("ACCESS_KEY")
BUCKET_NAME_DATASETS = os.getenv("BUCKET_NAME_DATASETS")
BUCKET_NAME_MODELS = os.getenv("BUCKET_NAME_MODELS")
MAX_MODEL_SIZE_BYTES = 500 * 1024 * 1024
MODEL_EXTENSION = ".onnx"

SECRET_AWS_KEY = os.getenv("SECRET_AWS_KEY")

session = aioboto3.Session()

@asynccontextmanager
async def _s3_client():
    async with session.client(
        service_name='s3',
        endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_AWS_KEY,
        region_name="auto"
    ) as s3_client:
        yield s3_client

async def get_presigned_url(object_key: str, expiration: int = 3600) -> str:
    async with _s3_client() as s3_client:
        url = await s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': BUCKET_NAME_DATASETS,
                'Key': object_key
            },
            ExpiresIn=expiration
        )
        return url

async def get_object(object_key: str) -> Any :
    async with _s3_client() as s3_client:
        response = await s3_client.get_object(Bucket=BUCKET_NAME_DATASETS, Key=object_key)
        return response


async def put_object(object_key: str, body: bytes, content_type: str = "application/octet-stream") -> None:
    async with _s3_client() as s3_client:
        await s3_client.put_object(
            Bucket=BUCKET_NAME_DATASETS,
            Key=object_key,
            Body=body,
            ContentType=content_type,
        )


async def download_object_to_file(object_key: str, target_path: str, chunk_size: int = 1024 * 1024) -> bool:
    async with _s3_client() as s3_client:
        try:
            response = await s3_client.get_object(Bucket=BUCKET_NAME_DATASETS, Key=object_key)
        except Exception:
            return False

        async with aiofiles.open(target_path, "wb") as file_obj:
            while True:
                chunk = await response["Body"].read(chunk_size)
                if not chunk:
                    break
                await file_obj.write(chunk)

        return True

async def create_unofficial_dataset_bucket(dataset_name: str, videos: List[UploadFile]) -> None:
    if not videos:
        return
    for video in videos:
        if not video.filename.lower().endswith(".mp4"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file format for '{video.filename}'. Only .mp4 files are allowed."
            )
    try:
        for video in videos:
            object_key = f"{dataset_name}/{video.filename}"
            await video.seek(0)
            content = await video.read()
            await put_object(
                object_key=object_key,
                body=content,
                content_type=video.content_type
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload videos: {str(e)}"
        )

async def upload_inference_model(dataset_name: str, model_file: UploadFile) -> str:
    if not model_file.filename or not model_file.filename.lower().endswith(MODEL_EXTENSION):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model format for '{model_file.filename}'. Only {MODEL_EXTENSION} files are allowed."
        )
    temp_path = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=MODEL_EXTENSION)
        temp_path = temp_file.name
        temp_file.close()
        await model_file.seek(0)
        size_bytes = 0
        async with aiofiles.open(temp_path, "wb") as output:
            while True:
                chunk = await model_file.read(1024 * 1024)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > MAX_MODEL_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Inference model exceeds 500MB limit."
                    )
                await output.write(chunk)

        validate_3dcnn_onnx(temp_path)
    finally:
        await model_file.close()

    if not BUCKET_NAME_MODELS:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BUCKET_NAME_MODELS is not configured."
        )

    safe_name = os.path.basename(model_file.filename)
    object_key = f"models/{dataset_name}/{safe_name}"
    try:
        async with _s3_client() as s3_client:
            with open(temp_path, "rb") as file_obj:
                await s3_client.put_object(
                    Bucket=BUCKET_NAME_MODELS,
                    Key=object_key,
                    Body=file_obj,
                    ContentType=model_file.content_type or "application/octet-stream",
                )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    return object_key


async def delete_dataset_videos(dataset_name: str) -> None:
    async with _s3_client() as s3_client:
        paginator = s3_client.get_paginator('list_objects_v2')
        async for page in paginator.paginate(Bucket=BUCKET_NAME_DATASETS, Prefix=f"{dataset_name}/"):
            if 'Contents' in page:
                delete_objects = [{'Key': obj['Key']} for obj in page['Contents']]
                await s3_client.delete_objects(Bucket=BUCKET_NAME_DATASETS, Delete={'Objects': delete_objects})

async def delete_inference_model_object(object_key: str) -> None:
    if not object_key:
        return
    if not BUCKET_NAME_MODELS:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BUCKET_NAME_MODELS is not configured."
        )
    async with _s3_client() as s3_client:
        await s3_client.delete_object(Bucket=BUCKET_NAME_MODELS, Key=object_key)


async def delete_dataset_video_objects(object_keys: List[str]) -> None:
    if not object_keys:
        return
    async with _s3_client() as s3_client:
        chunk_size = 1000
        for i in range(0, len(object_keys), chunk_size):
            chunk = object_keys[i:i + chunk_size]
            delete_objects = [{'Key': key} for key in chunk]
            await s3_client.delete_objects(
                Bucket=BUCKET_NAME_DATASETS,
                Delete={'Objects': delete_objects}
            )

async def get_used_storage_gb() -> float:
    total_size_bytes = 0
    async with _s3_client() as s3_client:
        paginator = s3_client.get_paginator('list_objects_v2')
        async for page in paginator.paginate(Bucket=BUCKET_NAME_DATASETS):
            if 'Contents' in page:
                total_size_bytes += sum(obj['Size'] for obj in page['Contents'])
    total_size_gb = total_size_bytes / (1024 ** 3)
    return total_size_gb