import os
from typing import Any, List

import aiofiles
import aioboto3
from dotenv import load_dotenv
from fastapi import UploadFile, HTTPException
from starlette import status

load_dotenv()

ACCOUNT_ID = os.getenv("ACCOUNT_ID")
ACCESS_KEY = os.getenv("ACCESS_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")
SECRET_AWS_KEY = os.getenv("SECRET_AWS_KEY")

session = aioboto3.Session()

async def get_presigned_url(object_key: str, expiration: int = 3600) -> str:
    async with session.client(
        service_name='s3',
        endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_AWS_KEY,
        region_name="auto"
    ) as s3_client:
        url = await s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': object_key
            },
            ExpiresIn=expiration
        )
        return url

async def get_object(object_key: str) -> Any :
    async with session.client(
            service_name='s3',
            endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_AWS_KEY,
            region_name="auto"
    ) as s3_client:
        response = await s3_client.get_object(Bucket=BUCKET_NAME, Key=object_key)
        return response


async def put_object(object_key: str, body: bytes, content_type: str = "application/octet-stream") -> None:
    async with session.client(
        service_name='s3',
        endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_AWS_KEY,
        region_name="auto"
    ) as s3_client:
        await s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=object_key,
            Body=body,
            ContentType=content_type,
        )


async def download_object_to_file(object_key: str, target_path: str, chunk_size: int = 1024 * 1024) -> bool:
    async with session.client(
        service_name='s3',
        endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_AWS_KEY,
        region_name="auto"
    ) as s3_client:
        try:
            response = await s3_client.get_object(Bucket=BUCKET_NAME, Key=object_key)
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