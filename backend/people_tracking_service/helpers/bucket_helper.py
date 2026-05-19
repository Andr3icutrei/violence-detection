import os

import aiofiles
import aioboto3
from dotenv import load_dotenv

load_dotenv()

ACCOUNT_ID = os.getenv("ACCOUNT_ID")
ACCESS_KEY = os.getenv("ACCESS_KEY")
BUCKET_NAME_DATASETS = os.getenv("BUCKET_NAME_DATASETS")
SECRET_AWS_KEY = os.getenv("SECRET_AWS_KEY")

session = aioboto3.Session()

async def download_object_to_file(object_key: str, target_path: str, chunk_size: int = 1024 * 1024) -> bool:
    async with session.client(
        service_name='s3',
        endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_AWS_KEY,
        region_name="auto"
    ) as s3_client:
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