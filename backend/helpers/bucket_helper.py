import os
import aioboto3
from dotenv import load_dotenv

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

