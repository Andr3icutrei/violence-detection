import os

import aioboto3
import botocore
from dotenv import load_dotenv
from fastapi import HTTPException
from starlette.responses import StreamingResponse

session = aioboto3.Session()

async def get_object_from_bucket(bucket_name: str, object_name: str):
    load_dotenv()

    ACCOUNT_ID = os.getenv("ACCOUNT_ID")
    ACCESS_KEY = os.getenv("ACCESS_KEY")
    BUCKET_NAME = os.getenv("BUCKET_NAME")
    SECRET_AWS_KEY = os.getenv("SECRET_AWS_KEY")

    async with session.client(
            service_name='s3',
            endpoint_url=f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_AWS_KEY,
            region_name="auto"
    ) as s3_client:
        try:
            response = s3_client.get_object(Bucket=BUCKET_NAME, Key=object_name)

            async def iterfile():
                async for data in response["Body"]:
                    yield data

            return StreamingResponse(
                iterfile(),
                media_type=response.get('ContentType', 'application/octet-stream'),
                headers={
                    "Content-Disposition": f'attachment; filename="{object_name}"'
                }
            )
        except botocore.exceptions.ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == "NoSuchKey":
                raise HTTPException(status_code=404, detail="File not found")
            else:
                raise HTTPException(status_code=500, detail=f"Iinternal server error: {str(e)}")

