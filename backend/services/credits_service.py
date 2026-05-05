import os

from dotenv import load_dotenv
from fastapi import HTTPException


class CreditsService:
    async def get_credits_cronjob_update(self) -> int:
        load_dotenv(override=True)
        credits_cronjob_update: int | None = int(os.getenv("DEFAULT_CREDITS"))
        if credits_cronjob_update is None:
            raise HTTPException(
                status_code=500,
                detail="Error loading DEFAULT_CREDITS from environment variables."
            )
        return credits_cronjob_update

    async def patch_credits_cronjob_update(self, new_credits: int) -> None:
        try:
            load_dotenv()
            with open(".env", "r") as file:
                lines = file.readlines()
            with open(".env", "w") as file:
                for line in lines:
                    if line.startswith("DEFAULT_CREDITS="):
                        file.write(f"DEFAULT_CREDITS={new_credits}\n")
                    else:
                        file.write(line)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error updating DEFAULT_CREDITS in environment variables: {str(e)}"
            ) from e