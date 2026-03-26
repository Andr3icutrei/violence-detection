from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from repositories.users_repository import UsersRepository


class VideosService:
    def __init__(self):
        self.users_repository = UsersRepository()

    async def get_videos_paged(
            self,
            db: AsyncSession,
            search_term: str,
            asc: bool = True,
            page: int = 0,
            page_size: int = 40) -> List[Video]:

