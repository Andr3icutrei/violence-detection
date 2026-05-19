from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from models import InferenceModel, Dataset


class InferenceModelsRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_id(self, model_id: int) -> InferenceModel | None:
        result = await self.db.execute(select(InferenceModel).filter(InferenceModel.id == model_id))
        return result.scalars().first()

    async def create(self, name: str, path: str) -> InferenceModel:
        model = InferenceModel(name=name, path=path)
        self.db.add(model)
        await self.db.commit()
        await self.db.refresh(model)
        return model

    async def delete(self, model: InferenceModel) -> None:
        await self.db.delete(model)
        await self.db.flush()
        await self.db.commit()

    async def count_datasets(self, model_id: int) -> int:
        result = await self.db.execute(
            select(func.count(Dataset.id)).filter(
                Dataset.inference_model_id == model_id,
                Dataset.deleted_at.is_(None)
            )
        )
        return int(result.scalar_one())
