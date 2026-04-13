import uvicorn
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from api.routers import users_router, auth_router, videos_router, datasets_router, inference_actions_router
from api.routers.inference_actions_router import inference_actions_service
from core.database import get_db
from exception_handling.exception_handler import global_exception_handler
from services.inference_runtime import load_inference_runtime
from services.users_service import UsersService
from services.videos_service import VideosService


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference_runtime = load_inference_runtime()
    app.state.inference_runtime = inference_runtime
    app.state.videos_service = VideosService(inference_runtime)
    try:
        yield
    finally:
        app.state.videos_service = None
        app.state.inference_runtime = None


scheduler = AsyncIOScheduler()
app = FastAPI(title="Violence Detection API", lifespan=lifespan)

app.include_router(users_router.router)
app.include_router(videos_router.router)
app.include_router(auth_router.router)
app.include_router(datasets_router.router)
app.include_router(inference_actions_router.router)

origins = [
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Predicted-Label",
        "X-Confidence",
        "X-Predicted-Class-Probability",
        "X-Tracked-People-Count"
    ],
)

app.add_exception_handler(Exception, global_exception_handler)

async def cron_wrapper():
    from core.database import SessionLocal
    async with SessionLocal() as db:
        users_service = UsersService()
        await users_service.update_all_users_credits(db)

@app.on_event("startup")
async def schedule_jobs():
    trigger = CronTrigger(hour=0, minute=0)
    scheduler.add_job(cron_wrapper, trigger=trigger)
    scheduler.start()

@app.get("/")
def root():
    return {"message": "Violence Detection API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)