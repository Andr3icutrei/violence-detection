import uvicorn
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.params import Depends
from starlette.middleware.cors import CORSMiddleware
from apscheduler.triggers.cron import CronTrigger

from api.dependencies import get_users_service
from api.routers import users_router, auth_router, videos_router, datasets_router, inference_actions_router, \
    inference_history_router, users_ws_router, datasets_ws_router, credits_router
from exception_handling.exception_handler import global_exception_handler
from helpers.env_helper import get_env_variable
from services.inference_runtime import load_inference_runtime
from services.users_service import UsersService

@asynccontextmanager
async def lifespan(app: FastAPI):
    inference_runtime = load_inference_runtime()
    app.state.inference_runtime = inference_runtime

    trigger = CronTrigger(hour=0, minute=0)
    scheduler.add_job(cron_wrapper, trigger=trigger)
    scheduler.start()

    try:
        yield
    finally:
        scheduler.shutdown()
        app.state.videos_service = None
        app.state.inference_runtime = None


scheduler = AsyncIOScheduler()
app = FastAPI(title="Violence Detection API", lifespan=lifespan)

app.include_router(users_router.router)
app.include_router(videos_router.router)
app.include_router(auth_router.router)
app.include_router(datasets_router.router)
app.include_router(inference_actions_router.router)
app.include_router(inference_history_router.router)
app.include_router(users_ws_router.router)
app.include_router(datasets_ws_router.router)
app.include_router(credits_router.router)

origins = [
    origin.strip()
    for origin in get_env_variable("CORS_ALLOW_ORIGINS", "https://localhost:4200").split(",")
    if origin.strip()
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

async def cron_wrapper(users_service: UsersService = Depends(get_users_service)):
    await users_service.update_all_users_credits()

@app.get("/")
def root():
    return {"message": "Violence Detection API is running"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem"
    )