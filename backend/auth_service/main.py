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
from services.users_service import UsersService


app = FastAPI(title="Violens auth API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-User-Id",
        "X-User-Email",
        "X-User-Role",
    ],
)

app.add_exception_handler(Exception, global_exception_handler)

@app.get("/")
def root():
    return {"message": "Violence auth API is running"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="./key.pem",
        ssl_certfile="./cert.pem"
    )