import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from core.routers import people_tracking_router
from exception_handling.exception_handler import global_exception_handler
from services.inference_runtime import load_inference_runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference_runtime = load_inference_runtime()
    app.state.inference_runtime = inference_runtime
    try:
        yield
    finally:
        app.state.videos_service = None
        app.state.inference_runtime = None


app = FastAPI(title="Violence Inference API", lifespan=lifespan)

app.include_router(people_tracking_router.router)

origins = [
    "https://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(Exception, global_exception_handler)

@app.get("/")
def root():
    return {"message": "Violence People tracking is running"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        ssl_keyfile="./key.pem",
        ssl_certfile="./cert.pem"
    )