import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.routers import users_router, auth_router, videos_router, datasets_router
from exception_handling.exception_handler import global_exception_handler

app = FastAPI(title="Violence Detection API")

app.include_router(users_router.router)
app.include_router(videos_router.router)
app.include_router(auth_router.router)
app.include_router(datasets_router.router)

origins = [
    "http://localhost:4200",
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
    return {"message": "Violence Detection API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)