from fastapi import FastAPI
from api.routers import users_router

app = FastAPI(title="Violence Detection API")

app.include_router(users_router.router)

@app.get("/")
def root():
    return {"message": "Violence Detection API is running"}