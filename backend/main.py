from fastapi import FastAPI
from routers import documents, qa

app = FastAPI(title="DocSage API")

app.include_router(documents.router)
app.include_router(qa.router)


@app.get("/")
def root():
    return {"message": "DocSage backend is alive"}


@app.get("/health")
def health_check():
    return {"status": "ok"}

