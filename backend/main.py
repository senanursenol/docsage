from fastapi import FastAPI

from routers.documents import router as documents_router
from routers.qa_router import router as qa_router

app = FastAPI(title="Document QA API")

app.include_router(documents_router)
app.include_router(qa_router)
