from fastapi import FastAPI

# HATA BURADAYDI: "from routers.qa_router" yerine dosya adınız olan "routers.qa" kullanılmalı
from routers.documents import router as documents_router
from routers.qa import router as qa_router 

app = FastAPI(title="Document QA API")

# Router'ları ana uygulamaya ekliyoruz
app.include_router(documents_router)
app.include_router(qa_router)