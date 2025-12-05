from fastapi import FastAPI
from app.routes.analyze_route import router as analyze_router

app = FastAPI(
    title="Bowliverse v13",
    version="13.0.0"
)

# Register analyze endpoint
app.include_router(analyze_router, tags=["analysis"])
