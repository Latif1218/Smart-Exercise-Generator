from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import exercise, ocr, help


app = FastAPI(
    title="Smart Exercise Generator API",
    description="""
    Smart Exercise Generator Backend API
    
    This API serves as the backend for the Flutter mobile application.
    
    Workflow:
    1. OCR Endpoints — Upload images and extract text
    2. Exercise Endpoints — Generate questions from extracted text using AI
    
    Main Features:
    - Single/Multiple image OCR (Tesseract)
    - AI Exercise Generation (DeepSeek LLM)
    - MCQ, Fill in the Blank, Short Answer questions
    - Clean JSON responses for Flutter app
    """,
    version="1.0.0",
    docs_url="/docs",        
    redoc_url="/redoc"       
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
async def root():
    """API health check endpoint"""
    return {
        "status": "running",
        "message": "Smart Exercise Generator API is running!",
        "docs": "/docs",
        "version": "1.0.0"
    }



# ==========================================================================
app.include_router(ocr.router, prefix="/api/v1")
app.include_router(exercise.router, prefix="/api/v1")
app.include_router(help.router,  prefix="/api/v1)")

# ==========================================================================




@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check — confirm that the API is functioning properly"""
    return {
        "status": "healthy",
        "services": {
            "ocr": "available",
            "llm": "available"
        }
    }



@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )