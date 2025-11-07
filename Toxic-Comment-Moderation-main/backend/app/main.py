import json
import logging
import os
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from app.schemas import (
    ModerateRequest,
    ModerateResponse,
    BatchModerateRequest,
    BatchModerateResponse,
    HealthResponse,
    MetricsResponse,
    ConfigResponse
)
from app.model_loader import ModelLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model loader instance (will be initialized in lifespan)
model_loader = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown event handler."""
    global model_loader
    # Startup
    try:
        logger.info("Starting up... Loading model and tokenizer...")
        logger.info("=" * 60)
        model_loader = ModelLoader()
        if model_loader.is_loaded():
            logger.info("=" * 60)
            logger.info("✓ Model loaded successfully!")
            logger.info(f"✓ Device: {model_loader.device}")
            logger.info(f"✓ Threshold: {model_loader.threshold}")
            logger.info("=" * 60)
        else:
            raise RuntimeError("Model failed to load - check logs for details")
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"✗ Failed to initialize model: {e}")
        logger.error("✗ Server will not start without a loaded model.")
        logger.error("=" * 60)
        logger.error("\nTroubleshooting steps:")
        logger.error("1. Check that 'final_model/' directory exists in project root")
        logger.error("2. Verify all model files are present (config.json, model.safetensors, etc.)")
        logger.error("3. Check that 'threshold.json' exists in project root")
        logger.error("4. Verify all dependencies are installed: pip install -r requirements.txt")
        logger.error("=" * 60)
        raise

    yield

    # Shutdown
    logger.info("Shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Toxic Comment Moderation API",
    description="API for moderating toxic comments using DistilBERT",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend access - allow all localhost ports in development
# Get allowed origins from environment or use defaults
ALLOWED_ORIGINS: List[str] = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://localhost:5174,http://127.0.0.1:5173,http://127.0.0.1:3000,http://127.0.0.1:5174"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with model status verification."""
    if model_loader is None or not model_loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Model not loaded"
        )
    return HealthResponse(status="ok")


@app.post("/moderate", response_model=ModerateResponse)
async def moderate(request: ModerateRequest):
    """
    Moderate a single text comment.

    Returns probability of toxicity, label (TOXIC/NON-TOXIC), and threshold used.
    """
    if model_loader is None or not model_loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Model not loaded"
        )

    try:
        # Sanitize input (basic length check)
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        if len(text) > 10000:
            raise HTTPException(status_code=400, detail="Text exceeds maximum length of 10000 characters")

        # Get prediction
        prob, label = model_loader.predict(text)

        # Log request (without storing raw text for privacy)
        logger.info(
            f"Moderation request - Length: {len(text)}, "
            f"Probability: {prob:.4f}, Label: {label}"
        )

        return ModerateResponse(
            prob=round(prob, 6),
            label=label,
            threshold=model_loader.threshold
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error in moderation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/moderate_batch", response_model=BatchModerateResponse)
async def moderate_batch(request: BatchModerateRequest):
    """
    Moderate multiple texts in batch.

    Returns arrays of probabilities and labels.
    """
    if model_loader is None or not model_loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Model not loaded"
        )

    try:
        # Validate input
        texts = [text.strip() for text in request.texts if text.strip()]

        if not texts:
            raise HTTPException(status_code=400, detail="No valid texts provided")

        if len(texts) > 100:
            raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 100")

        # Get predictions
        probs, labels = model_loader.predict_batch(texts)

        # Log batch request
        logger.info(f"Batch moderation request - Count: {len(texts)}")

        return BatchModerateResponse(
            probs=[round(p, 6) for p in probs],
            labels=labels,
            threshold=model_loader.threshold
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error in batch moderation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get test metrics from final_metrics.json."""
    try:
        project_root = Path(__file__).parent.parent.parent
        metrics_file = project_root / "final_metrics.json"

        if not metrics_file.exists():
            raise HTTPException(status_code=404, detail="Metrics file not found")

        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        return MetricsResponse(**metrics)

    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current threshold configuration."""
    if model_loader is None or not model_loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Model not loaded"
        )
    return ConfigResponse(threshold=model_loader.threshold)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    if isinstance(exc, HTTPException):
        # Let HTTPExceptions pass through
        raise exc
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

