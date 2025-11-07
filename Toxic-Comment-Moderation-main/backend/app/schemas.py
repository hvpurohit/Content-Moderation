from pydantic import BaseModel, Field
from typing import List, Optional


class ModerateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to moderate")


class ModerateResponse(BaseModel):
    prob: float = Field(..., ge=0.0, le=1.0, description="Probability of toxic class")
    label: str = Field(..., description="TOXIC or NON-TOXIC")
    threshold: float = Field(..., description="Threshold used for classification")


class BatchModerateRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to moderate")


class BatchModerateResponse(BaseModel):
    probs: List[float] = Field(..., description="Probabilities for each text")
    labels: List[str] = Field(..., description="Labels for each text")
    threshold: float = Field(..., description="Threshold used for classification")


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Service status")


class MetricsResponse(BaseModel):
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    test_auc: float
    threshold: float


class ConfigResponse(BaseModel):
    threshold: float = Field(..., description="Current threshold value")

