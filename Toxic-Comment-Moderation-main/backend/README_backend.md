# Backend API Documentation

## Overview

This FastAPI backend provides a REST API for toxic comment moderation using a DistilBERT model. The model and tokenizer are loaded from the local `final_model/` directory at startup.

## Directory Layout

```
backend/
├── app/
│   ├── main.py          # FastAPI application with all endpoints
│   ├── model_loader.py  # Singleton for model/tokenizer management
│   └── schemas.py       # Pydantic request/response models
├── requirements.txt     # Python dependencies
└── README_backend.md    # This file
```

## API Contract

### POST /moderate

Moderate a single text comment.

**Request:**
```json
{
  "text": "Your comment here"
}
```

**Response:**
```json
{
  "prob": 0.853,
  "label": "TOXIC",
  "threshold": 0.9249592423439026
}
```

**Status Codes:**
- 200: Success
- 400: Invalid input (empty text or exceeds max length)
- 500: Server error

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### POST /moderate_batch

Moderate multiple texts in batch.

**Request:**
```json
{
  "texts": ["Text 1", "Text 2", "Text 3"]
}
```

**Response:**
```json
{
  "probs": [0.123, 0.956, 0.234],
  "labels": ["NON-TOXIC", "TOXIC", "NON-TOXIC"],
  "threshold": 0.9249592423439026
}
```

### GET /metrics

Get test metrics from `final_metrics.json`.

**Response:**
```json
{
  "test_accuracy": 0.9558,
  "test_precision": 0.8533,
  "test_recall": 0.6198,
  "test_f1": 0.7181,
  "test_auc": 0.9723,
  "threshold": 0.9249592423439026
}
```

### GET /config

Get current threshold configuration.

**Response:**
```json
{
  "threshold": 0.9249592423439026
}
```

## Validation Locally

1. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Run the server:**
   ```bash
   python -m app.main
   # Or
   uvicorn app.main:app --reload
   ```

3. **Test the API:**
   ```bash
   # Health check
   curl http://localhost:8000/health

   # Moderate a comment
   curl -X POST http://localhost:8000/moderate \
     -H "Content-Type: application/json" \
     -d '{"text": "This is a test comment"}'
   ```

4. **Verify model loading:**
   - Check logs on startup - should show "Model loaded successfully"
   - Ensure no internet connection is required (model loads from `final_model/`)

5. **Sanity tests:**
   - Test with clearly toxic text: "You are an idiot"
   - Test with clearly non-toxic text: "Have a great day!"
   - Verify probabilities and labels make sense

## Key Implementation Details

- **Model Loading**: Model and tokenizer loaded once at startup as singletons
- **Threshold**: Read from `threshold.json` at startup
- **Max Tokens**: 192 (matches training configuration)
- **Preprocessing**: Truncation and padding to 192 tokens
- **Device**: Automatically uses GPU if available, otherwise CPU
- **Logging**: Logs timestamp, input length, probability, and label (not raw text)

## Security Considerations

- Request size limits (max 10000 characters per text)
- Input sanitization (strip whitespace, validate)
- CORS enabled for frontend access
- Error handling to prevent information leakage

