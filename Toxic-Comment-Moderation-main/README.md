# Toxic Comment Moderation - Full Stack Application

A full-stack web application for moderating toxic comments using a DistilBERT model. The backend provides a FastAPI REST API, and the frontend is a React + Vite application with a modern UI.

## Project Structure

```
Htianshu/
├── final_model/              # Model artifacts (organized)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
├── backend/                  # FastAPI backend
│   ├── app/
│   │   ├── main.py          # FastAPI application
│   │   ├── model_loader.py  # Model/tokenizer loading
│   │   └── schemas.py       # Pydantic models
│   ├── requirements.txt
│   └── README_backend.md
├── frontend/                 # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── components/
│   │   └── styles/
│   ├── package.json
│   └── vite.config.js
├── threshold.json           # Decision threshold
├── final_metrics.json       # Test metrics
└── README.md                # This file
```

## Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI server:
   ```bash
   python -m app.main
   # Or with uvicorn directly:
   uvicorn app.main:app --reload
   ```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:5173`

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
  "texts": ["Text 1", "Text 2"]
}
```

### GET /metrics

Get test metrics from `final_metrics.json`.

### GET /config

Get current threshold configuration.

See `backend/README_backend.md` for detailed API documentation.

## Features

- **Backend:**
  - FastAPI REST API
  - Model loaded from local `final_model/` directory (no internet required)
  - Threshold-based classification
  - Batch processing support
  - CORS enabled for frontend access
  - Comprehensive error handling and logging

- **Frontend:**
  - Modern React UI with Vite
  - Real-time moderation results
  - Example buttons for quick testing
  - Color-coded results (green for non-toxic, red for toxic)
  - Loading states and error handling
  - Responsive design

## Model Details

- **Model Type**: DistilBERT for Sequence Classification
- **Max Tokens**: 192 (truncation/padding)
- **Threshold**: 0.9249592423439026
- **Test Metrics**:
  - Accuracy: 0.9558
  - Precision: 0.8533
  - Recall: 0.6198
  - F1: 0.7181
  - AUC: 0.9723

## Validation Checklist

Before deploying, verify:

- [x] Model loads from `final_model/` without internet
- [x] Threshold matches `threshold.json` value
- [x] Preprocessing: 192 token truncation/padding
- [x] CORS enabled for frontend
- [x] Error handling and logging in place

## Development

### Backend Development

The backend uses FastAPI with automatic API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Frontend Development

The frontend uses Vite for fast development with hot module replacement.

## Deployment

### Backend Deployment

For production, use Gunicorn with Uvicorn workers:

```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend Deployment

Build the frontend for production:

```bash
cd frontend
npm run build
```

The built files will be in `frontend/dist/` directory.

### Docker Deployment

See deployment documentation for containerization details. Ensure `final_model/` and `threshold.json` are mounted or copied into the container.

## License

[Add your license here]

