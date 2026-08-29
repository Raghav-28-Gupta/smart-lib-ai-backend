# SmartLib — AI Backend

The AI Backend hosts the **XGBoost no-show prediction model** and the **hybrid book recommendation engine**. It is an **internal microservice** called strictly by the Node.js Core Backend (`src/lib/aiClient.ts`), never directly by the Flutter client.

## Tech Stack
- **Language**: Python 3.11+ (developed on 3.13)
- **Framework**: FastAPI + Uvicorn + Pydantic v2 + Pydantic Settings
- **Machine Learning**: XGBoost, scikit-learn, implicit, pandas, numpy, scipy
- **Model Serialization**: joblib

---

## Setup & Running

Always run these commands from the `ai-backend/` directory:

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start development server on port 8000
uvicorn app.main:app --reload --port 8000
```

> **Windows Tip**: If `uvicorn` is not recognized globally, call it via the venv python executable:
> `.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000`

- **Interactive API Docs**: `http://localhost:8000/docs`
- **Health Check**: `curl http://localhost:8000/health`

---

## Machine Learning Responsibilities

### 1. No-Show Prediction (`app/ml/noshow/` & `app/routers/noshow.py`)
- Predicts the likelihood that a student will fail to attend a physical study seat or room reservation.
- **Fairness-Bounded Priority**: The raw probability from the XGBoost classifier is converted to a bounded penalty factor to prevent irreversible deprioritization.
- **Synthetic Simulation**: Uses synthetic booking distributions to bootstrap the model prior to real production history.

### 2. Hybrid Book Recommendations (`app/ml/recommend/` & `app/routers/recommend.py`)
- **Content-Based (TF-IDF)**: Computes book vector similarities based on title, author, genre, and description for cold-start users and new catalog additions.
- **Collaborative Filtering (Implicit ALS)**: Leverages implicit feedback (borrows, renewals, views) as user interaction history accumulates.
- **Dynamic Blending**: Automatically shifts weighting from content-based to collaborative filtering as user activity increases.

---

## Code Organization

```
ai-backend/
├── app/
│   ├── main.py              # FastAPI app instantiation & router registration
│   ├── core/
│   │   └── config.py        # Pydantic BaseSettings (app name, port, env)
│   ├── routers/
│   │   ├── health.py        # GET /health
│   │   ├── noshow.py        # POST /predict/no-show
│   │   └── recommend.py     # POST /recommend/books
│   ├── schemas/
│   │   ├── noshow.py        # Pydantic input/output models for no-show predictions
│   │   └── recommend.py     # Pydantic input/output models for book recommendations
│   └── ml/
│       ├── noshow/          # XGBoost model training, inference, and feature engineering
│       └── recommend/       # TF-IDF and implicit ALS recommendation engines
├── requirements.txt         # Pinned Python package dependencies
└── .venv/                   # Local virtual environment (ignored)
```

---

## Sibling Access
When started from `ai-backend/`, Claude has access to `../backend` to reference the Node.js backend schemas, Prisma data models, and API client request formats.
