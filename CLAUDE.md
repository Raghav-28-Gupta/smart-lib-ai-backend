# SmartLib — AI Backend

The AI Backend hosts the **XGBoost no-show prediction model** and the **hybrid book recommendation engine**. It is an **internal microservice** called strictly by the Node.js Core Backend (`src/lib/aiClient.ts`), never directly by the Flutter client.

## Tech Stack
- **Language**: Python 3.13 (developed and trained on 3.13.1; 3.11 is unchecked against the pinned numpy and pandas)
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

# Install dependencies (requirements-dev.txt adds pytest and httpx for the tests)
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run the tests (no network, no database)
python -m pytest

# Retrain and rewrite the committed no-show artifact + metrics.json (fixed seed, about 5 seconds)
python -m app.ml.noshow.train --seed 42

# Start development server on port 8000
uvicorn app.main:app --reload --port 8000
```

> **Windows Tip**: If `uvicorn` is not recognized globally, call it via the venv python executable:
> `.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000`
> (likewise `.\.venv\Scripts\python.exe -m pytest`; never use the global Python).

- **Interactive API Docs**: `http://localhost:8000/docs`
- **Health Check**: `curl http://localhost:8000/health`

---

## Machine Learning Responsibilities

### 1. No-Show Prediction (`app/ml/noshow/` & `app/routers/noshow.py`)
- Predicts the likelihood that a student will fail to attend a physical study seat or room reservation.
- **Stateless endpoint**: `POST /predict/no-show` takes the ten features the Node backend computes (`FEATURE_ORDER` in `features.py`: `hour_of_day`, `day_of_week`, `duration_minutes`, `lead_time_hours`, `resource_type` (`seat`/`room`), `past_attendance_ratio`, `consecutive_attended_count`, `recent_noshows_count`, `reliability_score`, `bookings_last_30d`) and answers `{predicted_probability, penalty_factor, model_version}`. 422 on invalid input, 503 `model_unavailable` when no artifact loads. Never read a database here.
- **Fairness-Bounded Priority**: `penalty_factor(p, alpha=0.5, cap=0.2) = min(max(p * alpha, 0), cap)` (`model.py`; `MAX_PENALTY_CAP` is in `features.py`). Never exceed the cap; the Node side clamps again.
- **Synthetic Simulation**: `simulate(n_students, seed)` (`simulate.py`) bootstraps the model prior to real production history. Features are built by `features_from_history` (`history.py`), shared with the Node backend through `tests/fixtures/noshow_features.json`; change the definitions in both repos and the fixture together. The backend's `test/fixtures/noshow-features.json` is a byte-identical copy: re-copy it and `cmp` the two after any change (`.gitattributes` keeps both LF). The model is trained on SIMULATED bookings only: say so wherever its numbers appear.
- **One encoder**: `encode_features` (`features.py`) turns raw features into the model's float matrix (`seat` = 0, `room` = 1). Training and serving both call it; never encode a feature anywhere else. A saved model records the feature order and encoding and refuses to load if either changes.
- **Training**: `python -m app.ml.noshow.train --seed 42` (`train.py`): simulate, split BY STUDENT 60/20/20 (train / calibration / test), `XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=seed, tree_method="hist", n_jobs=1)`, calibrate (isotonic or sigmoid by out-of-fold Brier; a method that outputs exactly 0 or 1 is not eligible; always calibrate the `LogOddsClassifier` wrapper in `model.py`, never the bare `XGBClassifier`: it has no `decision_function`, so scikit-learn would fit the sigmoid to the probability and floor it at about 0.06), evaluate on the test split. Keep seeds fixed; `metrics.json` holds no timestamp so one seed gives the same file.
- **Artifacts are committed**: `app/ml/artifacts/noshow-<yyyymmdd>-<hash8>.joblib` (hash8 = first 8 hex of the sha256 of its bytes) and `metrics.json` (it names the artifact). A deploy needs no training. After retraining, commit both files. `model_version` (the file name without `.joblib`) is in every response. The service loads the newest artifact once at startup (latest date in the name, then file time, then name) and answers 503 if none loads.
- **Never load a model from a request-supplied path**: joblib unpickles. Only `app/ml/artifacts/` is read, and the bytes must match the hash in the file name.
- **Model note**: the simulated `reliability_score` applies the -10 at the booking's start, while the live system applies it after the grace period and the sweep's lag, so the training signal slightly leads the serving signal.
- **`metrics.json`** holds the seed, row counts, ROC AUC, Brier score (and the base-rate predictor's), log loss, the chosen calibration method, ten equal-count calibration bins, the feature importances and the library versions; it names its artifact and records no timestamp, so the same seed gives the same file (only the artifact name and `model_version` change with the training date). The test numbers (AUC about 0.73, Brier 0.114 against 0.127) describe the simulator, not real students.
- **How the backend calls it**: a 2-second timeout, in the background, never awaited; it fails open (a 503, a timeout or garbage leaves a booking unscored, priority 1.0). It makes no call while its `AI_BACKEND_URL` is unset or blank. A booking over 24 hours stays unscored because this service refuses `duration_minutes` over 1440 (422). Keep the field list in `app/schemas/noshow.py` (it forbids extra fields) and the backend's `aiClient.ts` `NoShowFeatures` in step.
- **Deployment (not done yet, an owner step)**: a second web service next to the Node backend, installed from `requirements.txt` exactly (numpy 2.5.2, scikit-learn 1.9.0, xgboost 3.4.1, joblib 1.5.3 among the pins: the artifact is a pickle and answers 503 `model_unavailable` on other versions), then the backend's `AI_BACKEND_URL` set to its address. Until then production priority stays 1.0. The service has no authentication of its own and a hosted web service is publicly reachable: do not share its address, and add a shared secret before the recommender returns real output. Use Python 3.13 (`PYTHON_VERSION=3.13.1`) and `uvicorn app.main:app --host 0.0.0.0 --port $PORT` (see README Deploy).

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
│   │   ├── noshow.py        # POST /predict/no-show (loads the committed model once)
│   │   └── recommend.py     # POST /recommend/books
│   ├── schemas/
│   │   ├── noshow.py        # Pydantic input/output models for no-show predictions
│   │   └── recommend.py     # Pydantic input/output models for book recommendations
│   └── ml/
│       ├── noshow/          # features.py (order, encoding), history.py, simulate.py, train.py, model.py
│       ├── artifacts/       # COMMITTED: noshow-<yyyymmdd>-<hash8>.joblib + metrics.json
│       └── recommend/       # TF-IDF and implicit ALS recommendation engines
├── .gitattributes           # fixture and metrics.json stay LF, the .joblib artifact is binary
├── tests/                   # pytest (contract, history fixture, simulator, training, serving)
├── requirements.txt         # Pinned Python package dependencies
├── requirements-dev.txt     # + pytest, httpx
└── .venv/                   # Local virtual environment (ignored)
```

---

## Sibling Access
When started from `ai-backend/`, Claude has access to `../backend` to reference the Node.js backend schemas, Prisma data models, and API client request formats.
