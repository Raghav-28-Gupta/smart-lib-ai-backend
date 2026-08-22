# SmartLib — AI Backend (Python + FastAPI + uvicorn)

Hosts the no-show classifier and the book recommender. It exists as a separate
service because XGBoost, scikit-learn, and `implicit` are Python-native and
don't belong forced into the TypeScript runtime.

**This service is internal.** Only the Node backend calls it — never the client.

## Prerequisites

- Python 3.11+ (developed on 3.13)

## Setup

```bash
cd ai-backend
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

If `uvicorn` isn't on your PATH after activating, use the venv directly:
`.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000`

Interactive API docs: <http://localhost:8000/docs>

## Endpoints (Phase 1)

| Method | Path | Status |
|---|---|---|
| GET | `/health` | Real |
| POST | `/predict/no-show` | Stub — always returns `0.0` |
| POST | `/recommend/books` | Stub — always returns `[]` |

The stubs return the real response shapes so the Node backend can be wired
against them before Phase 3 trains the actual models.

```bash
curl localhost:8000/health
# {"status":"ok","service":"ai-backend"}

curl -X POST localhost:8000/predict/no-show \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","resource_booking_id":"b1"}'
# {"predicted_probability":0.0,"model_version":"stub"}

curl -X POST localhost:8000/recommend/books \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","limit":5}'
# {"user_id":"u1","results":[],"model_version":"stub"}
```

`model_version` is `"stub"` today; Phase 3 replaces it with a real identifier so
predictions can be traced back to the model that produced them.

## Layout

```
app/
├── main.py             # FastAPI app, router registration
├── core/config.py      # pydantic-settings configuration
├── routers/
│   ├── health.py
│   ├── noshow.py       # stub
│   └── recommend.py    # stub
├── schemas/            # request/response models
└── ml/
    ├── noshow/         # XGBoost classifier — Phase 3
    └── recommend/      # hybrid recommender — Phase 3
```

## Note on requirements.txt

It's a full `pip freeze`, so it pins transitive dependencies too. The direct
dependencies are: `fastapi`, `uvicorn[standard]`, `pydantic`,
`pydantic-settings`, `xgboost`, `scikit-learn`, `pandas`, `implicit`. The ML
libraries are installed now but unused until Phase 3.
