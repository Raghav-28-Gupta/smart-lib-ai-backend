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
- **Deployment (not done yet, an owner step)**: a second web service next to the Node backend, installed from `requirements.txt` exactly (numpy 2.5.2, scikit-learn 1.9.0, xgboost 3.4.1, joblib 1.5.3 among the pins: the artifact is a pickle and answers 503 `model_unavailable` on other versions), then the backend's `AI_BACKEND_URL` set to its address. Until then production priority stays 1.0. The recommender routes are protected by the shared secret `AI_TOKEN` (set the same value on both services, in the host's environment settings: it is read from the PROCESS environment only, an `ai-backend/.env` file does not set it); `/predict/no-show` needs none. A hosted web service is publicly reachable and there is no request-body size limit (see section 2): do not share its address, and add a platform limit or a guard that counts the streamed bytes when deploying publicly (a `Content-Length` check does not bound a chunked body). Use Python 3.13 (`PYTHON_VERSION=3.13.1`) and `uvicorn app.main:app --host 0.0.0.0 --port $PORT` (see README Deploy).

### 2. Hybrid Book Recommendations (`app/ml/recommend/` & `app/routers/recommend.py`)
- **Stateless batch endpoint**: `POST /recommend/batch` takes the whole catalog and every student's interactions and answers the top picks per student. Nothing is trained ahead of time, stored or saved: every call fits TF-IDF and ALS from its own payload with fixed seeds, so the same payload gives the same answer (on one host with one thread configuration; scores can differ in the last bits across chunk sizes, BLAS thread counts and hosts; no model artifact, no metrics file). It replaces the old `POST /recommend/books` stub (404 now). Never read a database here.
- **Contract** (`app/schemas/recommend.py`, extra fields forbidden, 422 never echoes a value of the input: an unknown key's NAME can appear in the error's `loc`): `books` 1 to 5,000 `{id, title, author, genre, description}` (duplicate ids are a 422), `interactions` 0 to 200,000 `{user_id, book_id, weight}` with `0 < weight <= 3`, `limit` 1 to 50 (default 10). An interaction on an unknown book is ignored; a repeated (student, book) pair is summed and capped at 3. Answer: `{model_version, users: [{user_id, results: [{book_id, score, method}]}]}`, users in id order, only students with at least one result, `score` in (0, 1], `method` `content` | `collaborative` | `hybrid`, never a book the student already has, score 0 dropped, ties by book id ascending. `model_version` is `tfidf-als-1` when the collaborative part ran, else `tfidf-1`. The committed `tests/fixtures/recommend_batch_example.json` (request + answer) is shared with the backend (`backend/test/fixtures/recommend-batch-example.json` is a byte-identical copy: change both together and `cmp`); it stays below the ALS minimums.
- **Content-Based (TF-IDF)** (`content.py`): one text per book (title, author, genre twice, description), `TfidfVectorizer(stop_words="english", sublinear_tf=True)`; a student's profile is the weight-summed vectors of their books, unit length; the score is the cosine. A catalog whose texts are all empty or all stop words gives score 0 everywhere, never an error.
- **Collaborative Filtering (implicit ALS)** (`collaborative.py`): only with at least 5 students and 20 interactions (distinct student-book pairs): `AlternatingLeastSquares(factors=16, regularization=0.1, iterations=15, random_state=42, use_gpu=False)` on the students-by-books matrix (implicit 0.7.3 expects users by items) times the one fixed constant `ALS_CONFIDENCE = 40`, inside `threadpool_limits(limits=1, user_api="blas")` (the model is built inside it too: implicit checks BLAS there). `implicit` is imported inside `fit_als`, so a host without it, or a fit that raises, degrades to content only (`tfidf-1`; the failure is logged by exception type, never by value). The factors are kept as float64 so that a score depends far less on the chunk size: scores for two chunk sizes differ by about 1e-16 (float32 gave 2e-7 to 2e-6 in two probes), and ids, order and methods were identical in every probe. Two students with identical histories can score up to about 3e-8 apart (implicit's random per-student initialisation), so a near-tie between them could order differently. `als_scores` returns the whole students-by-books matrix and exists for the tests; the engine uses `fit_als` and scores one chunk at a time.
- **Dynamic Blending** (`engine.py`): per student each part is divided by its maximum over the candidates (books the student does not have); with `n` the student's distinct books, `w = collaborative_weight(n) = min(0.7, 0.1 * n)` and the score is `(1 - w) * content + w * collaborative`; `hybrid` when both parts are above 0, `collaborative` when only that one is, `content` otherwise. Students are processed `USER_CHUNK` (512) at a time so the dense score blocks stay bounded; any change inside that loop needs `test_chunking_does_not_change_the_answer` (it catches a student served another student's picks past a chunk boundary).
- **Shared secret** (`app/core/security.py`): every `/recommend/*` route needs the header `x-ai-token`, compared with `hmac.compare_digest` against env `AI_TOKEN`, read from the PROCESS environment on every request (not through `Settings`, so `ai-backend/.env` does not set it). Unset or blank: 404 with the body of an unknown route (the recommender is off). Missing or wrong: 401. The token never appears in a log, a response or an error: never format it. Edges, all accepted and documented in the README Deploy section: a malformed JSON body answers 422 before the token is checked (FastAPI parses the body first), there is no request-body size limit (a body of about 23 MB at the caps is parsed before the token is checked, as on `/predict/no-show`; do not share the address; add a platform limit or a guard that counts the streamed bytes when deploying publicly, since a `Content-Length` check does not bound a `Transfer-Encoding: chunked` body, which uvicorn accepts; whether Render has a configurable request-size limit is unverified), and while `AI_TOKEN` is unset `/openapi.json` and `/docs` still list `/recommend/batch` and a `GET` of it answers 405, not 404 (a `POST` answers 404).
- **Size and speed** (measured on one Windows laptop, over real HTTP with uvicorn on 127.0.0.1, a fresh server process per run, three runs per row, random payload of 5,000 books and 8,000 students with 36-character UUID ids as the backend sends, compact JSON; the peak is the OS's peak working set of the server process; the service alone is 166 MB): 50,000 interactions (the backend's default send cap) = 7.2 MB body, peak 395 to 396 MB, 7.8 to 8.0 s, 7.8 MB answer; 100,000 = 12.5 MB, 434 to 444 MB, 9.5 to 10.1 s; 200,000 (the contract's cap) = 23.1 MB, 510 to 530 MB, 11.3 to 11.4 s (MB is 2^20 bytes; full table in the README). **A free 512 MB instance is not enough at the 200,000 cap.** The old figures (13.9 MB body, 5.5 MB answer, about 470 MB) used short ids and were too low. Linux and Render were NOT measured, nor a 0.1-CPU host. The ALS fit alone is about 0.8 s and about 20 to 40 MB of the peak (earlier short-id, in-process figures, not re-run). The answer grows with the number of STUDENTS (40,000 students with 200,000 interactions: about 27 s, also from the earlier run; 200,000 distinct students was not run and extrapolates past a gigabyte and past the backend's 90 s timeout). The seeded demo is a tiny fraction of the caps; a live campus grows towards them, because the backend sends every loan ever made. An out-of-memory restart fails open but also interrupts `/predict/no-show` on a shared instance (its bookings stay unscored, priority 1.0); students keep their cached picks for up to 14 days, then get the popularity list.
- **Tests**: `test_recommend_content.py` (hand-made catalogs), `test_recommend_collaborative.py` (planted structure: 40 students in two groups of 20 whose books share no word, so only ALS can find an unseen same-group book; the blend, including that its weight counts distinct books, not total weight; chunking; a missing or failing `implicit`), `test_recommend_api.py` (contract and token). The planted results are for `random_state=42` (the planted pick leads the runner-up by at least 0.25 there); under other seeds some students miss their book (seeds 0 to 39: 28 to 40 of 40 find it), so the all-students test asserts that at least half (20 of 40) find their book and the single-student tests carry `seed_42` in their names. On an `implicit` upgrade or an ALS parameter change, re-check them by hand, do not re-seed. The simulated data proves the plumbing, not real-world quality: say so wherever the recommender's behaviour is described.
- **How the backend calls it**: from its 60-second timer, in the background, with the catalog and all students' interactions, a 90 s timeout and the `x-ai-token` header; it validates the answer, replaces its `RecommendationCache` in one transaction, and serves `GET /recommendations/me` from that cache only. It fails open: while this service is down students keep the cached picks of the last good refresh for up to 14 days (`RECOMMEND_MAX_AGE_MS`); the popularity list shows only for students without cached picks or after 14 days. No name or email is sent (book titles, authors, genres and descriptions are, because the content part needs them); no name, email or title is logged: ids, counts and the model version only.

---

## Code Organization

```
ai-backend/
├── app/
│   ├── main.py              # FastAPI app instantiation & router registration
│   ├── core/
│   │   ├── config.py        # Pydantic BaseSettings (app name, port, env)
│   │   └── security.py      # require_ai_token: the x-ai-token check (404 off, 401 wrong)
│   ├── routers/
│   │   ├── health.py        # GET /health
│   │   ├── noshow.py        # POST /predict/no-show (loads the committed model once)
│   │   └── recommend.py     # POST /recommend/batch (needs x-ai-token)
│   ├── schemas/
│   │   ├── noshow.py        # Pydantic input/output models for no-show predictions
│   │   └── recommend.py     # Pydantic models for the batch recommender (BatchRequest, BatchResponse)
│   └── ml/
│       ├── noshow/          # features.py (order, encoding), history.py, simulate.py, train.py, model.py
│       ├── artifacts/       # COMMITTED: noshow-<yyyymmdd>-<hash8>.joblib + metrics.json
│       └── recommend/       # content.py (TF-IDF), collaborative.py (implicit ALS), engine.py (blend, top picks)
├── .gitattributes           # fixture and metrics.json stay LF, the .joblib artifact is binary
├── tests/                   # pytest (contract, history fixture, simulator, training, serving, recommender)
├── requirements.txt         # Pinned Python package dependencies
├── requirements-dev.txt     # + pytest, httpx
└── .venv/                   # Local virtual environment (ignored)
```

---

## Sibling Access
When started from `ai-backend/`, Claude has access to `../backend` to reference the Node.js backend schemas, Prisma data models, and API client request formats.
