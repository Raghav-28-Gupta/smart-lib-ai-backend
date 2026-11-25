# SmartLib — AI Backend (Python + FastAPI + uvicorn)

Hosts the no-show classifier and the book recommender. It exists as a separate
service because XGBoost, scikit-learn, and `implicit` are Python-native and
don't belong forced into the TypeScript runtime.

**This service is internal.** Only the Node backend calls it — never the client.

## Prerequisites

- Python 3.13 (developed and trained on 3.13.1; 3.11 is unchecked against the pinned numpy and pandas)

## Setup

```bash
cd ai-backend
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
# to run the tests as well (adds pytest and httpx):
pip install -r requirements-dev.txt
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

If `uvicorn` isn't on your PATH after activating, use the venv directly:
`.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000`

Interactive API docs: <http://localhost:8000/docs>

## Test

```bash
python -m pytest
```

No network and no database. The suite trains a few small models in a temporary
directory (a few seconds) and also checks the committed model artifact.

## Endpoints

| Method | Path | Status |
|---|---|---|
| GET | `/health` | Real |
| POST | `/predict/no-show` | Real — the trained XGBoost model |
| POST | `/recommend/books` | Stub — always returns `[]` |

```bash
curl localhost:8000/health
# {"status":"ok","service":"ai-backend"}

curl -X POST localhost:8000/predict/no-show \
  -H 'Content-Type: application/json' \
  -d '{"hour_of_day":10,"day_of_week":2,"duration_minutes":120,"lead_time_hours":24.0,
       "resource_type":"room","past_attendance_ratio":0.9,"consecutive_attended_count":5,
       "recent_noshows_count":0,"reliability_score":80,"bookings_last_30d":4}'
# {"predicted_probability":0.21825159820088116,"penalty_factor":0.10912579910044058,"model_version":"noshow-20261124-e60bbd87"}
# (model_version is whichever artifact is committed under app/ml/artifacts/)

curl -X POST localhost:8000/recommend/books \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","limit":5}'
# {"user_id":"u1","results":[],"model_version":"stub"}
```

### `POST /predict/no-show`

The service is stateless: the Node backend works the ten features out from its
database and sends them; this service never reads one.

| Field | Range |
|---|---|
| `hour_of_day` | 0-23 (library time zone) |
| `day_of_week` | 0-6, Monday = 0 |
| `duration_minutes` | 1-1440 |
| `lead_time_hours` | >= 0 |
| `resource_type` | `"seat"` or `"room"` |
| `past_attendance_ratio` | 0-1 |
| `consecutive_attended_count` | integer 0-1,000,000 |
| `recent_noshows_count` | integer 0-1,000,000 |
| `reliability_score` | 0-100 |
| `bookings_last_30d` | integer 0-1,000,000 |

Answers `200 {"predicted_probability": 0-1, "penalty_factor": 0-0.2, "model_version": "..."}`.

- `penalty_factor = min(max(probability * 0.5, 0), 0.2)`: the fairness bound. No
  booking is ever penalized by more than 0.2, whatever the probability.
- `422` for a missing, out-of-range or unknown field (the response never echoes
  the input back).
- `503 {"detail": "model_unavailable"}` when no model artifact could be used: none
  committed, a corrupt or altered file, or a model that unpickles but fails the one
  test prediction made when it is loaded at startup. The Node backend treats that as
  "no nudge".
- `model_version` is the artifact's file name without `.joblib`
  (`noshow-<yyyymmdd>-<hash8>`) and is in every response, so a prediction can be
  traced to the model that made it. The recommender stub still answers `"stub"`.

How the Node backend uses it (`../backend/README.md`, *No-shows*): after a booking
commits it calls this endpoint in the background with a 2-second timeout and never
waits for it, so a 503, a timeout or a garbled answer simply leaves the booking
unscored (priority 1.0): the backend **fails open**. It makes no call at all while
its `AI_BACKEND_URL` is unset or blank (production today), and it leaves a booking
longer than 24 hours unscored, because this service refuses a `duration_minutes`
over 1440. The penalty becomes a nudge of at most 72 minutes in the librarian's
list of room requests (0.2 x a 6-hour window).

## The no-show model

The model is trained on **simulated** bookings (`app/ml/noshow/simulate.py`, seeded
with `numpy.random.default_rng`), because there is no production history yet. Each
booking's features come from `features_from_history` (`history.py`), the same
definitions the Node backend implements; `tests/fixtures/noshow_features.json` is
the shared contract.

```bash
python -m app.ml.noshow.train --seed 42            # about 5 seconds
python -m app.ml.noshow.train --seed 42 --students 2000 --out app/ml/artifacts   # the same, spelled out
```

It simulates 2000 students, splits them **by student** 60 / 20 / 20 into train,
calibration and test sets (a student is never in two), fits
`XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, eval_metric="logloss",
random_state=seed, tree_method="hist", n_jobs=1)` on the train set, calibrates the
probabilities on the calibration set, and scores the test set. It writes:

- `app/ml/artifacts/noshow-<yyyymmdd>-<hash8>.joblib` — the calibrated model, the
  feature order and the encoding. `hash8` is the first 8 hex characters of the
  sha256 of the file's bytes; the date is the training day.
- `app/ml/artifacts/metrics.json` — ROC AUC, Brier score (and the base-rate
  predictor's), log loss, ten calibration bins, the mean absolute calibration gap,
  feature importances, the seed, row counts and library versions. It names the
  artifact it describes and records no timestamp, so the same seed gives the same
  figures.

The committed `metrics.json` (seed 42, 30,535 simulated bookings, 6,243 of them in
the test split, base no-show rate about 15 percent) reports on the held-out test
students: ROC AUC 0.727, Brier score 0.1137 (the base-rate predictor's is 0.1273),
log loss 0.376 and a mean calibration gap of 0.011. These describe the simulator,
not real students. It also holds the chosen calibration method (sigmoid), ten
equal-count calibration bins, the feature importances and the library versions.

**Both files are committed**, so a deploy needs no training. Retrain only when the
features or the simulator change, then commit the new pair (the old artifact can
be deleted; the service loads the newest by the date in the file name, then by
file time, then by name). Models are only ever loaded from `app/ml/artifacts/`:
the artifact is a pickle, and no path ever comes from a request. The bytes are
checked against the hash in the file name before they are unpickled.

Model notes:

- **Seeds are fixed** (`--seed 42` by default, in `metrics.json`): one seed, one
  dataset, one model.
- **Calibration** is isotonic or sigmoid, chosen by the out-of-fold Brier score on
  the calibration set, but a method that outputs exactly 0 or 1 is not eligible
  (isotonic's flat ends do). `metrics.json` says which was chosen. The sigmoid is
  fitted to the classifier's log-odds, not to its probability: scikit-learn uses a
  classifier's `decision_function` when it has one and the probability itself when it
  does not, and `XGBClassifier` has none, so the classifier is wrapped in
  `LogOddsClassifier` (`model.py`, an app module so the artifact can unpickle it),
  which offers XGBoost's raw margin. Fitted to the probability instead, the sigmoid
  could not output below about 0.06 and lifted the whole low end. On this simulated
  data the raw XGBoost output is already close to calibrated, so calibration changes
  little (`metrics.json` has both sets of figures).
- **Calibration bins are equal-count** (rows sorted by predicted probability, cut
  into ten runs of about 600), so every bin is well populated.
- **The simulated `reliability_score` leads the live one.** The simulator applies
  the -10 for a no-show at the booking's start; the live system applies it after
  the grace period and the sweep's lag. The training signal therefore slightly
  leads the serving signal.
- **`recent_noshows_count` has low importance (0.038).** Recovery after a no-show
  shows mainly through the attendance ratio (0.385) and the reliability score
  (0.148), which move as the student attends again.
- The labels are drawn from a logistic of a student's latent reliability, the lead
  time, a late hour and room versus seat, so the model learns the simulator, not
  real students. Retraining on real rows (the backend logs them) is future work.

## Deploy

Not deployed yet: the backend's production `AI_BACKEND_URL` is unset, so no
booking is scored and every priority stays 1.0. To deploy, run this as a **second
web service** next to the Node backend and set the backend's `AI_BACKEND_URL` to
its address. The service has no authentication of its own, so keep its address to
the backend. Install exactly `requirements.txt`, which pins numpy 2.5.2, scikit-learn 1.9.0,
xgboost 3.4.1 and joblib 1.5.3: the committed artifact is a pickle and fails to
load on other versions, and the service then answers 503 `model_unavailable` (the
backend treats that as "no nudge"). Loading also prints a joblib/NumPy
deprecation warning every time with these versions (harmless now; unpickling
breaks once NumPy removes that API, so retrain then).

What a hosted service needs (checked against the files, not yet tried on Render):

- **Build command** `pip install -r requirements.txt`; **start command**
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT` (the `--reload --port 8000`
  line above binds 127.0.0.1 and is for local use only).
- **Python 3.13.** The artifact and every pin come from Python 3.13.1 (it is in
  `metrics.json`), so set `PYTHON_VERSION=3.13.1` rather than rely on the host's
  default. Whether numpy 2.5.2 and pandas 3.0.5 install on 3.11 has not been checked.
- **`implicit==0.7.3`** is in `requirements.txt` for the future recommender and
  unused today; it is the pin most likely to need a compiler on Linux. If the
  first build fails there, drop it until the recommender slice (the backend
  keeps working at priority 1.0).
- **Cold start.** A sleeping free instance takes tens of seconds to import and
  load the model, so the first booking after idle hits the backend's 2-second
  timeout and stays unscored. That is the designed fail-open path, not a fault.
- **"Internal only" is a convention, not enforced:** a hosted web service is
  publicly reachable. `POST /predict/no-show` is a pure function of ten numbers
  (no database, no user data), so that is harmless today, but do not share the
  address. Before the recommender returns real output (`/recommend/books` takes a
  `user_id`), add a shared secret: the backend sends an `x-ai-token` header from an
  environment variable and one FastAPI dependency compares it in constant time
  (answering 404 when the variable is unset, as `SWEEP_TOKEN` does), or use a
  private service if the host offers one.

## Layout

```
app/
├── main.py             # FastAPI app, router registration, loads the model at startup
├── core/config.py      # pydantic-settings configuration
├── routers/
│   ├── health.py
│   ├── noshow.py       # POST /predict/no-show; loads the committed model once
│   └── recommend.py    # stub
├── schemas/            # request/response models
└── ml/
    ├── noshow/         # features.py (order, encoding), history.py, simulate.py,
    │                   # train.py, model.py (load, predict, penalty_factor)
    ├── artifacts/      # the committed model and metrics.json
    └── recommend/      # hybrid recommender — Phase 3
tests/                  # pytest: contract, history features, simulator, training, serving
                        # fixtures/noshow_features.json: the feature contract shared with the
                        # backend (a byte-identical copy lives at backend/test/fixtures/
                        # noshow-features.json; change both together, check with cmp)
.gitattributes          # keeps the fixture and metrics.json LF and the artifact binary
```

## Note on requirements.txt

It's a full `pip freeze`, so it pins transitive dependencies too. The direct
dependencies are: `fastapi`, `uvicorn[standard]`, `pydantic`,
`pydantic-settings`, `xgboost`, `scikit-learn`, `pandas`, `implicit`. The no-show
model uses XGBoost, scikit-learn, pandas, numpy and joblib; `implicit` is unused
until the recommender. `requirements-dev.txt` adds `pytest` and `httpx` on top.
