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
| POST | `/recommend/batch` | Real — TF-IDF content picks blended with implicit ALS; needs the `x-ai-token` header |

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

# the recommender answers 404 until AI_TOKEN is set in the server's environment, and 401 without the header
curl -X POST 127.0.0.1:8000/recommend/batch \
  -H 'Content-Type: application/json' -H 'x-ai-token: <the value of AI_TOKEN>' \
  -d '{"books":[{"id":"a1","title":"Introduction to Algorithms","author":"Rivera","genre":"Algorithms",
       "description":"Sorting, graphs and dynamic programming."},
      {"id":"a2","title":"Data Structures and Algorithms in Java","author":"Okafor","genre":"Data Structures",
       "description":"Trees, graphs, hash tables and algorithms."},
      {"id":"n1","title":"Computer Networks","author":"Lindqvist","genre":null,"description":null}],
      "interactions":[{"user_id":"u1","book_id":"a1","weight":1}],"limit":5}'
# {"model_version":"tfidf-1","users":[{"user_id":"u1","results":[{"book_id":"a2","score":1.0,"method":"content"}]}]}
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
  a value of the input back; the name of an unknown field can appear in the error's `loc`).
- `503 {"detail": "model_unavailable"}` when no model artifact could be used: none
  committed, a corrupt or altered file, or a model that unpickles but fails the one
  test prediction made when it is loaded at startup. The Node backend treats that as
  "no nudge".
- `model_version` is the artifact's file name without `.joblib`
  (`noshow-<yyyymmdd>-<hash8>`) and is in every response, so a prediction can be
  traced to the model that made it.

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

## Recommender

`POST /recommend/batch` answers "which books next?" for every student in one call. The service is
**stateless and keeps no model artifact**: the Node backend sends the whole catalog and every student's
interactions, and each call fits TF-IDF and ALS from that payload alone, with fixed seeds, so the same
payload gives the same answer (on one host with one thread configuration; across block sizes or BLAS
thread counts scores differ by about 1e-16, and ids, order and methods did not change in any probe). The
backend calls it from a background timer, stores the answer in its
`RecommendationCache` table and serves students from that cache, so a student's screen never waits for
this service; it fails open when this service is asleep, down or not deployed: students keep the cached
picks of the last good refresh for up to 14 days, and the popularity list shows only for students
without cached picks or once the cache is older than that.
Nothing here is tuned or retrained, and the tests prove the plumbing and a planted structure on simulated
data, not real-world quality.

**The request** (422 on anything invalid, the response never echoes a value of the input back (the name of
an unknown key can appear in the error's `loc`), extra fields are refused):

| Field | Rule |
|---|---|
| `books` | 1 to 5,000 of `{id, title, author, genre, description}` (`genre` and `description` may be `null`, the texts may be empty); duplicate `id`s are a 422 |
| `interactions` | 0 to 200,000 of `{user_id, book_id, weight}` with `0 < weight <= 3` (the backend sends 1 per loan, +0.5 when renewed, 0.5 per reservation) |
| `limit` | 1 to 50, default 10 |

An interaction on a book that is not in `books` is ignored. A repeated (student, book) pair is summed and
capped at 3.

**The answer** is `{"model_version", "users": [{"user_id", "results": [{"book_id", "score", "method"}]}]}`:
students in `user_id` order, only those with at least one result (a student without interactions never
appears), at most `limit` results each, `score` in (0, 1], `method` `content`, `collaborative` or `hybrid`.
A student never gets a book they already have, a book whose final score is 0 is dropped, and equal scores
break on the book id, ascending. `model_version` is `tfidf-als-1` when the collaborative part ran and
`tfidf-1` when it did not.

**The engine** (`app/ml/recommend/`):

- **Content** (`content.py`). One text per book: title, author, genre (twice, so it counts more) and
  description. A `TfidfVectorizer` (English stop words, sublinear tf) is fitted on the catalog. A student's
  profile is the weight-summed TF-IDF vectors of their books, unit length; a candidate's content score is
  its cosine similarity to it. A catalog whose texts are all empty or all stop words gives every content
  score 0 and no error.
- **Collaborative** (`collaborative.py`). Only when the payload has at least 5 students and 20 interactions
  (distinct student-book pairs on catalog books): `implicit`'s `AlternatingLeastSquares(factors=16,
  regularization=0.1, iterations=15, random_state=42, use_gpu=False)`, fitted on the students-by-books matrix
  of weights multiplied by one fixed confidence constant (`ALS_CONFIDENCE = 40`, the value the original
  implicit-feedback ALS paper reports; not tuned), with BLAS limited to one thread (`threadpoolctl`).
  A student's score for a book is the dot product of their factors and the book's, clipped below at 0.
  Below the minimums, when `implicit` cannot be imported or when its fit raises, the answer is content only
  (`tfidf-1`): a failure is logged by its exception type alone. Two students with identical histories can
  get ALS scores up to about 3e-8 apart (implicit starts each student's factors from a random value), so a
  near-tie between them could order differently; in the one probe made their picks were identical.
- **Blend** (`engine.py`). Per student, the content scores and the collaborative scores are each divided by
  their maximum over the candidates (the books the student does not have; a maximum of 0 leaves them 0). With
  `n` the number of distinct books the student has, the collaborative weight is `w = min(0.7, 0.1 * n)` and
  the final score is `(1 - w) * content + w * collaborative`: a student with little history gets mostly
  content picks and one with a long history leans on the other students. A pick is `hybrid` when both
  parts are above 0, `collaborative` when only the collaborative part is, `content` otherwise.
- **Memory is bounded per chunk**: students are scored 512 at a time (a dense block of 512 x 5,000 floats
  is 20 MB), so the score blocks never grow with the number of students. The returned results do.

**The shared secret.** Every `/recommend/*` route needs the header `x-ai-token`, compared with the
environment variable `AI_TOKEN` in constant time (`hmac.compare_digest`). `AI_TOKEN` unset, empty or blank
means the recommender is off: the routes answer 404 with the body of a route that does not exist. A
missing or wrong header answers 401 `{"detail": "unauthorized"}`. The token never appears in a log, a
response or an error. `AI_TOKEN` is read on every request from the **process environment only**: an
`ai-backend/.env` file does not set it (that file only fills the declared settings), so export it in the
shell that starts uvicorn (PowerShell: `$env:AI_TOKEN = "..."`) or set it in the host's environment
settings. `/health` and `/predict/no-show` need no token. Two edges to know: a malformed JSON body answers
422 before the token is checked (FastAPI parses the body before it runs the token check), and while the
recommender is off `/openapi.json` and `/docs` still list `/recommend/batch`, and a `GET` of that path
answers 405 `Method Not Allowed` rather than 404 (a `POST` answers 404).

**Size and speed, measured** on one Windows laptop (16 cores, Python 3.13.1), over real HTTP with uvicorn on
127.0.0.1: one fresh server process per run, three runs per row, on a random payload of 5,000 books (about
45 words each) and 8,000 students, with 36-character UUID ids (what the backend sends: Prisma's `uuid()`) and
compact JSON. The peak is the operating system's own peak working set of the server process (the client ran
in another process); the service alone is 166 MB at startup, with the no-show model loaded.

| Interactions | Request body | Peak working set | Wall time | Answer |
|---|---|---|---|---|
| 50,000 (the backend's default send cap) | 7.2 MB | 395 to 396 MB | 7.8 to 8.0 s | 7.8 MB |
| 100,000 | 12.5 MB | 434 to 444 MB | 9.5 to 10.1 s | 7.8 MB |
| 200,000 (the contract's cap) | 23.1 MB | 510 to 530 MB | 11.3 to 11.4 s | 7.8 MB |

(MB is 2^20 bytes.) An earlier version of this section gave a 13.9 MB body, a 5.5 MB answer and a peak of
about 470 MB at the caps: that payload had short ids, so those figures understated what the backend sends;
do not rely on them. **A free 512 MB instance is not enough at the 200,000-interaction cap** (510 to 530 MB
here: at or over the limit). The 50,000 and 100,000 rows are below 512 MB here, by about 115 and 70 MB.
**Linux and Render were not measured**, nor a 512 MB or 0.1-CPU host (the free Render tier), where the call
will take much longer and Linux's accounting of memory differs from a Windows working set.

In process, the ALS fit alone took about 0.8 seconds on this laptop and about 0.3 seconds for 1,000
students, 2,000 books and 20,000 interactions (about 0.9 seconds for the whole call); the collaborative part
added about 20 to 40 MB to the peak (in process, so without the request body: 389 MB content only, 407 and
429 MB with ALS in two runs). Those in-process figures and the next one are from the earlier short-id
payload and were not re-run. The answer grows with the number of students, not of interactions: 40,000
students with 200,000 interactions took about 27 seconds and peaked at about 490 MB in process, and the
extreme the contract allows (200,000 students with one interaction each) was not run, but extrapolates to
more than a gigabyte and past the backend's 90-second timeout. The seeded demo is a tiny fraction of the
caps; a live campus grows towards them, because the backend sends every loan and reservation ever made
(never pruned) and only the send cap truncates it to the most recent ones.

**Tests** (`tests/test_recommend_*.py`, no network, a few seconds): the contract and the token
(`test_recommend_api.py`); the content engine on hand-made catalogs (`test_recommend_content.py`); the
collaborative part and the blend on a planted structure, 40 students in two groups of 20 whose books share
no word, so the content part sees nothing and an unseen same-group book can only be found by ALS
(`test_recommend_collaborative.py`; how many students find their book depends on the ALS seed, which is
fixed at 42: the test over all 40 asserts that at least half (20) find their book, and the claims about one student are labelled seed-42
observations, to be re-checked, not re-seeded, after an `implicit` upgrade). `tests/fixtures/recommend_batch_example.json` is a request and its
answer shared with the backend's contract test (a byte-identical copy lives at
`backend/test/fixtures/recommend-batch-example.json`; change both together and check with `cmp`;
`.gitattributes` keeps both LF). It stays below the ALS minimums, so it is a content-only contract.

## Deploy

Deployed on Render as a **second web service** next to the Node backend (checked on 2026-11-26 on the
free plan: `/health` answered 200 after a cold start of 48 to 68 seconds, and `POST /recommend/batch`
without the token answered 401). Without it, or with the backend's `AI_BACKEND_URL` unset, no booking is
scored, every priority stays 1.0 and students see the popularity list. To deploy it, run this as a **second
web service** next to the Node backend and set the backend's `AI_BACKEND_URL` to
its address. Set the **same long random `AI_TOKEN` on both services** (see *Recommender*): it is what protects the
recommender, which receives student ids. Install exactly `requirements.txt`, which pins numpy 2.5.2, scikit-learn 1.9.0,
xgboost 3.4.1 and joblib 1.5.3: the committed artifact is a pickle and fails to
load on other versions, and the service then answers 503 `model_unavailable` (the
backend treats that as "no nudge"). Loading also prints a joblib/NumPy
deprecation warning every time with these versions (harmless now; unpickling
breaks once NumPy removes that API, so retrain then).

What a hosted service needs (worked out from the files; the cold start and the two checks above are the only things measured on Render):

- **Build command** `pip install -r requirements.txt`; **start command**
  `uvicorn app.main:app --host 0.0.0.0 --port $PORT` (the `--reload --port 8000`
  line above binds 127.0.0.1 and is for local use only).
- **Python 3.13.** The artifact and every pin come from Python 3.13.1 (it is in
  `metrics.json`), so set `PYTHON_VERSION=3.13.1` rather than rely on the host's
  default. Whether numpy 2.5.2 and pandas 3.0.5 install on 3.11 has not been checked.
- **`implicit==0.7.3`** is the collaborative part of the recommender; it is the pin most
  likely to need a compiler on Linux. It installed and ran on Render (checked on 2026-11-26: the first
  refresh answered `tfidf-als-1`, so ALS ran). If the first build fails there, drop it from
  `requirements.txt`: the service then answers content-only picks (`tfidf-1`), a path the
  tests cover, and the backend keeps working.
- **Cold start.** A sleeping free instance takes tens of seconds to import and
  load the model, so the first booking after idle hits the backend's 2-second
  timeout and stays unscored. That is the designed fail-open path, not a fault.
  The recommender call has a 90-second timeout and is retried by the backend every 15 minutes.
  Measured twice, on 2026-11-26: a sleeping free instance answered `/health` after 47.7 and after 67.8 seconds,
  both inside that timeout, the second with only about 22 seconds to spare.
- **The shared secret is implemented.** Set `AI_TOKEN` to the same long random value on this
  service and on the backend: the backend sends it in the `x-ai-token` header on the
  recommender call only, and this service compares it in constant time. Unset or blank, the
  recommender is off and its routes answer 404 (401 for a missing or wrong header). It is read
  from the **process environment only**: an `ai-backend/.env` file does not set it, so put it
  in the host's environment settings.
- **"Internal only" is still a convention, not enforced:** a hosted web service is publicly
  reachable. `POST /predict/no-show` is a pure function of ten numbers (no database, no user
  data), so that is harmless, but do not share the address. There is **no request-body size
  limit**: FastAPI parses the body (about 23 MB at the contract's caps with UUID ids, and larger if
  someone sends it) before the token is checked, as `/predict/no-show` always did, so an anonymous
  caller who knows the address can make the service parse a big body. Add a platform limit or a
  guard when deploying publicly. A guard on the `Content-Length` header is not enough by itself: uvicorn also
  accepts a `Transfer-Encoding: chunked` body, which has no such header, so a guard has to count the bytes
  as they stream in. Whether Render offers a configurable request-size limit has not been checked. While
  `AI_TOKEN` is unset the routes answer 404 (a `GET` of `/recommend/batch` answers 405),
  but `/openapi.json` and `/docs` still list `/recommend/batch`, and a malformed JSON body
  answers 422 before the token is checked.
- **Memory.** Measured over HTTP on Windows with 36-character UUID ids (table in *Recommender*): the
  service alone is 166 MB, and a call peaked at about 395 MB with 50,000 interactions (the backend's
  default send cap), about 440 MB with 100,000 and 510 to 530 MB with 200,000 (the contract's cap). **A
  free 512 MB instance is not enough at 200,000 interactions.** The earlier figure of about 470 MB used
  short ids and was too low. The answer also grows with the number of students (the 200,000-student
  extreme would need more than a gigabyte and outlast the backend's 90-second timeout; not run). The
  seeded demo is a tiny fraction of these sizes; a live campus grows towards them, because the backend
  sends every loan ever made, so choose the instance size, or the send cap, for the history you expect.
  What an out-of-memory restart means: the refresh attempt fails (the backend retries every 15 minutes
  while due) and the restart is a cold start that also interrupts `/predict/no-show` if it shares the
  instance, so the bookings made meanwhile stay unscored (priority 1.0, fail-open). Students are not
  affected for a while: they keep the cached picks of the last good refresh for up to 14 days, then get
  the popularity list. Linux RSS and Render were not measured.

## Layout

```
app/
├── main.py             # FastAPI app, router registration, loads the model at startup
├── core/config.py      # pydantic-settings configuration
├── core/security.py    # the x-ai-token check (404 off, 401 wrong)
├── routers/
│   ├── health.py
│   ├── noshow.py       # POST /predict/no-show; loads the committed model once
│   └── recommend.py    # POST /recommend/batch (needs x-ai-token)
├── schemas/            # request/response models
└── ml/
    ├── noshow/         # features.py (order, encoding), history.py, simulate.py,
    │                   # train.py, model.py (load, predict, penalty_factor)
    ├── artifacts/      # the committed model and metrics.json
    └── recommend/      # content.py (TF-IDF), collaborative.py (implicit ALS), engine.py (blend, top picks)
tests/                  # pytest: contract, history features, simulator, training, serving, recommender
                        # fixtures/noshow_features.json: the feature contract shared with the
                        # backend (a byte-identical copy lives at backend/test/fixtures/
                        # noshow-features.json; change both together, check with cmp)
.gitattributes          # keeps the fixture and metrics.json LF and the artifact binary
```

## Note on requirements.txt

It's a full `pip freeze`, so it pins transitive dependencies too. The direct
dependencies are: `fastapi`, `uvicorn[standard]`, `pydantic`,
`pydantic-settings`, `xgboost`, `scikit-learn`, `pandas`, `implicit`. The no-show
model uses XGBoost, scikit-learn, pandas, numpy and joblib; the recommender uses
scikit-learn (TF-IDF), scipy and `implicit` (ALS). `requirements-dev.txt` adds `pytest` and `httpx` on top.
