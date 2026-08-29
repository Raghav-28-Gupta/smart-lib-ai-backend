---
name: ml-pipeline
description: Machine learning workflows, feature engineering, and model training patterns for XGBoost no-show prediction and hybrid book recommendations. Use when working on ML models in ai-backend/.
---

# ML Pipeline Guidelines

This skill describes workflows for feature engineering, model training, evaluation, and serialization in the `ai-backend/` package.

## 1. No-Show Prediction Pipeline (`app/ml/noshow/`)

### Feature Engineering
Key features derived for the XGBoost classifier:
- User historical show-rate / reliability score (`past_attendance_ratio`, `consecutive_attended_count`, `recent_noshows_count`)
- Booking context (`hour_of_day`, `day_of_week`, `duration_minutes`, `resource_type_seat_vs_room`)
- Advance notice (`booking_lead_time_hours`)

### Training & Evaluation
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

# Binary classification with calibrated probabilities
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42
)
```

### Fairness Bounding
Always bound the effect of the predicted probability $P(\text{no-show})$ when passing priority weights back to the booking engine:
$$\text{Penalty Factor} = \min(P \times \alpha, \text{MAX\_PENALTY\_CAP})$$
Ensure no user is indefinitely blocked or excessively penalized.

---

## 2. Hybrid Book Recommender Pipeline (`app/ml/recommend/`)

### Content-Based Component (TF-IDF)
- Compute cosine similarity across concatenated metadata: `f"{title} {author} {genre} {description}"`.
- Use `sklearn.feature_extraction.text.TfidfVectorizer` with sublinear TF scaling.

### Collaborative Filtering Component (Implicit ALS)
- Build sparse user-item interaction matrix from loan history, renewals, and catalog interactions.
- Train `implicit.als.AlternatingLeastSquares` with confidence weighting:
  $C_{ui} = 1 + \alpha R_{ui}$.

### Dynamic Blending
- For users with interaction count $N < 5$: return pure content-based recommendations.
- For users with $N \ge 5$: blend rank scores:
  $\text{Score} = w_{\text{cf}} \cdot S_{\text{als}} + (1 - w_{\text{cf}}) \cdot S_{\text{tfidf}}$.

---

## Model Serialization
- Save fitted models to `app/ml/artifacts/` using `joblib.dump(model, filepath)`.
- Always verify model artifact versions and include `model_version` metadata in API response payloads.
