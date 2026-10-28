# steps/train.py
from sklearn.ensemble import RandomForestClassifier

def estimator_fn(estimator_params=None):
    p = estimator_params or {}
    return RandomForestClassifier(
        n_estimators=p.get("n_estimators", 100),
        max_depth=p.get("max_depth", None),
        random_state=42
    )
