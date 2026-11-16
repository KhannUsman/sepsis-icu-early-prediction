"""Model pipelines. Every model = imputer (+ scaler) + estimator."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config as C

RS = C.RANDOM_STATE

# name -> (estimator class, needs scaling, default settings)
MODELS = {
    "dummy": (DummyClassifier, False, dict(strategy="prior")),
    "logistic_regression": (LogisticRegression, True, dict(
        max_iter=500, C=0.5, class_weight="balanced")),
    "decision_tree": (DecisionTreeClassifier, False, dict(
        max_depth=8, min_samples_leaf=100, class_weight="balanced",
        random_state=RS)),
    "random_forest": (RandomForestClassifier, False, dict(
        n_estimators=120, max_depth=12, min_samples_leaf=30, max_samples=0.4,
        class_weight="balanced_subsample", n_jobs=-1, random_state=RS)),
    "svm": (SVC, True, dict(
        kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
        probability=True, random_state=RS)),
}


def build_model(name: str, **overrides) -> Pipeline:
    """Build a pipeline. Pass overrides to change model settings."""
    cls, scale, params = MODELS[name]
    params = {**params, **overrides}
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", cls(**params)))
    return Pipeline(steps)


def predict_proba(pipe: Pipeline, X):
    """Probability of sepsis (class 1)."""
    return pipe.predict_proba(X)[:, 1]

