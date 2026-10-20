"""
Anomaly detectors for AstroSignal.

app.py already scores objects with an Isolation Forest. This module adds
Local Outlier Factor and One-Class SVM, and re-implements the Isolation
Forest with identical settings so all three can be compared fairly.

Every detector follows the same contract as score_dataframe() in app.py:
  - same feature list
  - one model fitted per class, so "anomalous" means unusual within its own
    class rather than globally
  - two output columns, a raw score where higher is more anomalous, and a
    percentile ranking within the class

One difference is deliberate. Isolation Forest splits one feature at a time
and ignores scale. LOF and One-Class SVM measure distances, so without
scaling mag_g (std 1.15) would drown out i_z (std 0.27) and the ranking
would mostly track brightness. The two distance based detectors therefore
get standardised input. This is recorded in the limitations section of the
README because it is a real difference between the methods, not a detail.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# Must stay identical to feature_candidates in app.py score_dataframe().
FEATURE_CANDIDATES = ["u_g", "g_r", "r_i", "i_z", "mag_g", "z"]

COLOUR_PAIRS = [
    ("u_g", "mag_u", "mag_g"),
    ("g_r", "mag_g", "mag_r"),
    ("r_i", "mag_r", "mag_i"),
    ("i_z", "mag_i", "mag_z"),
]

# app.py scores a class as 0.0 when it holds fewer rows than this.
MIN_ROWS_PER_CLASS = 5


def add_colour_features(df):
    """Add the u-g, g-r, r-i and i-z colour indices.

    Existing columns are left alone so a pre-computed export is not
    overwritten. Same rule as app.py.
    """
    df = df.copy()
    for out, left, right in COLOUR_PAIRS:
        if out not in df.columns and left in df.columns and right in df.columns:
            df[out] = (
                pd.to_numeric(df[left], errors="coerce")
                - pd.to_numeric(df[right], errors="coerce")
            )
    return df


def build_feature_matrix(df):
    """Return (matrix, feature_names) with non-finite values turned into NaN.

    Imputation deliberately does NOT happen here. app.py fills NaN with the
    median of the whole catalogue, but it then fits one model per class, and
    those two things disagree. Class medians are far apart: redshift sits at
    0.0001 for stars and 1.1704 for quasars against a global 0.0995. A star
    with a missing redshift would be handed a galaxy-like value and then look
    anomalous inside its own class, so the imputation would manufacture the
    anomaly it later reports.

    score_per_class fills NaN with the median of the class being fitted.
    Nothing is dropped, so every object still gets a score.

    This file has no missing values, so the choice changes none of the current
    numbers. It matters for any other export.
    """
    df = add_colour_features(df)
    features = [c for c in FEATURE_CANDIDATES if c in df.columns]
    if len(features) < 2:
        raise ValueError(
            "At least two of these features are required: "
            + ", ".join(FEATURE_CANDIDATES)
        )

    matrix = (
        df[features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([float("inf"), float("-inf")], float("nan"))
        .astype("float64")
    )
    return matrix, features


def normalize_class(series):
    """Uppercase, trimmed class labels. Same rule as app.py."""
    return series.fillna("UNKNOWN").astype(str).str.upper().str.strip()


def score_per_class(df, fit_score, scale=True, class_column="class"):
    """Fit one detector per class and return raw scores plus percentiles.

    fit_score(values) takes one class's rows as a 2D array and returns raw
    scores where higher is more anomalous. Each detector flips its own sign
    before returning, so this runner never needs to know which library it is
    calling.

    Imputation and scaling both happen inside the loop, so each class is
    filled and standardised against its own population rather than against
    the whole catalogue. That matches the per-class fitting. See
    build_feature_matrix for why the global version is unsafe.

    Returns (scores, percentiles, feature_names).
    """
    matrix, features = build_feature_matrix(df)
    labels = normalize_class(df[class_column])

    scores = pd.Series(index=df.index, dtype="float64")
    percentiles = pd.Series(index=df.index, dtype="float64")

    for label in labels.unique():
        idx = labels.index[labels == label]
        subset = matrix.loc[idx]

        if len(subset) < MIN_ROWS_PER_CLASS:
            raw = pd.Series(0.0, index=idx)
        else:
            subset = subset.fillna(subset.median()).fillna(0.0)
            values = subset.to_numpy()
            if scale:
                values = StandardScaler().fit_transform(values)
            raw = pd.Series(fit_score(values), index=idx)

        scores.loc[idx] = raw
        percentiles.loc[idx] = raw.rank(pct=True) * 100.0

    return scores, percentiles, features


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def iforest_scores(values, n_estimators=300, random_state=42):
    """Isolation Forest. Asks how few random splits isolate a point.

    These settings replicate app.py exactly, so comparisons are against the
    model the deployed app actually runs. score_samples is higher for
    inliers, so it is negated.
    """
    model = IsolationForest(
        n_estimators=n_estimators, contamination="auto", random_state=random_state
    )
    model.fit(values)
    return -model.score_samples(values)


def lof_scores(values, n_neighbors=20):
    """Local Outlier Factor. Compares a point's local density to its neighbours'.

    This finds objects sitting in a sparse pocket of an otherwise dense
    region, which is a different question from the one Isolation Forest
    asks. The two can disagree. On this dataset they do: their top 50
    candidates overlap by only 23 out of 50. See validation.py, which
    produces that number.

    n_neighbors is capped at len(values) - 1 because per-class fitting means
    a small class can have fewer rows than the requested neighbourhood.

    novelty stays False because every run refits on the data being scored,
    which is how app.py behaves when a user uploads a CSV. In that mode
    scikit-learn exposes the fitted scores on the estimator instead of
    through predict(). negative_outlier_factor_ is negative and falls as a
    point gets more unusual, so it is negated.
    """
    k = max(1, min(n_neighbors, len(values) - 1))
    model = LocalOutlierFactor(n_neighbors=k, contamination="auto")
    model.fit_predict(values)
    return -model.negative_outlier_factor_


def ocsvm_scores(values, nu=0.05, gamma="scale"):
    """One-Class SVM. Fits a boundary around the bulk and scores distance from it.

    Unlike LOF this judges a point against one boundary for the whole class
    rather than against its own neighbourhood, so it tends to flag points
    far from the centre rather than points in sparse pockets.

    nu is an upper bound on the fraction of training points allowed outside
    the boundary, so it sets roughly how much of each class counts as
    anomalous. 0.05 is a choice we made, not a tuned value, and the ranking
    depends on it. It belongs in the limitations section.

    Needs standardised input, which score_per_class supplies. The RBF kernel
    measures plain distances. decision_function is positive inside the
    boundary, so it is negated.
    """
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(values)
    return -model.decision_function(values)


# ---------------------------------------------------------------------------
# Convenience wrappers, each returning a copy of df with two new columns
# ---------------------------------------------------------------------------

def score_iforest(df, class_column="class"):
    """Isolation Forest columns. scale is False to match app.py."""
    scores, percentiles, features = score_per_class(
        df, iforest_scores, scale=False, class_column=class_column
    )
    out = df.copy()
    out["if_score"] = scores
    out["if_percentile"] = percentiles
    out["if_features"] = ", ".join(features)
    return out


def score_lof(df, n_neighbors=20, class_column="class"):
    """Local Outlier Factor columns."""
    scores, percentiles, features = score_per_class(
        df, lambda v: lof_scores(v, n_neighbors=n_neighbors), class_column=class_column
    )
    out = df.copy()
    out["lof_score"] = scores
    out["lof_percentile"] = percentiles
    out["lof_features"] = ", ".join(features)
    out["lof_n_neighbors"] = n_neighbors
    return out


def score_ocsvm(df, nu=0.05, gamma="scale", class_column="class"):
    """One-Class SVM columns."""
    scores, percentiles, features = score_per_class(
        df, lambda v: ocsvm_scores(v, nu=nu, gamma=gamma), class_column=class_column
    )
    out = df.copy()
    out["ocsvm_score"] = scores
    out["ocsvm_percentile"] = percentiles
    out["ocsvm_features"] = ", ".join(features)
    out["ocsvm_nu"] = nu
    return out


def score_all(df, class_column="class"):
    """Run all three detectors and return one frame with all six columns."""
    out = score_iforest(df, class_column=class_column)
    out = score_lof(out, class_column=class_column)
    out = score_ocsvm(out, class_column=class_column)
    return out
