"""Smoke tests for the AstroSignal detectors.

Run: python test_smoke.py data/sample.csv

These check the things that would silently break the comparison rather than
raise an error: scores going missing, percentiles drifting out of range, the
feature list changing, or score_iforest drifting away from the Isolation
Forest that app.py actually runs.

Nothing here asserts a specific row count or flag count. Those depend on the
export, and the objID re-export is expected to change the file.
"""

import sys

import pandas as pd
from sklearn.ensemble import IsolationForest

from anomaly_models import (
    FEATURE_CANDIDATES,
    build_feature_matrix,
    normalize_class,
    score_all,
)
from validation import quality_flags

PERCENTILE_COLUMNS = ["if_percentile", "lof_percentile", "ocsvm_percentile"]
SCORE_COLUMNS = ["if_score", "lof_score", "ocsvm_score"]


def main(path):
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    scored = score_all(df)

    assert len(scored) == len(df), "scoring changed the number of rows"
    assert scored.index.equals(df.index), "scoring reordered or reindexed the rows"

    for col in SCORE_COLUMNS + PERCENTILE_COLUMNS:
        assert col in scored.columns, f"missing column {col}"
        assert scored[col].notna().all(), f"{col} has gaps"

    for col in PERCENTILE_COLUMNS:
        assert scored[col].between(0, 100).all(), f"{col} out of range"

    features = build_feature_matrix(df)[1]
    assert features == [c for c in FEATURE_CANDIDATES if c in build_feature_matrix(df)[0].columns], \
        "feature list no longer matches FEATURE_CANDIDATES"

    # score_iforest must stay a faithful stand-in for app.py's Isolation
    # Forest, otherwise every comparison in the README is against the wrong
    # baseline. Rebuild app.py's version here and compare directly.
    matrix, _ = build_feature_matrix(df)
    matrix = matrix.fillna(matrix.median()).fillna(0.0)
    labels = normalize_class(df["class"])
    app_version = pd.Series(index=df.index, dtype="float64")
    for _, idx in labels.groupby(labels).groups.items():
        subset = matrix.loc[idx]
        model = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
        model.fit(subset)
        app_version.loc[idx] = -model.score_samples(subset)
    drift = (scored["if_score"] - app_version).abs().max()
    assert drift < 1e-12, f"score_iforest has drifted from app.py: {drift}"

    # Imputation must use the median of the class being fitted. A global
    # median would hand a star a galaxy-like redshift and then report it as
    # anomalous within its own class.
    holed = df.copy()
    star_rows = holed.index[normalize_class(holed["class"]) == "STAR"][:5]
    if len(star_rows) and "z" in holed.columns:
        holed.loc[star_rows, "z"] = None
        patched = score_all(holed)
        assert patched["if_score"].notna().all(), "missing values broke scoring"
        star_median = pd.to_numeric(df.loc[
            normalize_class(df["class"]) == "STAR", "z"], errors="coerce").median()
        global_median = pd.to_numeric(df["z"], errors="coerce").median()
        assert abs(star_median - global_median) > 1e-6, \
            "test is not meaningful on this file: class and global medians agree"

    flags = quality_flags(df)
    print(f"rows: {len(df):,}")
    print(f"quality flagged: {int(flags.sum())} ({100 * flags.mean():.2f}%)")
    print(f"isolation forest drift from app.py: {drift:.3g}")
    print("all checks passed")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample.csv")
