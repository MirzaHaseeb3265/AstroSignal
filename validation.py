"""
Evaluation for the AstroSignal anomaly detectors.

Section 9 of the PRD says not to present unlabeled anomaly scores as
accuracy. Nothing in the SDSS catalogue tells us which objects are genuinely
unusual, so there is no ground truth to score against. This module measures
three things that can be defended without it.

1. agreement_matrix   how far the detectors overlap in their top N. This
                      says how much "unusual" depends on which model you run.
2. flag_enrichment    how much of each top N is data the pipeline already
                      flagged as suspect. High values mean the detector is
                      finding broken measurements, not astrophysics.
3. injection_recall   recall against synthetic objects we planted ourselves.
                      This is the only place in the project where a true
                      positive is actually known.

Two kinds of synthetic object get planted and the contrast between them is
the whole point. One imitates a measurement failure. The other is unusual
only in combination. A detector doing useful scientific work should find the
second kind, not just the first.

Run `python validation.py <path-to-csv>` to reproduce the numbers in the
README.
"""

import sys

import numpy as np
import pandas as pd

from anomaly_models import add_colour_features, score_iforest, score_lof, score_ocsvm

MAGNITUDE_COLUMNS = ["mag_u", "mag_g", "mag_r", "mag_i", "mag_z"]
COLOUR_COLUMNS = ["u_g", "g_r", "r_i", "i_z"]


def _strip_derived(df):
    """Drop stored colour columns so they get recomputed from magnitudes.

    add_colour_features leaves existing columns alone, so a frame carrying
    pre-computed colours would keep stale values after we edit magnitudes.
    """
    return df.drop(columns=[c for c in COLOUR_COLUMNS if c in df.columns])


def inject_broken_photometry(df, n=50, rng=None, offset_range=(4.0, 8.0)):
    """Plant objects that imitate a measurement failure.

    One randomly chosen band is shifted hard, which is what the real
    artifacts in this dataset look like. The worst real example carries
    mag_i 24.60 next to mag_z 17.44, a seven magnitude jump between adjacent
    bands. These are the easy case. A detector that misses them is broken.
    """
    rng = np.random.default_rng(rng)
    base = _strip_derived(df)
    picks = base.sample(n=n, random_state=int(rng.integers(1 << 31))).copy()
    for pos in range(len(picks)):
        band = MAGNITUDE_COLUMNS[rng.integers(len(MAGNITUDE_COLUMNS))]
        offset = rng.uniform(*offset_range) * (1 if rng.random() < 0.5 else -1)
        picks.iloc[pos, picks.columns.get_loc(band)] += offset
    picks["injected"] = "broken_photometry"
    return picks


def inject_subtle_novelty(df, n=50, rng=None):
    """Plant objects that are unusual only in combination.

    Every magnitude is resampled independently from that column's own
    observed values. So each individual measurement is one a real object
    actually had, and nothing looks extreme band by band. What gets
    destroyed is the correlation between bands, which is where the
    astrophysics lives.

    These sit off the manifold without being an outlier in any single
    feature. That is the closest stand-in this project has for a real
    discovery rather than a broken pixel. It is still a proxy and the README
    says so.
    """
    rng = np.random.default_rng(rng)
    base = _strip_derived(df)
    picks = base.sample(n=n, random_state=int(rng.integers(1 << 31))).copy()
    for col in MAGNITUDE_COLUMNS:
        picks[col] = rng.choice(base[col].to_numpy(), size=n, replace=True)
    picks["injected"] = "subtle_novelty"
    return picks


def build_injected_frame(df, n=50, seed=0):
    """Original rows plus both families of synthetic objects."""
    real = _strip_derived(df).copy()
    real["injected"] = "none"
    planted = [
        inject_broken_photometry(df, n=n, rng=seed),
        inject_subtle_novelty(df, n=n, rng=seed + 1),
    ]
    return pd.concat([real] + planted, ignore_index=True)


def recall_at(percentiles, injected, kind, top_n):
    """Share of planted objects of one kind that reach the top N."""
    planted = injected == kind
    if planted.sum() == 0:
        return float("nan")
    return planted.loc[percentiles.nlargest(top_n).index].sum() / planted.sum()


def agreement_matrix(rankings, top_n=50):
    """Pairwise overlap counts between each detector's top N."""
    tops = {name: set(s.nlargest(top_n).index) for name, s in rankings.items()}
    names = list(tops)
    return pd.DataFrame(
        [[len(tops[a] & tops[b]) for b in names] for a in names],
        index=names, columns=names,
    )


def flag_enrichment(flags, rankings, top_ns=(10, 25, 50, 100)):
    """Share of each detector's top N that already carries a quality flag."""
    rows = {
        name: {f"top {n}": flags.loc[s.nlargest(n).index].mean() * 100 for n in top_ns}
        for name, s in rankings.items()
    }
    out = pd.DataFrame(rows).T
    out.loc["baseline"] = flags.mean() * 100
    return out.round(1)


def injection_recall(df, n=50, seeds=(1, 2, 3, 4, 5, 6, 7, 8), top_n=100,
                     verbose=True):
    """Mean and spread of recall for each detector and each planted kind.

    Several seeds are used because a single run of 50 planted objects is
    noisy enough to mislead. The spread is reported alongside the mean so
    the reader can judge whether a gap is real.

    This fits three detectors per class per seed, so it takes a while and
    prints progress. Without that it looks like it has hung.
    """
    collected = {}
    for position, seed in enumerate(seeds, start=1):
        if verbose:
            print(f"  seed {position} of {len(seeds)}...", flush=True)
        aug = build_injected_frame(df, n=n, seed=seed)
        rankings = {
            "IsolationForest": score_iforest(aug).if_percentile,
            "LOF": score_lof(aug).lof_percentile,
            "OneClassSVM": score_ocsvm(aug).ocsvm_percentile,
        }
        for kind in ("broken_photometry", "subtle_novelty"):
            for name, series in rankings.items():
                key = (kind, name)
                value = recall_at(series, aug.injected, kind, top_n) * 100
                collected.setdefault(key, []).append(value)

    rows = []
    for (kind, name), values in collected.items():
        values = np.array(values)
        rows.append({
            "planted": kind, "detector": name,
            f"recall@{top_n} mean %": round(values.mean(), 1),
            "std": round(values.std(), 1),
            "runs": len(values),
        })
    return pd.DataFrame(rows).sort_values(["planted", f"recall@{top_n} mean %"],
                                          ascending=[True, False])


def quality_flags(df):
    """Recreate app.py's quality flags so this module runs on a raw export.

    Colours are derived here rather than read off the frame. The score_*
    functions build their feature matrix internally and do not return the
    colour columns, so reading them off a scored frame would silently find
    nothing and report only the negative redshift flag.
    """
    df = add_colour_features(df)
    colours = [c for c in COLOUR_COLUMNS if c in df.columns]
    extreme = (df[colours].abs().gt(4).any(axis=1) if colours
               else pd.Series(False, index=df.index))
    negative = (pd.to_numeric(df["z"], errors="coerce") < 0 if "z" in df.columns
                else pd.Series(False, index=df.index))
    return extreme | negative


def main(path):
    raw = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    print(f"loaded {len(raw)} rows from {path}")
    print("scoring with all three detectors, this takes a few seconds...",
          flush=True)

    scored = score_ocsvm(score_lof(score_iforest(raw)))
    print()
    rankings = {
        "IsolationForest": scored.if_percentile,
        "LOF": scored.lof_percentile,
        "OneClassSVM": scored.ocsvm_percentile,
    }

    print("TOP 50 AGREEMENT")
    print(agreement_matrix(rankings), "\n")

    print("QUALITY FLAG RATE IN TOP N (percent)")
    print(flag_enrichment(quality_flags(scored), rankings), "\n")

    print("RECALL ON PLANTED OBJECTS")
    print("planting synthetic objects and re-scoring over 8 seeds.")
    print("this is the slow part, roughly one to three minutes.", flush=True)
    print(injection_recall(raw).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample.csv")
