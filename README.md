# AstroSignal

AstroSignal ranks objects from an SDSS catalogue export so a person can decide
which ones to look at first. It does not claim to find new astrophysics. Every
object it flags is a candidate for inspection, nothing more.

Built for the Open Scientific Intelligence hackathon.

## Data

Source: SDSS DR17 catalogue data, pulled through SkyServer SQL Search. The
exact query used to produce `data/sample.csv`:

```sql
SELECT TOP 5000
    objID,
    ra,
    dec,
    psfMag_u - extinction_u AS mag_u,
    psfMag_g - extinction_g AS mag_g,
    psfMag_r - extinction_r AS mag_r,
    psfMag_i - extinction_i AS mag_i,
    psfMag_z - extinction_z AS mag_z,
    class,
    z
FROM SpecPhoto
WHERE class IN ('GALAXY', 'STAR', 'QSO')
    AND psfMag_g BETWEEN 12 AND 25
    AND psfMag_r BETWEEN 12 AND 25
    AND psfMag_u IS NOT NULL
    AND psfMag_g IS NOT NULL
    AND psfMag_r IS NOT NULL
    AND psfMag_i IS NOT NULL
    AND psfMag_z IS NOT NULL
```

Three things this query decides, which matter when reading any result below.

Magnitudes are PSF magnitudes, corrected for galactic extinction. PSF
magnitudes fit a point source profile, so they are the right measurement for
stars and quasars and they underestimate the flux of galaxies, which are
extended. Since 85 percent of this sample is galaxies, every galaxy colour
here carries that systematic. Model or Petrosian magnitudes would be the
usual choice for extended sources.

The table is `SpecPhoto`, so every object has a spectrum. That is why `class`
and `z` are spectroscopic rather than photometric estimates, and it also
means the sample inherits SDSS spectroscopic target selection, which is not a
uniform sample of the sky.

`TOP 5000` has no `ORDER BY`, so this is the first 5,000 rows the server
returned, not a random draw. The result is a contiguous block: 55 degrees of
right ascension but only 2.5 degrees of declination, a narrow equatorial
strip. Consecutive objects sit about 0.004 degrees apart in right ascension.

The export timestamp was not recorded. The release is DR17 per the project
plan, though the query text itself does not pin it.

The working file holds 5,000 objects with these columns:

| column | meaning |
| --- | --- |
| `objID` | SDSS object identifier |
| `ra`, `dec` | sky position in degrees |
| `mag_u` ... `mag_z` | photometric magnitudes in the five SDSS bands |
| `class` | spectroscopic class, one of GALAXY, STAR, QSO |
| `z` | redshift |

Class balance is uneven: 4,232 GALAXY, 431 STAR, 338 QSO. A model that always
answers GALAXY would score 84.6 percent, so accuracy is not a useful number
here. Anything supervised should report macro F1 and use a stratified split.

### Known problem with objID

The current export wrote `objID` in scientific notation, as `1.23765E+18`. An
SDSS object ID is an 18 digit integer and a float only holds about 15
significant digits, so the trailing digits are gone. All 5,000 rows collapse
into 3 distinct values and the IDs cannot be recovered from this file.

What this breaks: object ID lookup, and any link back to an SDSS object page.

Workaround in place: `ra` and `dec` survived at full precision, so image
cutouts and SkyServer links are built from coordinates instead.

Proper fix: re-export from SkyServer with the ID cast to text, for example
`CAST(objID AS varchar) AS objID` in the SELECT. `data_loader` reads the
column as a string so a corrected export will not be damaged again.

### Other things worth knowing about this file

- 5,001 rows were exported, one of which is an exact duplicate of another.
- 118 rows carry a redshift at or below zero. Most of these are stars, where a
  redshift scattering around zero is normal. They are flagged, not dropped.
- No missing values and no -9999 sentinel magnitudes.
- `dec` only spans -1.26 to +1.26 degrees. This is a narrow strip of sky, not a
  random sample of the survey, which matters when interpreting anything the
  detectors flag.

## Running it

```
pip install -r requirements.txt
streamlit run app.py
```

To reproduce the evaluation numbers below:

```
python validation.py data/sample.csv
```

## Modules

| file | what it does |
| --- | --- |
| `app.py` | Streamlit interface, data loading, candidate views |
| `anomaly_models.py` | the three detectors and the per-class scoring runner |
| `validation.py` | agreement, quality flag enrichment, synthetic injection |

## Methods

Three detectors run over the same feature matrix: the four colour indices
(u-g, g-r, r-i, i-z), `mag_g`, and redshift. Each one is fitted separately per
class, so "anomalous" means unusual compared with other objects of the same
class, not unusual compared with the whole catalogue.

- **Isolation Forest** asks how few random splits are needed to separate a
  point from the rest. 300 trees, `contamination="auto"`.
- **Local Outlier Factor** compares a point's local density with the density of
  its neighbours, so it finds objects in sparse pockets. `n_neighbors=20`.
- **One-Class SVM** fits a boundary around the bulk of each class and scores
  distance from it. RBF kernel, `nu=0.05`.

Isolation Forest runs on unscaled features because it splits one feature at a
time and does not care about scale. The other two measure distances, so they
get standardised input. Without that, `mag_g` with a standard deviation of 1.15
would swamp `i_z` at 0.27 and the ranking would mostly track brightness.

## What we measured

Section 9 of the PRD says unlabeled anomaly scores must not be presented as
accuracy. The catalogue does not tell us which objects are genuinely unusual,
so there is no ground truth here. We measured three things that stand up
without it.

### 1. The detectors disagree about what is unusual

Overlap between each pair's top 50 candidates:

|  | Isolation Forest | LOF | One-Class SVM |
| --- | --- | --- | --- |
| **Isolation Forest** | 50 | 23 | 25 |
| **LOF** | 23 | 50 | 38 |
| **One-Class SVM** | 25 | 38 | 50 |

LOF and One-Class SVM agree with each other far more than either agrees with
Isolation Forest. The real split is distance based against tree based, not
three independent opinions. No single ranking is the answer, and the app should
not present one as if it were.

### 2. Top candidates are mostly data quality problems

Share of each detector's top N that already carries a quality flag, against a
baseline of 2.5 percent across the whole file:

| detector | top 10 | top 25 | top 50 | top 100 |
| --- | --- | --- | --- | --- |
| Isolation Forest | 60% | 28% | 20% | 13% |
| LOF | 80% | 40% | 24% | 13% |
| One-Class SVM | 70% | 40% | 26% | 18% |

The current top ranked candidate has `mag_i` 24.60 next to `mag_z` 17.44, a
seven magnitude jump between adjacent bands. That is a broken measurement
sitting at the 100th percentile.

This is not a model problem. All three detectors do it, because a bad
measurement is extreme under every metric. Quality flags exist in the pipeline
but they are applied after scoring, as a display filter, so they do not affect
the ranking at all.

### 3. The detectors are good at different things

We planted 50 synthetic objects of each of two kinds, then measured how many
reached the top 100. Repeated over 8 seeds.

- **Broken photometry**: one band shifted by 4 to 8 magnitudes. Imitates a
  measurement failure. The easy case.
- **Subtle novelty**: every magnitude resampled independently from that
  column's own values. Each measurement is one a real object had, so nothing
  looks extreme band by band, but the correlation between bands is destroyed.
  This is our stand-in for a real discovery.

| planted | Isolation Forest | LOF | One-Class SVM |
| --- | --- | --- | --- |
| broken photometry | 56% ± 7.6 | 54% ± 6.7 | **91% ± 7.4** |
| subtle novelty | 41% ± 7.0 | **72% ± 3.4** | 52% ± 6.0 |

One-Class SVM is a quality screen. It catches nine out of ten broken
measurements. LOF is the discovery ranker, clearly best on the subtle case and
the most consistent run to run. Isolation Forest, which is what the app
currently ranks by, is the weakest at both.

So adding detectors does not automatically improve anything. Averaging them
makes it worse, because consensus concentrates the artifacts every method
agrees on. Of the 21 objects all three put in their top 50, 33 percent are
quality flagged against the 2.5 percent baseline, while objects found by only
one method sit at 2 percent, which is normal.

### 4. Removing redshift does not help, and finding that out took two tries

Redshift is a strong feature. Stars sit near 0.0001, galaxies near 0.10 and
quasars near 1.17, so it nearly separates the classes on its own. We tested
whether dropping it changes the rankings.

First run, using the injection described above, which resamples magnitudes
and leaves redshift alone. Recall on subtle novelty at top 100:

| detector | with redshift | without redshift |
| --- | --- | --- |
| Isolation Forest | 41% | 62% |
| LOF | 72% | 77% |
| One-Class SVM | 52% | 62% |

That looks like a clear argument for dropping redshift. It is not. The
injection never touched redshift, so that feature carried no signal about
which objects were planted, and removing a feature that is pure noise for the
task will always help. The experiment was answering the wrong question.

Second run, with an injection that resamples redshift as well, so no feature
is left untouched:

| detector | with redshift | without redshift |
| --- | --- | --- |
| Isolation Forest | 74% | 79% |
| LOF | 92% | 90% |
| One-Class SVM | 93% | 91% |

The effect disappears. Two detectors get slightly worse. So redshift stays in
the feature set, and the first table is recorded here as a mistake we caught
rather than a result.

The general lesson matters more than the redshift question: with synthetic
injection, the design of the injection decides which features look important.
Any feature the injection does not perturb will look droppable.

**What we recommend:** screen with One-Class SVM first and set those objects
aside, rank what remains with LOF, and keep Isolation Forest as the documented
baseline to show why one model on its own is not enough.

## Limitations

- A quality flag is a heuristic, not confirmed corruption. `|colour| > 4` will
  catch some genuinely unusual objects alongside the broken ones. Nothing here
  should be called an artifact without checking the original SDSS record.
- Every ranking is in-sample. Each detector is fitted on the same objects it
  then scores, which is normal for anomaly detection over a fixed catalogue
  but means none of these numbers say anything about generalising to new data.
  A held-out design would be needed for that.
- LOF neighbourhood statistics get unstable for very small classes. The
  smallest class here is QSO with 338 rows so it is not a concern, but an
  uploaded file with a class of 5 to 20 objects would produce unreliable LOF
  scores. Classes under 5 rows are scored 0.
- Subtle novelty is a proxy. Resampling bands independently breaks the
  correlation between them, which is one specific way to be off the manifold.
  A real astronomical discovery may not look like this at all.
- 50 planted objects per run across 8 seeds is enough to separate the large
  gaps above, not enough for small ones.
- `nu=0.05` and `n_neighbors=20` were chosen, not tuned. The rankings depend on
  both.
- Missing values are filled with the median of the class being fitted, not
  the median of the whole catalogue. `app.py` uses the global median. This
  file has no missing values so the two agree exactly here, but class medians
  are far apart, redshift sits at 0.0001 for stars against 0.0995 globally, so
  the global version would hand a star a galaxy-like redshift and then report
  it as anomalous within its own class. Worth aligning `app.py` before any
  export with gaps is loaded.
- Scores from the three detectors are on different scales and two of them see
  standardised input, so raw scores are not comparable. Every comparison in
  this project uses the within-class percentile instead.
- Percentiles are ranked within class. Small classes produce ties at the top,
  so comparing candidates across classes needs care. We have not verified how
  much this affects the top 50.
- Redshift is included as a model feature, and the test above says keeping it
  is defensible. That test depends on our injection design, so it is evidence
  rather than proof.
- The sky coverage is a narrow strip, so selection effects are present and
  anything flagged may reflect them rather than anything physical.
- Galaxy photometry uses PSF magnitudes, which systematically underestimate
  the flux of extended sources. Four of the six model features are colours
  built from those magnitudes, so galaxy colours carry a systematic offset
  that a model or Petrosian magnitude would not have. Per-class fitting
  absorbs part of this, since every galaxy is compared with other galaxies,
  but it is a real property of the input rather than of the sky.
- The sample comes from SpecPhoto, so every object was selected for
  spectroscopy by SDSS targeting rules. It is a sample of what SDSS chose to
  observe, not of what is out there.
- `TOP 5000` with no `ORDER BY` means the rows are whatever the server
  returned first, so this is a contiguous block of the catalogue rather than a
  random sample. Any distribution here reflects that block.

## Data citation

SDSS data must be cited according to the collaboration's guidance at
https://www.sdss.org/collaboration/citing-sdss/

The query above is recorded for reproducibility. The export timestamp was not
captured at the time and the query text does not name a data release, so both
are stated from the project plan rather than from the export itself.
