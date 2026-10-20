from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# AstroSignal — SDSS-inspired astronomy anomaly exploration prototype
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AstroSignal | Astronomical Discovery",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path(__file__).parent / "astrosignal_ranked_candidates.csv"
SDSS_BASE = "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"
SDSS_EXPLORE = "https://skyserver.sdss.org/dr17/VisualTools/quickobj"

# Shared visual theme: keep palette changes centralized here and in config.toml.
THEME = {
    "bg": "#070a14", "panel": "#0d1425", "line": "#26334b",
    "text": "#edf4ff", "muted": "#9aabc4", "cyan": "#55e6f2",
    "blue": "#6ca8ff", "violet": "#a78bfa", "orange": "#ff9d4d",
}

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@500;600;700&family=Rajdhani:wght@500;600;700&display=swap');
      :root { --bg:#070a14; --panel:rgba(13,20,37,.78); --line:rgba(170,198,235,.17); --text:#edf4ff; --muted:#9aabc4; --cyan:#55e6f2; --blue:#6ca8ff; --violet:#a78bfa; --orange:#ff9d4d; }
      html, body, [class*="css"] { font-family:'Inter',sans-serif; }
      .stApp { background:transparent; color:var(--text); }
      [data-testid="stAppViewContainer"] { background:linear-gradient(135deg,#070a14 0%,#0b1020 55%,#070a14 100%); }
      [data-testid="stAppViewContainer"]:before { content:""; position:fixed; inset:0; z-index:0; pointer-events:none; opacity:.7; background-image:radial-gradient(1px 1px at 8% 18%,#d8f7ff 98%,transparent),radial-gradient(1px 1px at 22% 73%,#b9d9ff 98%,transparent),radial-gradient(2px 2px at 71% 23%,#e5e9ff 98%,transparent),radial-gradient(1px 1px at 84% 66%,#d8f7ff 98%,transparent),radial-gradient(1px 1px at 52% 91%,#c9d9ff 98%,transparent),radial-gradient(ellipse at 14% 5%,rgba(55,93,170,.15),transparent 42%),radial-gradient(ellipse at 88% 25%,rgba(115,70,180,.11),transparent 36%); background-size:260px 260px,340px 340px,420px 420px,300px 300px,390px 390px,auto,auto; animation:starDrift 90s linear infinite; }
      [data-testid="stMainBlockContainer"], [data-testid="stSidebar"] { position:relative; z-index:1; }
      @keyframes starDrift { from { background-position:0 0,0 0,0 0,0 0,0 0,center,center; } to { background-position:130px 260px,-160px 340px,210px -420px,-120px 300px,190px -390px,center,center; } }
      @media (prefers-reduced-motion: reduce) { [data-testid="stAppViewContainer"]:before { animation:none; } * { transition:none !important; } }
      [data-testid="stHeader"] { background:rgba(7,10,20,.72); }
      [data-testid="stSidebar"] { background:rgba(7,11,23,.88); border-right:1px solid var(--line); backdrop-filter:blur(16px); }
      [data-testid="stSidebar"] * { color:var(--text); }
      h1,h2,h3 { color:#f1f6ff; letter-spacing:-.025em; }
      h1,h2,h3,.eyebrow,.section-kicker { font-family:'Rajdhani','Orbitron',sans-serif; }
      p, li, label { color:#d1daeb; }
      .hero { padding:1.65rem 1.8rem; border:1px solid var(--line); border-radius:16px; background:linear-gradient(125deg,rgba(18,31,56,.86),rgba(10,16,31,.75)); backdrop-filter:blur(12px); margin-bottom:1.25rem; }
      .eyebrow { color:var(--cyan); font-family:'Orbitron',sans-serif; font-size:.68rem; letter-spacing:.13em; text-transform:uppercase; }
      .hero h1 { font-family:'Rajdhani',sans-serif; font-size:clamp(2.1rem,4vw,3.3rem); line-height:1.02; margin:.55rem 0 .7rem; }
      .hero p { max-width:850px; color:#c4d1e7; font-size:.94rem; line-height:1.65; }
      .section-kicker { color:var(--cyan); font-family:'Orbitron',sans-serif; text-transform:uppercase; letter-spacing:.1em; font-size:.65rem; }
      .metric-card,[data-testid="stMetric"] { background:rgba(13,20,37,.76); border:1px solid var(--line); padding:1rem 1.05rem; border-radius:11px; backdrop-filter:blur(10px); }
      .metric-card { min-height:96px; }
      .metric-label { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.07em; }
      .metric-value { color:var(--text); font-size:1.75rem; font-weight:700; margin-top:.35rem; }
      .metric-note,.small-muted { color:var(--muted); font-size:.78rem; }
      .info-panel { background:rgba(13,20,37,.72); border:1px solid var(--line); padding:1rem 1.1rem; border-radius:11px; backdrop-filter:blur(10px); }
      .stButton button,.stDownloadButton button { border-radius:9px; border:1px solid var(--cyan); background:rgba(85,230,242,.12); color:#eafcff; min-height:2.35rem; transition:all .24s ease; }
      .stButton button[kind="primary"] { background:#38cfe0; border-color:#55e6f2; color:#06131b; font-weight:700; }
      .stButton button:hover,.stDownloadButton button:hover { border-color:#9af5ff; box-shadow:0 0 16px rgba(85,230,242,.16); transform:translateY(-1px); }
      .stDownloadButton button { background:rgba(85,230,242,.16); color:#f0fdff; font-weight:600; }
      [data-testid="stMetricLabel"] { color:var(--muted); } [data-testid="stMetricValue"] { color:var(--text); }
      div[data-baseweb="select"] > div,.stTextInput input,.stNumberInput input,.stTextArea textarea { background:rgba(13,20,37,.9); border-color:var(--line); color:var(--text); }
      [data-testid="stFileUploader"] section { background:rgba(13,20,37,.7); border:1px dashed rgba(85,230,242,.42); border-radius:10px; }
      [data-testid="stExpander"] { border:1px solid var(--line); border-radius:10px; background:rgba(13,20,37,.5); }
      div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
      hr { border-color:var(--line); } a { color:var(--cyan) !important; } code { color:var(--cyan); }
      .warning-badge { display:inline-block; border:1px solid rgba(255,157,77,.55); background:rgba(255,157,77,.12); color:#ffc38d; border-radius:6px; padding:2px 7px; font-size:.72rem; }
    </style>
    """, unsafe_allow_html=True,
)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5,10,22,0.18)",
    font=dict(family="Inter, sans-serif", color="#cbd5e1", size=11),
    margin=dict(l=48, r=20, t=72, b=54),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0, title_text=""),
    hoverlabel=dict(bgcolor="#101a2d", bordercolor="#40536f", font=dict(color="#edf4ff", family="Inter, sans-serif")),
)
CLASS_COLORS = {"STAR":"#6ca8ff", "QSO":"#a78bfa", "QUASAR":"#a78bfa", "GALAXY":"#55e6c1", "UNKNOWN":"#9aa9bf", "ALL":"#6ca8ff"}
ANOMALY_COLOR = "#ff9d4d"
PLOTLY_TEMPLATE = go.layout.Template(layout=go.Layout(**PLOT_LAYOUT, xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.06)", zeroline=False, showline=True, linecolor="rgba(190,210,240,.22)"), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.06)", zeroline=False, showline=True, linecolor="rgba(190,210,240,.22)")))
px.defaults.template = PLOTLY_TEMPLATE


def add_colour_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add colour indices if the input contains the required magnitudes."""
    df = df.copy()
    pairs = [("u_g", "mag_u", "mag_g"), ("g_r", "mag_g", "mag_r"),
             ("r_i", "mag_r", "mag_i"), ("i_z", "mag_i", "mag_z")]
    for out, left, right in pairs:
        if out not in df.columns and left in df.columns and right in df.columns:
            df[out] = pd.to_numeric(df[left], errors="coerce") - pd.to_numeric(df[right], errors="coerce")
    for col in df.columns:
        if col not in {"class", "objID", "candidate_row", "specObjID"}:
            # pandas 3.0 no longer accepts errors="ignore" in to_numeric.
            # Convert only columns that contain at least one numeric value;
            # preserve descriptive text columns unchanged.
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any() or df[col].isna().all():
                df[col] = converted
    return df



# Common SDSS-style and astronomy column aliases. These are deliberately
# conservative: values are not converted between units or surveys automatically.
COLUMN_ALIASES = {
    "ra": ["right_ascension", "raj2000", "ra_deg", "alpha"],
    "dec": ["declination", "dej2000", "dec_deg", "delta"],
    "class": ["object_class", "objclass", "type", "spectral_class"],
    "mag_u": ["u_mag", "u_magnitude", "umag", "modelmag_u", "psfmag_u", "petromag_u", "u"],
    "mag_g": ["g_mag", "g_magnitude", "gmag", "modelmag_g", "psfmag_g", "petromag_g", "g"],
    "mag_r": ["r_mag", "r_magnitude", "rmag", "modelmag_r", "psfmag_r", "petromag_r", "r"],
    "mag_i": ["i_mag", "i_magnitude", "imag", "modelmag_i", "psfmag_i", "petromag_i", "i"],
    "mag_z": ["z_mag", "z_magnitude", "zmag", "modelmag_z", "psfmag_z", "petromag_z"],
    "z": ["redshift", "photo_z", "photoz", "z_phot"],
    "u_g": ["u-g", "u_minus_g", "u_g_color", "u_g_colour"],
    "g_r": ["g-r", "g_minus_r", "g_r_color", "g_r_colour"],
    "r_i": ["r-i", "r_minus_i", "r_i_color", "r_i_colour"],
    "i_z": ["i-z", "i_minus_z", "i_z_color", "i_z_colour"],
}

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip header whitespace/BOM and map common column aliases to canonical names."""
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    canonical_by_lower = {name.lower(): name for name in COLUMN_ALIASES}
    # Normalize casing for canonical headers first (e.g. MAG_U -> mag_u).
    case_rename = {c: canonical_by_lower[str(c).lower()] for c in df.columns
                   if str(c).lower() in canonical_by_lower and c != canonical_by_lower[str(c).lower()]}
    if case_rename:
        df = df.rename(columns=case_rename)
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    existing_lower = {str(c).strip().lower() for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical.lower() in existing_lower:
            continue
        for alias in aliases:
            original = lower_map.get(alias.lower())
            if original is not None and canonical not in df.columns and original not in rename:
                rename[original] = canonical
                break
    if rename:
        df = df.rename(columns=rename)
    return df

def validate_photometry_input(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate that the CSV has enough recognized astronomy features to score."""
    df = standardize_column_names(df)
    df = add_colour_features(df)
    feature_candidates = ["u_g", "g_r", "r_i", "i_z", "mag_g", "z"]
    usable = []
    for col in feature_candidates:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().sum() >= 2 and values.nunique(dropna=True) > 1:
                usable.append(col)
    if len(df) < 5:
        return False, "Please upload at least 5 rows. Very small datasets cannot support meaningful anomaly ranking."
    if len(usable) < 2:
        expected = "mag_u, mag_g, mag_r, mag_i, mag_z and/or colour columns u_g, g_r, r_i, i_z; redshift z is optional."
        return False, f"I could not identify at least two usable photometry features. Expected {expected}"
    return True, f"Recognized {len(usable)} usable model features: {', '.join(usable)}."

def find_column(df: pd.DataFrame, options):
    lower_map = {str(c).lower(): c for c in df.columns}
    for option in options:
        if option.lower() in lower_map:
            return lower_map[option.lower()]
    return None


def ensure_candidate_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "candidate_row" not in df.columns:
        # Internal row number only. Do not treat a low-precision objID as unique.
        df["candidate_row"] = range(1, len(df) + 1)
    return df


def normalize_class(series: pd.Series) -> pd.Series:
    return series.fillna("UNKNOWN").astype(str).str.upper().str.strip()


def quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "dq_extreme_color" not in df.columns:
        color_cols = [c for c in ["u_g", "g_r", "r_i", "i_z"] if c in df.columns]
        if color_cols:
            df["dq_extreme_color"] = df[color_cols].apply(pd.to_numeric, errors="coerce").abs().gt(4).any(axis=1)
        else:
            df["dq_extreme_color"] = False
    if "dq_negative_z" not in df.columns:
        if "z" in df.columns:
            df["dq_negative_z"] = pd.to_numeric(df["z"], errors="coerce") < 0
        else:
            df["dq_negative_z"] = False
    return df


def get_percentile_column(df: pd.DataFrame):
    return find_column(df, ["class_percentile", "anomaly_percentile", "percentile", "anomaly_rank_percentile"])


def get_score_column(df: pd.DataFrame):
    return find_column(df, ["anomaly_score", "score", "anomaly"])


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Run a per-class Isolation Forest on recognized photometry features."""
    df = standardize_column_names(df)
    valid, message = validate_photometry_input(df)
    if not valid:
        raise ValueError(message)
    df = add_colour_features(df)
    df = ensure_candidate_ids(df)
    df = quality_flags(df)
    class_col = find_column(df, ["class", "object_class", "type"])
    if class_col is None:
        df["class"] = "ALL"
        class_col = "class"
    else:
        df["class"] = normalize_class(df[class_col])

    feature_candidates = ["u_g", "g_r", "r_i", "i_z", "mag_g", "z"]
    features = [c for c in feature_candidates if c in df.columns]
    if len(features) < 2:
        raise ValueError("At least two supported numeric features are needed. Provide SDSS magnitudes (mag_u, mag_g, mag_r, mag_i, mag_z) and/or colour columns (u_g, g_r, r_i, i_z).")

    # Keep model input numeric and replace missing/infinite values with per-column medians.
    matrix = df[features].apply(pd.to_numeric, errors="coerce").replace([float("inf"), -float("inf")], float("nan"))
    matrix = matrix.astype("float64")
    for col in matrix.columns:
        median = matrix[col].median()
        matrix[col] = matrix[col].fillna(0.0 if pd.isna(median) else median)

    scores = pd.Series(index=df.index, dtype="float64")
    percentiles = pd.Series(index=df.index, dtype="float64")
    for label, idx in df.groupby("class", dropna=False).groups.items():
        subset = matrix.loc[idx]
        if len(subset) < 5:
            # A tiny group is too small for a meaningful forest; still allow the app to run.
            raw = pd.Series(0.0, index=idx)
        else:
            model = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
            model.fit(subset)
            raw = pd.Series(-model.score_samples(subset), index=idx)
        scores.loc[idx] = raw
        percentiles.loc[idx] = raw.rank(pct=True) * 100.0

    df["anomaly_score"] = scores
    df["class_percentile"] = percentiles
    df["model_features"] = ", ".join(features)
    return df.sort_values("class_percentile", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_sdss_image(ra: float, dec: float, size: int = 512) -> bytes | None:
    """Fetch a genuine SDSS image cutout. Returns None when unavailable."""
    params = urlencode({"ra": float(ra), "dec": float(dec), "scale": 0.4,
                        "width": int(size), "height": int(size), "opt": "G"})
    request = Request(f"{SDSS_BASE}?{params}", headers={"User-Agent": "AstroSignal/1.0 (astronomy research prototype)"})
    try:
        with urlopen(request, timeout=8) as response:
            content_type = response.headers.get("Content-Type", "")
            content = response.read()
            if "image" in content_type.lower() and content:
                return content
    except Exception:
        return None
    return None


def sdss_explore_url(ra, dec):
    return f"{SDSS_EXPLORE}?{urlencode({'ra': float(ra), 'dec': float(dec)})}"


def metric_card(label, value, note=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def _valid_plot_rows(df, columns):
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            return out.iloc[0:0]
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([float("inf"), -float("inf")], float("nan")).dropna(subset=columns)
    # Exclude common sentinels and obviously implausible magnitudes from plotting.
    for col in columns:
        out = out[out[col].abs() < 9000]
        if col.startswith("mag_"):
            out = out[out[col].between(-10, 30)]
    return out


def _add_hover_fields(fig, df):
    hover = [c for c in ["candidate_row", "objID", "class", "mag_u", "mag_g", "mag_r", "mag_i", "mag_z", "u_g", "g_r", "r_i", "i_z", "anomaly_score", "class_percentile"] if c in df.columns]
    # Callers supply hover_data to traces where possible; this helper standardizes layout only.
    fig.update_layout(template=PLOTLY_TEMPLATE, **PLOT_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,.06)", zeroline=False, showline=True, linecolor="rgba(190,210,240,.22)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,.06)", zeroline=False, showline=True, linecolor="rgba(190,210,240,.22)")
    return fig


def _scatter_class_layers(fig, plot_df, x, y, selected_candidate=None):
    """Normal points first, top-percentile candidates last with orange outline."""
    if "class" not in plot_df:
        plot_df = plot_df.copy(); plot_df["class"] = "ALL"
    percentile = pd.to_numeric(plot_df.get("class_percentile", pd.Series(0, index=plot_df.index)), errors="coerce").fillna(0)
    hover_cols = [c for c in ["candidate_row", "objID", "class", "mag_u", "mag_g", "mag_r", "mag_i", "mag_z", "u_g", "g_r", "r_i", "i_z", "anomaly_score", "class_percentile"] if c in plot_df.columns]
    for label in sorted(plot_df["class"].astype(str).unique()):
        subset = plot_df[plot_df["class"].astype(str) == label]
        norm = subset[pd.to_numeric(subset.get("class_percentile", pd.Series(0,index=subset.index)), errors="coerce").fillna(0) < 99]
        anom = subset[pd.to_numeric(subset.get("class_percentile", pd.Series(0,index=subset.index)), errors="coerce").fillna(0) >= 99]
        color = CLASS_COLORS.get(label.upper(), "#9aa9bf")
        if not norm.empty:
            fig.add_trace(go.Scattergl(x=norm[x], y=norm[y], mode="markers", name=label,
                marker=dict(size=4, color=color, opacity=.30), customdata=norm[hover_cols].astype(str).values,
                hovertemplate="<br>".join([f"{c}: %{{customdata[{i}]}}" for i,c in enumerate(hover_cols)])+"<extra></extra>"))
        if not anom.empty:
            fig.add_trace(go.Scattergl(x=anom[x], y=anom[y], mode="markers", name=f"{label} · ≥99th %ile",
                marker=dict(size=9, color=color, opacity=.95, line=dict(color=ANOMALY_COLOR, width=1.7)),
                customdata=anom[hover_cols].astype(str).values,
                hovertemplate="<b>Top-percentile candidate</b><br>"+"<br>".join([f"{c}: %{{customdata[{i}]}}" for i,c in enumerate(hover_cols)])+"<extra></extra>"))
    return fig


def _zoom_controls(df, x, y, key):
    valid = _valid_plot_rows(df, [x,y])
    if valid.empty: return
    xlo, xhi = valid[x].quantile([.01,.99]); ylo, yhi = valid[y].quantile([.01,.99])
    xp=max((xhi-xlo)*.08, .05); yp=max((yhi-ylo)*.08, .05)
    show_extremes = st.checkbox("Show extreme values outside the 1st–99th percentile view", value=False, key=f"extremes_{key}")
    return None if show_extremes else (xlo-xp, xhi+xp, ylo-yp, yhi+yp)


def make_colour_colour_chart(df: pd.DataFrame, height=420):
    if not {"g_r", "r_i"}.issubset(df.columns):
        st.info("Colour–colour chart needs g-r and r-i; provide those indices or compatible magnitudes."); return
    plot_df = _valid_plot_rows(df, ["g_r", "r_i"])
    if plot_df.empty: st.info("No valid g-r / r-i measurements to plot."); return
    limits = _zoom_controls(plot_df, "g_r", "r_i", "cc")
    fig = go.Figure(); _scatter_class_layers(fig, plot_df, "g_r", "r_i")
    fig.update_layout(title=dict(text="Colour–colour space · g−r vs r−i", x=0, xanchor="left"), height=height, xaxis_title="g − r (magnitude)", yaxis_title="r − i (magnitude)")
    if limits: fig.update_xaxes(range=limits[:2]); fig.update_yaxes(range=limits[2:])
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})
    st.caption("How to read this: each dot is an object; classes have consistent colors and orange rings mark objects at or above the 99th within-class percentile.")


def make_score_histogram(df: pd.DataFrame, height=420):
    score_col = get_score_column(df)
    if score_col is None:
        st.info("Anomaly scores are not available for this dataset."); return
    work = df.copy(); work[score_col] = pd.to_numeric(work[score_col], errors="coerce"); work = work.dropna(subset=[score_col])
    if work.empty: st.info("No numeric anomaly scores are available."); return
    fig = go.Figure()
    for label in sorted(work.get("class", pd.Series("ALL",index=work.index)).astype(str).unique()):
        vals=work.loc[work.get("class", pd.Series("ALL",index=work.index)).astype(str)==label, score_col]
        if len(vals): fig.add_trace(go.Histogram(x=vals, name=label, opacity=.58, marker_color=CLASS_COLORS.get(label.upper(),"#9aa9bf"), nbinsx=35))
    threshold = pd.to_numeric(work.loc[pd.to_numeric(work.get("class_percentile", pd.Series(0,index=work.index)),errors="coerce").ge(99), score_col], errors="coerce")
    if not threshold.empty: fig.add_vline(x=float(threshold.min()), line_dash="dash", line_color=ANOMALY_COLOR, annotation_text="99th-percentile review boundary", annotation_position="top right")
    fig.update_layout(title=dict(text="Anomaly-score distribution by class",x=0), barmode="overlay", height=height, xaxis_title="Isolation Forest anomaly score (higher = more unusual)", yaxis_title="Objects")
    _add_hover_fields(fig, work); st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})
    st.caption("How to read this: score shapes show how model scores vary by class; the dashed line is a review boundary estimated from the top-percentile candidates, not a probability cutoff.")


def make_pca_chart(df: pd.DataFrame, height=420):
    features=[c for c in ["u_g","g_r","r_i","i_z","mag_g","z"] if c in df.columns]
    if len(features)<2 or len(df)<3: st.info("PCA needs at least two supported features and three rows."); return
    matrix=df[features].apply(pd.to_numeric,errors="coerce").replace([float("inf"),-float("inf")],float("nan")); matrix=matrix.dropna(axis=1,how="all"); matrix=matrix.loc[:,matrix.nunique(dropna=True)>1]
    if matrix.shape[1]<2: st.info("PCA needs two varying numeric features."); return
    for col in matrix: matrix[col]=matrix[col].fillna(matrix[col].median())
    if len(matrix)>10000: matrix=matrix.sample(10000,random_state=42)
    scaled=StandardScaler().fit_transform(matrix); model=PCA(n_components=2,random_state=42); projected=model.fit_transform(scaled)
    plot_df=df.loc[matrix.index].copy(); plot_df["PC1"]=projected[:,0]; plot_df["PC2"]=projected[:,1]
    fig=go.Figure(); _scatter_class_layers(fig,plot_df,"PC1","PC2")
    fig.update_layout(title=dict(text="PCA projection · photometry patterns",x=0),height=height,xaxis_title=f"PC1 ({model.explained_variance_ratio_[0]*100:.1f}% variance)",yaxis_title=f"PC2 ({model.explained_variance_ratio_[1]*100:.1f}% variance)")
    _add_hover_fields(fig,plot_df); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    st.caption("How to read this: PCA compresses the selected features into two axes; visible separation is exploratory only and is not proof of a physical class or discovery.")


def make_sky_position_chart(df: pd.DataFrame, height=300, selected_candidate=None):
    ra_col=find_column(df,["ra","right_ascension","ra_deg"]); dec_col=find_column(df,["dec","declination","dec_deg"])
    if ra_col is None or dec_col is None: st.info("Sky-position chart requires RA and Dec columns."); return
    plot_df=df.copy(); plot_df["_ra"]=pd.to_numeric(plot_df[ra_col],errors="coerce"); plot_df["_dec"]=pd.to_numeric(plot_df[dec_col],errors="coerce")
    plot_df=plot_df[plot_df["_ra"].between(0,360)&plot_df["_dec"].between(-90,90)].dropna(subset=["_ra","_dec"])
    if plot_df.empty: st.info("No valid RA/Dec pairs found."); return
    fig=go.Figure(); p=pd.to_numeric(plot_df.get("class_percentile",pd.Series(0,index=plot_df.index)),errors="coerce").fillna(0)
    hover=[c for c in ["candidate_row","objID","class","_ra","_dec","anomaly_score","class_percentile"] if c in plot_df.columns]
    fig.add_trace(go.Scattergl(x=plot_df["_ra"],y=plot_df["_dec"],mode="markers",name="Catalogue objects",marker=dict(size=4,color=p,colorscale="Viridis",cmin=0,cmax=100,opacity=.8,colorbar=dict(title="Within-class<br>percentile")),customdata=plot_df[hover].astype(str).values,hovertemplate="<br>".join([f"{c}: %{{customdata[{i}]}}" for i,c in enumerate(hover)])+"<extra></extra>"))
    if selected_candidate is not None and "candidate_row" in plot_df:
        chosen=plot_df[plot_df["candidate_row"].astype(str)==str(selected_candidate)]
        if not chosen.empty: fig.add_trace(go.Scatter(x=chosen["_ra"],y=chosen["_dec"],mode="markers",name="Selected candidate",marker=dict(size=15,color="rgba(0,0,0,0)",line=dict(color=ANOMALY_COLOR,width=2.5))))
    fig.update_layout(title=dict(text="Sky position · catalogue footprint",x=0),height=height,xaxis_title="Right ascension (degrees; reversed)",yaxis_title="Declination (degrees)")
    fig.update_xaxes(autorange="reversed",constrain="domain"); _add_hover_fields(fig,plot_df); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    st.caption("How to read this: points show catalogue coordinates, not image detections; RA runs right-to-left by astronomy convention. Spatial patterns can reflect survey coverage.")


def make_colour_magnitude_chart(df: pd.DataFrame, height=420):
    if not {"g_r","mag_g"}.issubset(df.columns): st.info("Colour–magnitude chart needs g-r and g magnitude."); return
    plot_df=_valid_plot_rows(df,["g_r","mag_g"])
    if plot_df.empty: st.info("No valid colour/magnitude measurements to plot."); return
    limits=_zoom_controls(plot_df,"g_r","mag_g","cm")
    normal=plot_df[pd.to_numeric(plot_df.get("class_percentile",pd.Series(0,index=plot_df.index)),errors="coerce").fillna(0)<99]
    anomalies=plot_df[pd.to_numeric(plot_df.get("class_percentile",pd.Series(0,index=plot_df.index)),errors="coerce").fillna(0)>=99]
    fig=go.Figure()
    if len(normal): fig.add_trace(go.Histogram2d(x=normal["g_r"],y=normal["mag_g"],colorscale=[[0,"rgba(13,20,37,0)"],[.2,"#244d77"],[.55,"#347eaa"],[1,"#55e6f2"]],nbinsx=45,nbinsy=45,showscale=False,opacity=.85,name="Normal density",hovertemplate="g-r: %{x:.3f}<br>g magnitude: %{y:.3f}<extra>Normal density</extra>"))
    hover=[c for c in ["candidate_row","objID","class","mag_u","mag_g","mag_r","mag_i","mag_z","g_r","anomaly_score","class_percentile"] if c in anomalies.columns]
    if len(anomalies): fig.add_trace(go.Scattergl(x=anomalies["g_r"],y=anomalies["mag_g"],mode="markers",name="≥99th percentile",marker=dict(size=8,color=ANOMALY_COLOR,line=dict(color="#ffe1c4",width=.7)),customdata=anomalies[hover].astype(str).values,hovertemplate="<br>".join([f"{c}: %{{customdata[{i}]}}" for i,c in enumerate(hover)])+"<extra></extra>"))
    fig.update_layout(title=dict(text="Colour–magnitude diagram · g−r vs g",x=0),height=height,xaxis_title="g − r (magnitude)",yaxis_title="g magnitude (brighter at top)")
    if limits: fig.update_xaxes(range=limits[:2]); fig.update_yaxes(range=limits[2:])
    fig.update_yaxes(autorange="reversed"); _add_hover_fields(fig,plot_df); st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False})
    st.caption("How to read this: denser background regions contain more ordinary objects; orange points are the top-percentile candidates. Smaller g magnitudes are brighter.")

def show_candidate_inspector(row: pd.Series, source_label="SDSS DR17"):
    st.markdown('<div class="section-kicker">Candidate inspector</div>', unsafe_allow_html=True)
    label = row.get("candidate_row", row.name)
    object_class = str(row.get("class", "Unknown"))
    percentile = pd.to_numeric(pd.Series([row.get("class_percentile")]), errors="coerce").iloc[0]
    score = pd.to_numeric(pd.Series([row.get("anomaly_score")]), errors="coerce").iloc[0]
    st.subheader(f"Candidate #{label} · {object_class}")
    ra = pd.to_numeric(pd.Series([row.get("ra")]), errors="coerce").iloc[0] if "ra" in row.index else float("nan")
    dec = pd.to_numeric(pd.Series([row.get("dec")]), errors="coerce").iloc[0] if "dec" in row.index else float("nan")

    left, right = st.columns([1.08, 1])
    with left:
        if pd.notna(ra) and pd.notna(dec):
            with st.spinner("Requesting image cutout from SDSS SkyServer…"):
                image_bytes = fetch_sdss_image(float(ra), float(dec))
            if image_bytes:
                st.image(BytesIO(image_bytes), caption="Image cutout · Source: SDSS SkyServer DR17", use_container_width=True)
            else:
                st.warning("An SDSS image cutout could not be retrieved for these coordinates. The object may be outside available imaging coverage or the service may be temporarily unavailable.")
            st.markdown(f"[Open this position in SDSS SkyServer ↗]({sdss_explore_url(ra, dec)})")
            st.caption(f"RA {ra:.6f}° · Dec {dec:.6f}°")
        else:
            st.info("Coordinates (RA and Dec) are not present for this row, so an SDSS image cannot be requested.")
    with right:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Class percentile", f"{percentile:.2f}" if pd.notna(percentile) else "—", help="Rank within this object's class; not a probability.")
        with m2:
            st.metric("Anomaly score", f"{score:.4f}" if pd.notna(score) else "—", help="Higher scores indicate greater unusualness under this model.")
        st.markdown("**Photometry and measurements**")
        preferred = ["mag_u", "mag_g", "mag_r", "mag_i", "mag_z", "u_g", "g_r", "r_i", "i_z", "z"]
        measurements = {c: row.get(c) for c in preferred if c in row.index and pd.notna(row.get(c))}
        if measurements:
            mag_keys = [k for k in ["mag_u", "mag_g", "mag_r", "mag_i", "mag_z"] if k in measurements]
            mag_vals = pd.to_numeric(pd.Series([measurements[k] for k in mag_keys]), errors="coerce") if mag_keys else pd.Series(dtype=float)
            suspicious = bool(len(mag_vals.dropna()) >= 3 and ((mag_vals.max() - mag_vals.min()) > 8))
            measure_df = pd.DataFrame({"Measurement": list(measurements.keys()), "Value": [measurements[k] for k in measurements]})
            if suspicious:
                st.markdown('<span class="warning-badge">⚠ Check photometry: unusually wide magnitude spread</span>', unsafe_allow_html=True)
            st.dataframe(measure_df, hide_index=True, use_container_width=True)
            if mag_keys and st.checkbox("Compare this object's magnitudes with its class median", key=f"mag_compare_{label}_{source_label}"):
                source = st.session_state.get("uploaded_analysis") if source_label == "Uploaded CSV" else None
                if source is None:
                    try: source = base_df
                    except NameError: source = None
                if source is not None and "class" in source:
                    class_rows = source[source["class"].astype(str) == object_class]
                    med = [pd.to_numeric(class_rows.get(k, pd.Series(dtype=float)), errors="coerce").median() for k in mag_keys]
                    figm = go.Figure()
                    figm.add_trace(go.Scatter(x=mag_keys, y=[measurements[k] for k in mag_keys], mode="lines+markers", name="Selected object", line=dict(color=ANOMALY_COLOR,width=2), marker=dict(size=8)))
                    if len(med) == len(mag_keys) and pd.Series(med).notna().any(): figm.add_trace(go.Scatter(x=mag_keys,y=med,mode="lines+markers",name=f"{object_class} median",line=dict(color=THEME["cyan"],dash="dash"),marker=dict(size=6)))
                    figm.update_layout(title="u-g-r-i-z magnitude profile",height=300,xaxis_title="Band",yaxis_title="Magnitude (lower = brighter)",yaxis_autorange="reversed")
                    _add_hover_fields(figm, pd.DataFrame()); st.plotly_chart(figm,use_container_width=True)
        else:
            st.caption("No supported photometry columns found for this row.")
        warnings = []
        if bool(row.get("dq_extreme_color", False)):
            warnings.append("One or more colour indices exceed |4|; inspect photometry carefully.")
        if bool(row.get("dq_negative_z", False)):
            warnings.append("Redshift z is negative; verify how this value was measured or encoded.")
        if warnings:
            st.warning("**Quality-control notes**\n\n" + "\n\n".join(warnings))
        else:
            st.info("No built-in quality-control flag is recorded for this row. This does not establish that its measurements are reliable.")
        st.caption("Interpretation: a high score is a review priority, not a confirmed astronomical discovery. Check the source data and literature before making scientific claims.")


# Load the saved, pre-ranked SDSS table when present.
@st.cache_data(show_spinner=False)
def load_default_data(path_string: str, modified_time: float):
    df = standardize_column_names(pd.read_csv(path_string))
    df = add_colour_features(df)
    df = ensure_candidate_ids(df)
    df = quality_flags(df)
    class_col = find_column(df, ["class", "object_class", "type"])
    if class_col:
        df["class"] = normalize_class(df[class_col])
    return df


if "uploaded_analysis" not in st.session_state:
    st.session_state.uploaded_analysis = None

with st.sidebar:
    st.markdown("<div class='eyebrow'>✦ AstroSignal</div>", unsafe_allow_html=True)
    st.caption("Astronomical anomaly exploration")
    page = st.radio("Navigate", ["Mission overview", "Discover candidates", "Upload & analyze"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Project status**")
    st.caption("Prototype · SDSS photometry · Isolation Forest")
    st.markdown("---")
    st.caption("Built to prioritize objects for human investigation — not to certify discoveries.")


def render_header():
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Survey data · Machine learning · Human investigation</div>
          <h1>Find the unusual.<br>Investigate the evidence.</h1>
          <p>AstroSignal ranks astronomical objects whose photometric measurements look unusual relative to similar objects. Explore the measurements, inspect genuine survey images when available, and follow each candidate back to SDSS.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


base_df = None
load_error = None
if DATA_FILE.exists():
    try:
        base_df = load_default_data(str(DATA_FILE), DATA_FILE.stat().st_mtime)
    except Exception as exc:
        load_error = str(exc)

if page == "Mission overview":
    render_header()
    if base_df is None:
        st.warning("The saved candidate table was not found beside app.py. Upload a CSV in **Upload & analyze** to begin.")
        if load_error:
            st.caption(f"Load detail: {load_error}")
    else:
        rows = len(base_df)
        class_col = find_column(base_df, ["class"])
        n_classes = base_df[class_col].nunique() if class_col else "—"
        percentile_col = get_percentile_column(base_df)
        n_review = int(pd.to_numeric(base_df[percentile_col], errors="coerce").ge(99).sum()) if percentile_col else "—"
        flags = int((base_df["dq_extreme_color"].astype(bool) | base_df["dq_negative_z"].astype(bool)).sum())
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Objects in table", f"{rows:,}", "Saved SDSS photometry sample")
        with c2: metric_card("Object classes", str(n_classes), "Classes represented in data")
        with c3: metric_card("Top-percentile candidates", f"{n_review:,}" if isinstance(n_review, int) else str(n_review), "At or above 99th percentile")
        with c4: metric_card("QC flags", f"{flags:,}", "Review indicators, not automatic rejections")
        st.markdown("### Featured candidates")
        st.markdown('<p class="small-muted">A short shortlist keeps the starting view focused. Open Discover to filter and inspect the full table.</p>', unsafe_allow_html=True)
        if percentile_col:
            featured = base_df.copy()
            featured[percentile_col] = pd.to_numeric(featured[percentile_col], errors="coerce")
            featured = featured.sort_values(percentile_col, ascending=False).head(5)
            display_cols = [c for c in ["candidate_row", "class", "class_percentile", "anomaly_score", "ra", "dec"] if c in featured.columns]
            st.dataframe(featured[display_cols], hide_index=True, use_container_width=True)
            if not featured.empty:
                choice = st.selectbox("Inspect a featured candidate", featured.index, format_func=lambda i: f"Row {featured.loc[i].get('candidate_row', i)} · {featured.loc[i].get('class', 'Unknown')} · percentile {featured.loc[i].get('class_percentile', float('nan')):.2f}")
                show_candidate_inspector(featured.loc[choice])
        else:
            st.info("This table has no saved anomaly percentile. Use Upload & analyze to generate scores.")
    st.markdown("---")
    st.markdown("### How to read AstroSignal")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**01 · Rank**")
        st.write("The model finds objects that differ from others in their class across the selected photometric features.")
    with b:
        st.markdown("**02 · Inspect**")
        st.write("Use colour–colour plots, measurements, and available SDSS cutouts to understand why an object stands out.")
    with c:
        st.markdown("**03 · Verify**")
        st.write("Follow the SDSS link and check data quality before treating any candidate as scientifically interesting.")

elif page == "Discover candidates":
    render_header()
    if base_df is None:
        st.warning("The saved ranked CSV could not be loaded. Please add `astrosignal_ranked_candidates.csv` beside app.py, or use Upload & analyze.")
        if load_error:
            st.caption(load_error)
    else:
        df = base_df.copy()
        pcol = get_percentile_column(df)
        scol = get_score_column(df)
        if pcol and pcol != "class_percentile":
            df["class_percentile"] = pd.to_numeric(df[pcol], errors="coerce")
        if scol and scol != "anomaly_score":
            df["anomaly_score"] = pd.to_numeric(df[scol], errors="coerce")
        st.markdown("### Candidate search")
        with st.container(border=True):
            f1, f2, f3 = st.columns([1, 1, 1.25])
            classes = sorted(df["class"].dropna().astype(str).unique()) if "class" in df.columns else []
            with f1:
                selected_classes = st.multiselect("Object class", classes, default=classes)
            with f2:
                min_percentile = st.slider("Minimum class percentile", 0, 100, 95, 1) if "class_percentile" in df.columns else 0
            with f3:
                only_flagged = st.checkbox("Only show quality-control flagged objects", value=False)
            search_text = st.text_input("Search candidate row or SDSS object ID", placeholder="e.g. 2240 or an object ID")
        filtered = df.copy()
        if selected_classes and "class" in filtered.columns:
            filtered = filtered[filtered["class"].isin(selected_classes)]
        if "class_percentile" in filtered.columns:
            filtered["class_percentile"] = pd.to_numeric(filtered["class_percentile"], errors="coerce")
            filtered = filtered[filtered["class_percentile"].ge(min_percentile)]
        if only_flagged:
            filtered = filtered[filtered["dq_extreme_color"].astype(bool) | filtered["dq_negative_z"].astype(bool)]
        if search_text.strip():
            search_cols = [c for c in ["candidate_row", "objID", "specObjID"] if c in filtered.columns]
            if search_cols:
                mask = pd.Series(False, index=filtered.index)
                for col in search_cols:
                    mask |= filtered[col].astype(str).str.contains(search_text.strip(), case=False, na=False, regex=False)
                filtered = filtered[mask]
        sort_col = "class_percentile" if "class_percentile" in filtered.columns else ("anomaly_score" if "anomaly_score" in filtered.columns else None)
        if sort_col:
            filtered = filtered.sort_values(sort_col, ascending=False)
        st.caption(f"Showing {len(filtered):,} of {len(df):,} objects matching these filters.")
        table_cols = [c for c in ["candidate_row", "class", "class_percentile", "anomaly_score", "ra", "dec", "dq_extreme_color", "dq_negative_z"] if c in filtered.columns]
        st.dataframe(filtered[table_cols].head(1000), hide_index=True, use_container_width=True)
        if len(filtered) > 1000:
            st.caption("The table is limited to 1,000 visible rows for readability. Narrow the filters to inspect a smaller set.")
        if not filtered.empty:
            options = filtered.index.tolist()
            selected_idx = st.selectbox(
                "Choose a candidate to inspect",
                options,
                format_func=lambda i: f"Row {filtered.loc[i].get('candidate_row', i)} · {filtered.loc[i].get('class', 'Unknown')} · percentile {filtered.loc[i].get('class_percentile', float('nan')):.2f}" if pd.notna(filtered.loc[i].get('class_percentile', float('nan'))) else f"Row {filtered.loc[i].get('candidate_row', i)} · {filtered.loc[i].get('class', 'Unknown')}",
            )
            show_candidate_inspector(filtered.loc[selected_idx])
        else:
            st.info("No objects match these filters. Try lowering the percentile threshold or changing the class selection.")
        st.markdown("---")
        st.markdown("### Scientific visual exploration")
        st.caption("Use the original astronomy measurements alongside model rankings. Colour indicates relative anomaly percentile where available; it is not a discovery probability.")
        chart1, chart2 = st.columns(2)
        with chart1:
            make_colour_colour_chart(df)
        with chart2:
            make_score_histogram(df)
        chart3, chart4 = st.columns(2)
        with chart3:
            make_colour_magnitude_chart(df)
        with chart4:
            make_pca_chart(df)
        make_sky_position_chart(df, selected_candidate=filtered.loc[selected_idx].get("candidate_row") if not filtered.empty else None)
        st.download_button("Download filtered candidates as CSV", data=filtered.to_csv(index=False).encode("utf-8"), file_name="astrosignal_filtered_candidates.csv", mime="text/csv")

else:  # Upload & analyze
    render_header()
    st.markdown("### Analyze another astronomy CSV")
    st.markdown('<div class="info-panel"><b>Supported input</b><br><span class="small-muted">Best supported: SDSS-style photometry with mag_u, mag_g, mag_r, mag_i, mag_z and optional z, class, ra, dec. You may instead provide colour indices u_g, g_r, r_i, i_z. Common aliases such as modelMag_g, psfMag_g, redshift, right_ascension and declination are recognized. Column aliases do not convert units or make a different survey scientifically equivalent.</span></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            raw_uploaded_df = pd.read_csv(uploaded)
            uploaded_df = standardize_column_names(raw_uploaded_df)
            st.write(f"Loaded **{len(uploaded_df):,} rows** and **{len(uploaded_df.columns)} columns**.")
            with st.expander("Preview uploaded data"):
                st.dataframe(uploaded_df.head(10), use_container_width=True)
            st.caption("Headers are automatically cleaned and common astronomy aliases are mapped. If a key column is still not recognized, map it below.")
            canonical_targets = ["ra", "dec", "class", "mag_u", "mag_g", "mag_r", "mag_i", "mag_z", "z", "u_g", "g_r", "r_i", "i_z"]
            missing_targets = [c for c in canonical_targets if c not in uploaded_df.columns]
            with st.expander("Optional: map your CSV column names", expanded=False):
                st.write("Only map columns whose meaning and units match the target. Leave a target unmapped if your file does not contain it.")
                manual_map = {}
                options = ["— leave unmapped —"] + [str(c) for c in uploaded_df.columns]
                cols = st.columns(2)
                for i, target in enumerate(missing_targets):
                    with cols[i % 2]:
                        source = st.selectbox(f"Map a column to `{target}`", options, key=f"map_{uploaded.name}_{target}")
                    if source != "— leave unmapped —":
                        manual_map[source] = target
            if manual_map:
                # Do not overwrite an existing canonical field; an explicit mapping wins only when free.
                rename_map = {source: target for source, target in manual_map.items() if target not in uploaded_df.columns}
                uploaded_df = uploaded_df.rename(columns=rename_map)
                if rename_map:
                    st.info("Applied column mappings: " + ", ".join(f"{src} → {dst}" for src, dst in rename_map.items()))
            valid, validation_message = validate_photometry_input(uploaded_df)
            if valid:
                st.success(validation_message)
            else:
                st.warning(validation_message)
            if st.button("Run anomaly analysis", type="primary", disabled=not valid):
                with st.spinner("Preparing features and fitting Isolation Forest models by object class…"):
                    try:
                        result = score_dataframe(uploaded_df)
                        st.session_state.uploaded_analysis = result
                        st.success("Analysis complete. Scores are relative to this uploaded dataset and selected features.")
                    except Exception as exc:
                        st.error(f"Could not analyze this CSV: {exc}")
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")

    result = st.session_state.uploaded_analysis
    if result is not None:
        st.markdown("---")
        st.markdown("### Analysis results")
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Objects analyzed", f"{len(result):,}")
        with m2: st.metric("Classes", str(result["class"].nunique()) if "class" in result.columns else "1")
        with m3: st.metric("Objects ≥ 99th percentile", f"{int(result['class_percentile'].ge(99).sum()):,}")
        display_cols = [c for c in ["candidate_row", "class", "class_percentile", "anomaly_score", "ra", "dec", "dq_extreme_color", "dq_negative_z"] if c in result.columns]
        st.dataframe(result[display_cols].head(100), hide_index=True, use_container_width=True)
        st.caption("The percentile is a within-class ranking for this uploaded dataset, not a probability. Colour and magnitude diagrams show astronomy measurements; PCA is a compressed exploratory view.")
        col_a, col_b = st.columns(2)
        with col_a: make_colour_colour_chart(result)
        with col_b: make_score_histogram(result)
        col_c, col_d = st.columns(2)
        with col_c: make_colour_magnitude_chart(result)
        with col_d: make_pca_chart(result)
        make_sky_position_chart(result)
        candidate_idx = st.selectbox("Inspect an analyzed candidate", result.index.tolist(), format_func=lambda i: f"Row {result.loc[i].get('candidate_row', i)} · {result.loc[i].get('class', 'Unknown')} · percentile {result.loc[i].get('class_percentile', 0):.2f}")
        show_candidate_inspector(result.loc[candidate_idx], source_label="Uploaded CSV")
        st.download_button("Download full ranked results", data=result.to_csv(index=False).encode("utf-8"), file_name="astrosignal_ranked_candidates.csv", mime="text/csv")
        st.markdown('<p class="small-muted">Model note: a separate Isolation Forest is fit within each class when the class column is available. The percentile is a within-class rank, not a probability or a calibrated comparison between classes. Missing numeric feature values are filled with the feature median for this run.</p>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p class="small-muted">AstroSignal is an exploratory research prototype. Image cutouts are retrieved from SDSS SkyServer when available. Anomaly rankings prioritize review; they do not establish that an object is new, invalid, or scientifically novel.</p>', unsafe_allow_html=True)
