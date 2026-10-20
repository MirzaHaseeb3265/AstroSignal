from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest

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

# A compact, high-contrast dark theme. Plotly charts use the same background.
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root { --bg:#070b16; --panel:#10182a; --line:#26334b; --muted:#a2b0c8; --text:#edf3ff; --cyan:#6be7f2; }
      .stApp { background: radial-gradient(ellipse at 15% 0%, #14213c 0%, #090f1e 38%, #070b16 76%); color:var(--text); }
      html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
      [data-testid="stHeader"] { background:rgba(7,11,22,.82); }
      [data-testid="stSidebar"] { background:rgba(8,14,27,.98); border-right:1px solid #26334b; }
      [data-testid="stSidebar"] * { color:var(--text); }
      h1,h2,h3 { letter-spacing:-.035em; color:#f1f6ff; }
      p, li, label { color:#d1daeb; }
      .hero { padding:1.7rem 1.8rem; border:1px solid #2b3a57; border-radius:22px;
              background:linear-gradient(120deg,rgba(22,39,69,.96),rgba(11,19,37,.88));
              box-shadow:0 18px 55px rgba(0,0,0,.2); margin-bottom:1.2rem; }
      .eyebrow { color:#6be7f2; font-family:'DM Mono',monospace; font-size:.76rem; letter-spacing:.16em; text-transform:uppercase; }
      .hero h1 { font-size:clamp(2rem,4vw,3.3rem); line-height:1.1; margin:.55rem 0 .7rem; }
      .hero p { max-width:850px; color:#c4d1e7; font-size:1rem; line-height:1.7; }
      .section-kicker { color:#6be7f2; font-family:'DM Mono',monospace; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; }
      .metric-card { background:rgba(16,24,42,.86); border:1px solid #263650; padding:1rem 1.1rem; border-radius:16px; min-height:104px; }
      .metric-label { color:#9caec9; font-size:.78rem; text-transform:uppercase; letter-spacing:.07em; }
      .metric-value { color:#f2f6ff; font-size:1.8rem; font-weight:800; margin-top:.35rem; }
      .metric-note { color:#8ea2c0; font-size:.75rem; margin-top:.1rem; }
      .info-panel { background:rgba(15,24,42,.76); border:1px solid #273751; padding:1rem 1.15rem; border-radius:15px; }
      .small-muted { color:#9eafc8; font-size:.86rem; }
      .stButton button, .stDownloadButton button { border-radius:10px; border:1px solid #4a6b8c; }
      [data-testid="stMetric"] { background:rgba(16,24,42,.78); border:1px solid #263650; padding:13px 15px; border-radius:14px; }
      [data-testid="stMetricLabel"] { color:#a9bad3; }
      [data-testid="stMetricValue"] { color:#edf5ff; }
      div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input { background:#111b2e; }
      div[data-testid="stDataFrame"] { border:1px solid #263650; border-radius:12px; overflow:hidden; }
      hr { border-color:#263650; }
      a { color:#6be7f2 !important; }
      code { color:#83e9f0; }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(9,15,29,0.55)",
    font=dict(family="Manrope, sans-serif", color="#dce8fb"),
    margin=dict(l=15, r=15, t=55, b=15),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
)


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
    """Run a per-class Isolation Forest on the supported photometry features."""
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


def make_colour_colour_chart(df: pd.DataFrame, height=430):
    needed = {"g_r", "r_i"}
    if not needed.issubset(df.columns):
        st.info("The colour–colour diagram needs g-r and r-i. Provide mag_g, mag_r and mag_i, or those colour columns directly.")
        return
    plot_df = df.copy()
    plot_df["g_r"] = pd.to_numeric(plot_df["g_r"], errors="coerce")
    plot_df["r_i"] = pd.to_numeric(plot_df["r_i"], errors="coerce")
    plot_df = plot_df.dropna(subset=["g_r", "r_i"])
    if len(plot_df) > 3500:
        plot_df = plot_df.sample(3500, random_state=42)
    color_col = "class" if "class" in plot_df.columns else None
    if "class_percentile" in plot_df.columns:
        plot_df["Model flag"] = pd.to_numeric(plot_df["class_percentile"], errors="coerce").ge(99)
        fig = px.scatter(plot_df, x="g_r", y="r_i", color="Model flag",
                         symbol=color_col, hover_data=[c for c in ["candidate_row", "class", "class_percentile", "anomaly_score"] if c in plot_df.columns],
                         color_discrete_map={True: "#ffbd69", False: "#57c7e8"}, opacity=0.72,
                         title="Colour–colour space · g−r vs r−i")
    else:
        fig = px.scatter(plot_df, x="g_r", y="r_i", color=color_col,
                         hover_data=[c for c in ["candidate_row", "class", "anomaly_score"] if c in plot_df.columns],
                         opacity=0.72, title="Colour–colour space · g−r vs r−i")
    fig.update_layout(**PLOT_LAYOUT, height=height, xaxis_title="g − r (magnitude)", yaxis_title="r − i (magnitude)")
    fig.update_traces(marker=dict(size=7, line=dict(width=0)))
    st.plotly_chart(fig, use_container_width=True)


def make_score_histogram(df: pd.DataFrame):
    score_col = get_percentile_column(df) or get_score_column(df)
    if score_col is None:
        st.info("An anomaly score/percentile is not available for this dataset yet.")
        return
    values = pd.to_numeric(df[score_col], errors="coerce")
    plot_df = pd.DataFrame({score_col: values}).dropna()
    fig = px.histogram(plot_df, x=score_col, nbins=40, title=f"Distribution of {score_col.replace('_', ' ')}",
                       color_discrete_sequence=["#6be7f2"])
    fig.update_layout(**PLOT_LAYOUT, height=350, xaxis_title=score_col.replace("_", " "), yaxis_title="Objects")
    st.plotly_chart(fig, use_container_width=True)


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
            measure_df = pd.DataFrame({"Measurement": list(measurements.keys()), "Value": [measurements[k] for k in measurements]})
            st.dataframe(measure_df, hide_index=True, use_container_width=True)
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
    df = pd.read_csv(path_string)
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
        chart1, chart2 = st.columns([1.25, 1])
        with chart1:
            make_colour_colour_chart(df)
        with chart2:
            make_score_histogram(df)
        if {"g_r", "mag_g"}.issubset(df.columns):
            cmd = df.copy()
            cmd["g_r"] = pd.to_numeric(cmd["g_r"], errors="coerce")
            cmd["mag_g"] = pd.to_numeric(cmd["mag_g"], errors="coerce")
            cmd = cmd.dropna(subset=["g_r", "mag_g"])
            if len(cmd) > 3500:
                cmd = cmd.sample(3500, random_state=42)
            fig = px.scatter(cmd, x="g_r", y="mag_g", color="class" if "class" in cmd.columns else None,
                             hover_data=[c for c in ["candidate_row", "class_percentile", "anomaly_score"] if c in cmd.columns],
                             opacity=.7, title="Colour–magnitude diagram · g−r vs g")
            # Astronomical magnitudes are conventionally brighter at lower values.
            fig.update_yaxes(autorange="reversed", title="g magnitude (brighter ↑)")
            fig.update_layout(**PLOT_LAYOUT, height=410, xaxis_title="g − r (magnitude)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("These plots show patterns in the supplied photometry. They are exploratory visualizations; survey selection effects and measurement uncertainty still matter.")
        st.download_button("Download filtered candidates as CSV", data=filtered.to_csv(index=False).encode("utf-8"), file_name="astrosignal_filtered_candidates.csv", mime="text/csv")

else:  # Upload & analyze
    render_header()
    st.markdown("### Analyze another SDSS-style CSV")
    st.markdown('<div class="info-panel"><b>Expected columns</b><br><span class="small-muted">Recommended: ra, dec, class, mag_u, mag_g, mag_r, mag_i, mag_z, z. You can also provide the colour columns u_g, g_r, r_i, i_z directly. Missing optional columns are okay; at least two numeric model features are required.</span></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            st.write(f"Loaded **{len(uploaded_df):,} rows** and **{len(uploaded_df.columns)} columns**.")
            with st.expander("Preview uploaded data"):
                st.dataframe(uploaded_df.head(10), use_container_width=True)
            if st.button("Run anomaly analysis", type="primary"):
                with st.spinner("Preparing features and fitting Isolation Forest models by object class…"):
                    try:
                        result = score_dataframe(uploaded_df)
                        st.session_state.uploaded_analysis = result
                        st.success("Analysis complete. Scores are relative to this uploaded dataset and the selected feature set.")
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
        col_a, col_b = st.columns(2)
        with col_a: make_colour_colour_chart(result)
        with col_b: make_score_histogram(result)
        candidate_idx = st.selectbox("Inspect an analyzed candidate", result.index.tolist(), format_func=lambda i: f"Row {result.loc[i].get('candidate_row', i)} · {result.loc[i].get('class', 'Unknown')} · percentile {result.loc[i].get('class_percentile', 0):.2f}")
        show_candidate_inspector(result.loc[candidate_idx], source_label="Uploaded CSV")
        st.download_button("Download full ranked results", data=result.to_csv(index=False).encode("utf-8"), file_name="astrosignal_ranked_candidates.csv", mime="text/csv")
        st.markdown('<p class="small-muted">Model note: a separate Isolation Forest is fit within each class when the class column is available. The percentile is a within-class rank, not a probability or a calibrated comparison between classes. Missing numeric feature values are filled with the feature median for this run.</p>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p class="small-muted">AstroSignal is an exploratory research prototype. Image cutouts are retrieved from SDSS SkyServer when available. Anomaly rankings prioritize review; they do not establish that an object is new, invalid, or scientifically novel.</p>', unsafe_allow_html=True)
