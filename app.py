from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# IMPORT YOUR TEAMMATE'S MULTI-MODEL ENGINE
from anomaly_models import score_all, build_feature_matrix

# -----------------------------------------------------------------------------
# AstroSignal — SDSS-inspired astronomy anomaly exploration prototype
# Immersive mission-control front-end
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AstroSignal | Mission Control",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path(__file__).parent / "astrosignal_ranked_candidates.csv"
SDSS_BASE = "https://skyserver.sdss.org/dr17/SkyServerWS/ImgCutout/getjpeg"
SDSS_EXPLORE = "https://skyserver.sdss.org/dr17/VisualTools/quickobj"

# Shared visual theme
THEME = {
    "bg": "#03050b", "bg2": "#050814", "panel": "#0b1220", "line": "#1c2a44",
    "text": "#eaf4ff", "muted": "#8fa3bf", "cyan": "#00f3ff", "cyan2": "#38cfe0",
    "blue": "#5b8bff", "violet": "#a78bfa", "amber": "#ff9d4d",
}

# -----------------------------------------------------------------------------
# GLOBAL CSS — HUD / mission-control theme
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@500;600;700;800&family=Rajdhani:wght@500;600;700&display=swap');

      :root {
        --bg:#03050b; --bg2:#050814; --panel:rgba(10,17,32,.72);
        --panel-solid:#0b1220; --line:rgba(120,180,255,.18);
        --line-hi:rgba(0,243,255,.55);
        --text:#eaf4ff; --muted:#8fa3bf;
        --cyan:#00f3ff; --cyan2:#38cfe0; --blue:#5b8bff;
        --violet:#a78bfa; --amber:#ff9d4d;
      }

      html, body, [class*="css"] { font-family:'Inter',sans-serif; }
      .stApp { background:transparent; color:var(--text); }

      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(ellipse 1200px 800px at 15% 10%, rgba(60,110,220,.16), transparent 60%),
          radial-gradient(ellipse 1000px 700px at 85% 85%, rgba(140,80,220,.14), transparent 60%),
          radial-gradient(ellipse 900px 600px at 70% 30%, rgba(0,180,220,.10), transparent 65%),
          linear-gradient(160deg, #03050b 0%, #050814 55%, #03050b 100%);
        position:relative;
      }
      [data-testid="stAppViewContainer"]::after {
        content:"";
        position:fixed; top:50%; left:50%;
        width:160vmax; height:160vmax;
        transform:translate(-50%,-50%);
        background:
          radial-gradient(circle, rgba(0,243,255,.055) 0%, transparent 22%),
          conic-gradient(from 0deg,
            rgba(0,243,255,.00) 0deg, rgba(0,243,255,.045) 40deg,
            rgba(167,139,250,.00) 90deg, rgba(167,139,250,.055) 160deg,
            rgba(0,243,255,.00) 220deg, rgba(0,243,255,.045) 300deg,
            rgba(167,139,250,.00) 360deg);
        border-radius:50%;
        z-index:0; pointer-events:none;
        animation:accretionSpin 180s linear infinite;
        opacity:.85;
        filter:blur(28px);
      }
      @keyframes accretionSpin { to { transform:translate(-50%,-50%) rotate(360deg); } }

      @media (prefers-reduced-motion: reduce) {
        [data-testid="stAppViewContainer"]::after { animation:none; }
        * { transition:none !important; animation:none !important; }
      }

      [data-testid="stHeader"] { background:rgba(3,5,11,.7); backdrop-filter:blur(6px); }
      [data-testid="stMainBlockContainer"], [data-testid="stSidebar"] { position:relative; z-index:1; }

      [data-testid="stSidebar"] {
        background:linear-gradient(180deg, rgba(3,6,14,.94), rgba(6,12,24,.94));
        border-right:1px solid var(--line);
        backdrop-filter:blur(18px);
      }
      [data-testid="stSidebar"] * { color:var(--text); }

      h1,h2,h3 { color:#f3f8ff; letter-spacing:-.02em; }
      h1,h2,h3,.eyebrow,.section-kicker,.brand { font-family:'Rajdhani','Orbitron',sans-serif; }
      p, li, label { color:#c8d5e8; }
      hr { border-color:var(--line); }
      a { color:var(--cyan) !important; }
      code { color:var(--cyan); background:rgba(0,243,255,.08); padding:1px 5px; border-radius:4px; }

      .hero {
        position:relative; padding:2rem 2.1rem 1.9rem;
        border:1px solid var(--line-hi);
        border-radius:14px;
        background:linear-gradient(125deg, rgba(14,26,48,.9), rgba(6,12,26,.86));
        backdrop-filter:blur(14px);
        margin-bottom:1.4rem;
        overflow:hidden;
        box-shadow:0 0 0 1px rgba(0,243,255,.06) inset,
                    0 0 40px rgba(0,180,220,.08),
                    0 20px 60px -20px rgba(0,0,0,.9);
      }
      .hero::before, .hero::after {
        content:""; position:absolute; width:26px; height:26px;
        border:2px solid var(--cyan); pointer-events:none;
      }
      .hero::before { top:10px; left:10px; border-right:none; border-bottom:none; }
      .hero::after  { bottom:10px; right:10px; border-left:none; border-top:none; }
      .hero h1 {
        font-family:'Orbitron',sans-serif;
        font-weight:700;
        font-size:clamp(1.7rem,3.1vw,2.7rem);
        line-height:1.05; margin:.7rem 0 .85rem;
        background:linear-gradient(100deg,#eaf6ff 0%,#7de9ff 45%,#c9b5ff 100%);
        -webkit-background-clip:text; background-clip:text; color:transparent;
        letter-spacing:-.015em;
      }
      .hero p { max-width:880px; color:#b9c8de; font-size:.95rem; line-height:1.7; }

      .eyebrow {
        color:var(--cyan); font-family:'Orbitron',sans-serif;
        font-size:.68rem; letter-spacing:.22em; text-transform:uppercase;
        display:inline-flex; align-items:center; gap:.55rem;
      }
      .eyebrow::before {
        content:""; width:8px; height:8px; border-radius:50%;
        background:var(--cyan); box-shadow:0 0 10px var(--cyan);
        animation:pulseDot 2s ease-in-out infinite;
      }
      @keyframes pulseDot { 0%,100%{opacity:1;transform:scale(1);} 50%{opacity:.4;transform:scale(.75);} }

      .section-kicker {
        color:var(--cyan); font-family:'Orbitron',sans-serif;
        text-transform:uppercase; letter-spacing:.16em; font-size:.66rem;
        display:inline-flex; align-items:center; gap:.5rem; margin-bottom:.35rem;
      }
      .section-kicker::before {
        content:""; width:18px; height:1px; background:linear-gradient(90deg,var(--cyan),transparent);
      }

      [data-testid="stMetric"], .metric-card {
        position:relative;
        background:linear-gradient(160deg, rgba(14,24,44,.85), rgba(6,12,26,.75));
        border:1px solid var(--line);
        padding:1rem 1.05rem;
        border-radius:10px;
        backdrop-filter:blur(10px);
        overflow:hidden;
      }
      [data-testid="stMetric"]::before, .metric-card::before {
        content:""; position:absolute; top:0; left:0; width:34px; height:2px;
        background:linear-gradient(90deg,var(--cyan),transparent);
      }
      [data-testid="stMetric"]::after, .metric-card::after {
        content:""; position:absolute; bottom:0; right:0; width:34px; height:2px;
        background:linear-gradient(270deg,var(--violet),transparent);
      }
      .metric-card { min-height:112px; }
      .metric-label {
        color:var(--muted); font-family:'Rajdhani',sans-serif;
        font-size:.74rem; letter-spacing:.14em; text-transform:uppercase;
        display:flex; align-items:center; gap:.45rem;
      }
      .metric-label::before {
        content:""; width:6px; height:6px; border-radius:50%;
        background:var(--cyan); box-shadow:0 0 8px var(--cyan);
      }
      .metric-value {
        color:#f5fbff;
        font-family:'Orbitron',sans-serif; font-weight:600;
        font-size:1.85rem; margin-top:.45rem;
        text-shadow:0 0 22px rgba(0,243,255,.28);
      }
      .metric-note, .small-muted { color:var(--muted); font-size:.78rem; margin-top:.3rem; }

      [data-testid="stMetricLabel"] { color:var(--muted) !important; font-family:'Rajdhani',sans-serif; letter-spacing:.06em; text-transform:uppercase; }
      [data-testid="stMetricValue"] { color:#f5fbff !important; font-family:'Orbitron',sans-serif; }

      .info-panel {
        background:linear-gradient(160deg, rgba(14,24,44,.72), rgba(6,12,26,.68));
        border:1px solid var(--line);
        padding:1rem 1.15rem;
        border-radius:10px;
        backdrop-filter:blur(10px);
      }
      [data-testid="stExpander"] {
        border:1px solid var(--line); border-radius:10px;
        background:rgba(10,17,32,.55); backdrop-filter:blur(8px);
      }

      .stButton button, .stDownloadButton button {
        border-radius:8px;
        border:1px solid var(--line-hi);
        background:linear-gradient(180deg, rgba(0,243,255,.10), rgba(0,243,255,.04));
        color:#eafdff;
        font-family:'Rajdhani',sans-serif;
        font-weight:600; letter-spacing:.05em;
        min-height:2.4rem;
        transition:all .22s ease;
        text-transform:uppercase; font-size:.85rem;
      }
      .stButton button[kind="primary"] {
        background:linear-gradient(180deg, #2ee6f4, #17b7c8);
        border-color:#7df1ff; color:#02121a; font-weight:700;
        box-shadow:0 0 18px rgba(0,243,255,.35);
      }
      .stButton button:hover, .stDownloadButton button:hover {
        border-color:#9af5ff;
        box-shadow:0 0 20px rgba(0,243,255,.35);
        transform:translateY(-1px);
      }
      .stDownloadButton button { background:rgba(0,243,255,.14); color:#f0fdff; font-weight:600; }

      div[data-baseweb="select"] > div,
      .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background:rgba(9,15,28,.9) !important;
        border-color:var(--line) !important;
        color:var(--text) !important;
      }
      [data-testid="stFileUploader"] section {
        background:rgba(9,15,28,.7);
        border:1px dashed rgba(0,243,255,.42);
        border-radius:10px;
      }
      [data-testid="stFileUploader"] section:hover { border-color:var(--cyan); }

      div[data-testid="stDataFrame"] {
        border:1px solid var(--line);
        border-radius:10px; overflow:hidden;
        background:rgba(6,12,26,.6);
      }

      .warning-badge {
        display:inline-block;
        border:1px solid rgba(255,157,77,.55);
        background:rgba(255,157,77,.12);
        color:#ffc38d; border-radius:6px;
        padding:3px 9px; font-size:.72rem;
        font-family:'Rajdhani',sans-serif; letter-spacing:.06em;
      }
      .hud-status {
        display:inline-flex; align-items:center; gap:.4rem;
        border:1px solid var(--line-hi);
        background:rgba(0,243,255,.06);
        padding:4px 10px; border-radius:999px;
        font-family:'Rajdhani',sans-serif; font-size:.72rem;
        letter-spacing:.12em; text-transform:uppercase;
        color:var(--cyan);
      }
      .hud-status::before {
        content:""; width:7px; height:7px; border-radius:50%;
        background:var(--cyan); box-shadow:0 0 8px var(--cyan);
        animation:pulseDot 1.8s ease-in-out infinite;
      }

      .brand {
        font-family:'Orbitron',sans-serif; font-weight:700;
        font-size:1.15rem; letter-spacing:.06em;
        color:#eaf6ff;
        display:flex; align-items:center; gap:.5rem;
      }
      .brand .dot {
        width:9px; height:9px; border-radius:50%;
        background:var(--cyan); box-shadow:0 0 12px var(--cyan);
      }
      .sidebar-hud {
        font-family:'Rajdhani',sans-serif; font-size:.74rem;
        letter-spacing:.12em; color:var(--muted);
        text-transform:uppercase;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Layered Canvas star-field (parallax stars + occasional shooting stars)
# -----------------------------------------------------------------------------
components.html(
    """
    <script>
    (function() {
      const parent = window.parent.document;
      let canvas = parent.getElementById('astro-star-canvas');
      if (!canvas) {
        canvas = parent.createElement('canvas');
        canvas.id = 'astro-star-canvas';
        Object.assign(canvas.style, {
          position:'fixed', inset:'0', width:'100%', height:'100%',
          zIndex:0, pointerEvents:'none'
        });
        parent.body.appendChild(canvas);
      }
      const ctx = canvas.getContext('2d');
      function resize() {
        canvas.width = parent.documentElement.clientWidth;
        canvas.height = parent.documentElement.clientHeight;
      }
      resize();
      parent.addEventListener('resize', resize);

      const layers = [
        { count: 140, speed: 0.018, sizeMin: 0.4, sizeMax: 1.1, alpha: 0.55 },
        { count:  70, speed: 0.045, sizeMin: 0.7, sizeMax: 1.6, alpha: 0.8  },
        { count:  28, speed: 0.085, sizeMin: 1.1, sizeMax: 2.2, alpha: 1.0  }
      ];
      const tintPalette = ['#eaf6ff', '#b8e6ff', '#d9ccff', '#ffd7a8'];
      const stars = [];
      layers.forEach((L, li) => {
        for (let i = 0; i < L.count; i++) {
          stars.push({
            x: Math.random(), y: Math.random(),
            r: L.sizeMin + Math.random() * (L.sizeMax - L.sizeMin),
            a: L.alpha * (0.55 + Math.random() * 0.45),
            s: L.speed * (0.6 + Math.random() * 0.9),
            tw: Math.random() * Math.PI * 2,
            twSpeed: 0.6 + Math.random() * 1.4,
            c: tintPalette[Math.floor(Math.random() * tintPalette.length)],
            layer: li
          });
        }
      });

      const shooters = [];
      function spawnShooter() {
        if (shooters.length < 2 && Math.random() < 0.35) {
          shooters.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height * 0.5,
            vx: 3.5 + Math.random() * 3, vy: 1.4 + Math.random() * 1.2,
            life: 0, max: 90 + Math.random() * 40
          });
        }
      }

      const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      let t0 = performance.now();

      function frame(now) {
        const dt = Math.min(60, now - t0) / 1000;
        t0 = now;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (const s of stars) {
          if (!reduceMotion) {
            s.y += s.s * dt;
            s.tw += s.twSpeed * dt;
          }
          if (s.y > 1) { s.y = 0; s.x = Math.random(); }
          const alpha = s.a * (0.55 + 0.45 * Math.sin(s.tw));
          ctx.beginPath();
          ctx.fillStyle = s.c;
          ctx.globalAlpha = alpha;
          ctx.arc(s.x * canvas.width, s.y * canvas.height, s.r, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.globalAlpha = 1;

        for (let i = shooters.length - 1; i >= 0; i--) {
          const sh = shooters[i];
          sh.x += sh.vx; sh.y += sh.vy; sh.life++;
          const grad = ctx.createLinearGradient(sh.x, sh.y, sh.x - 90, sh.y - 30);
          grad.addColorStop(0, 'rgba(234,246,255,0.95)');
          grad.addColorStop(1, 'rgba(0,243,255,0)');
          ctx.strokeStyle = grad;
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.moveTo(sh.x, sh.y);
          ctx.lineTo(sh.x - 90, sh.y - 30);
          ctx.stroke();
          if (sh.life > sh.max) shooters.splice(i, 1);
        }
        spawnShooter();

        requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
    })();
    </script>
    """,
    height=0,
)

# -----------------------------------------------------------------------------
# Plotly theme
# -----------------------------------------------------------------------------
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(4,9,20,0.35)",
    font=dict(family="Inter, sans-serif", color="#c8d5e8", size=11),
    margin=dict(l=48, r=20, t=72, b=54),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="top", y=-0.22,
                xanchor="left", x=0, title_text="",
                font=dict(family="Rajdhani, sans-serif", size=11)),
    hoverlabel=dict(bgcolor="#0d1728", bordercolor="#1c2a44",
                    font=dict(color="#eaf4ff", family="Inter, sans-serif")),
    title=dict(font=dict(family="Orbitron, sans-serif", size=15, color="#eaf6ff")),
)
CLASS_COLORS = {"STAR":"#5b8bff", "QSO":"#a78bfa", "QUASAR":"#a78bfa",
                "GALAXY":"#22e0bd", "UNKNOWN":"#8fa3bf", "ALL":"#5b8bff"}
ANOMALY_COLOR = "#ff9d4d"
GRID_COLOR = "rgba(120,180,255,.08)"
AXIS_LINE = "rgba(120,180,255,.30)"

PLOTLY_TEMPLATE = go.layout.Template(layout=go.Layout(
    **PLOT_LAYOUT,
    xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
               showline=True, linecolor=AXIS_LINE),
    yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
               showline=True, linecolor=AXIS_LINE),
))
px.defaults.template = PLOTLY_TEMPLATE


# =============================================================================
# BACKEND LOGIC & PREPROCESSING UTILITIES
# =============================================================================
def add_colour_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    pairs = [("u_g", "mag_u", "mag_g"), ("g_r", "mag_g", "mag_r"),
             ("r_i", "mag_r", "mag_i"), ("i_z", "mag_i", "mag_z")]
    for out, left, right in pairs:
        if out not in df.columns and left in df.columns and right in df.columns:
            df[out] = pd.to_numeric(df[left], errors="coerce") - pd.to_numeric(df[right], errors="coerce")
    for col in df.columns:
        if col not in {"class", "objID", "candidate_row", "specObjID"}:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any() or df[col].isna().all():
                df[col] = converted
    return df


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
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    canonical_by_lower = {name.lower(): name for name in COLUMN_ALIASES}
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

def find_column(df: pd.DataFrame, options):
    lower_map = {str(c).lower(): c for c in df.columns}
    for option in options:
        if option.lower() in lower_map:
            return lower_map[option.lower()]
    return None

def ensure_candidate_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "candidate_row" not in df.columns:
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

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Unified preprocessing pipeline for both default data and uploads."""
    df = standardize_column_names(df)
    df = add_colour_features(df)
    df = ensure_candidate_ids(df)
    df = quality_flags(df)
    class_col = find_column(df, ["class", "object_class", "type"])
    if class_col:
        df["class"] = normalize_class(df[class_col])
    else:
        df["class"] = "ALL"
    return df

def get_percentile_column(df: pd.DataFrame):
    return find_column(df, ["class_percentile", "anomaly_percentile", "percentile", "anomaly_rank_percentile"])

def get_score_column(df: pd.DataFrame):
    return find_column(df, ["anomaly_score", "score", "anomaly"])


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_sdss_image(ra: float, dec: float, size: int = 512) -> bytes | None:
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
    for col in columns:
        out = out[out[col].abs() < 9000]
        if col.startswith("mag_"):
            out = out[out[col].between(-10, 30)]
    return out


def _add_hover_fields(fig, df):
    fig.update_layout(template=PLOTLY_TEMPLATE, **PLOT_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                     showline=True, linecolor=AXIS_LINE)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
                     showline=True, linecolor=AXIS_LINE)
    return fig


def _scatter_class_layers(fig, plot_df, x, y, selected_candidate=None):
    if "class" not in plot_df:
        plot_df = plot_df.copy(); plot_df["class"] = "ALL"
    hover_cols = [c for c in ["candidate_row", "objID", "class", "mag_u", "mag_g",
                              "mag_r", "mag_i", "mag_z", "u_g", "g_r", "r_i", "i_z",
                              "anomaly_score", "class_percentile"] if c in plot_df.columns]
    for label in sorted(plot_df["class"].astype(str).unique()):
        subset = plot_df[plot_df["class"].astype(str) == label]
        pct = pd.to_numeric(subset.get("class_percentile",
                                       pd.Series(0, index=subset.index)), errors="coerce").fillna(0)
        norm = subset[pct < 99]
        anom = subset[pct >= 99]
        color = CLASS_COLORS.get(label.upper(), "#8fa3bf")
        if not norm.empty:
            fig.add_trace(go.Scattergl(
                x=norm[x], y=norm[y], mode="markers", name=label,
                marker=dict(size=4, color=color, opacity=.30),
                customdata=norm[hover_cols].astype(str).values,
                hovertemplate="<br>".join([f"{c}: %{{customdata[{i}]}}"
                                            for i, c in enumerate(hover_cols)]) + "<extra></extra>"))
        if not anom.empty:
            fig.add_trace(go.Scattergl(
                x=anom[x], y=anom[y], mode="markers", name=f"{label} · ≥99th %ile",
                marker=dict(size=9, color=color, opacity=.95, line=dict(color=ANOMALY_COLOR, width=1.7)),
                customdata=anom[hover_cols].astype(str).values,
                hovertemplate="<b>Top-percentile candidate</b><br>" + "<br>".join([f"{c}: %{{customdata[{i}]}}"
                                                                                    for i, c in enumerate(hover_cols)]) + "<extra></extra>"))
    return fig


def _zoom_controls(df, x, y, key):
    valid = _valid_plot_rows(df, [x, y])
    if valid.empty: return None
    xlo, xhi = valid[x].quantile([.01, .99]); ylo, yhi = valid[y].quantile([.01, .99])
    xp = max((xhi - xlo) * .08, .05); yp = max((yhi - ylo) * .08, .05)
    show_extremes = st.checkbox("Show extreme values outside the 1st–99th percentile view", value=False, key=f"extremes_{key}")
    return None if show_extremes else (xlo - xp, xhi + xp, ylo - yp, yhi + yp)


def make_colour_colour_chart(df: pd.DataFrame, height=420):
    if not {"g_r", "r_i"}.issubset(df.columns):
        st.info("Colour–colour chart needs g-r and r-i; provide those indices or compatible magnitudes.")
        return
    plot_df = _valid_plot_rows(df, ["g_r", "r_i"])
    if plot_df.empty:
        st.info("No valid g-r / r-i measurements to plot.")
        return
    limits = _zoom_controls(plot_df, "g_r", "r_i", "cc")
    fig = go.Figure()
    _scatter_class_layers(fig, plot_df, "g_r", "r_i")
    fig.update_layout(title=dict(text="Colour–colour space · g−r vs r−i", x=0, xanchor="left"), height=height, xaxis_title="g − r (magnitude)", yaxis_title="r − i (magnitude)")
    if limits:
        fig.update_xaxes(range=limits[:2])
        fig.update_yaxes(range=limits[2:])
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.caption("How to read this: each dot is an object; classes have consistent colors and orange rings mark objects at or above the 99th within-class percentile.")


def make_score_histogram(df: pd.DataFrame, height=420):
    score_col = get_score_column(df)
    if score_col is None:
        st.info("Anomaly scores are not available for this dataset.")
        return
    work = df.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=[score_col])
    if work.empty:
        st.info("No numeric anomaly scores are available.")
        return
    fig = go.Figure()
    for label in sorted(work.get("class", pd.Series("ALL", index=work.index)).astype(str).unique()):
        vals = work.loc[work.get("class", pd.Series("ALL", index=work.index)).astype(str) == label, score_col]
        if len(vals):
            fig.add_trace(go.Histogram(x=vals, name=label, opacity=.58, marker_color=CLASS_COLORS.get(label.upper(), "#8fa3bf"), nbinsx=35))
    threshold = pd.to_numeric(work.loc[pd.to_numeric(work.get("class_percentile", pd.Series(0, index=work.index)), errors="coerce").ge(99), score_col], errors="coerce")
    if not threshold.empty:
        fig.add_vline(x=float(threshold.min()), line_dash="dash", line_color=ANOMALY_COLOR, annotation_text="99th-percentile review boundary", annotation_position="top right")
    fig.update_layout(title=dict(text="Anomaly-score distribution by class", x=0), barmode="overlay", height=height, xaxis_title="Anomaly score (higher = more unusual)", yaxis_title="Objects")
    _add_hover_fields(fig, work)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.caption("How to read this: score shapes show how model scores vary by class; the dashed line is a review boundary estimated from the top-percentile candidates, not a probability cutoff.")


def make_pca_chart(df: pd.DataFrame, height=420):
    features = [c for c in ["u_g", "g_r", "r_i", "i_z", "mag_g", "z"] if c in df.columns]
    if len(features) < 2 or len(df) < 3:
        st.info("PCA needs at least two supported features and three rows.")
        return
    matrix = df[features].apply(pd.to_numeric, errors="coerce").replace([float("inf"), -float("inf")], float("nan"))
    matrix = matrix.dropna(axis=1, how="all")
    matrix = matrix.loc[:, matrix.nunique(dropna=True) > 1]
    if matrix.shape[1] < 2:
        st.info("PCA needs two varying numeric features.")
        return
    for col in matrix:
        matrix[col] = matrix[col].fillna(matrix[col].median())
    if len(matrix) > 10000:
        matrix = matrix.sample(10000, random_state=42)
    scaled = StandardScaler().fit_transform(matrix)
    model = PCA(n_components=2, random_state=42)
    projected = model.fit_transform(scaled)
    plot_df = df.loc[matrix.index].copy()
    plot_df["PC1"] = projected[:, 0]
    plot_df["PC2"] = projected[:, 1]
    fig = go.Figure()
    _scatter_class_layers(fig, plot_df, "PC1", "PC2")
    fig.update_layout(title=dict(text="PCA projection · photometry patterns", x=0), height=height, xaxis_title=f"PC1 ({model.explained_variance_ratio_[0]*100:.1f}% variance)", yaxis_title=f"PC2 ({model.explained_variance_ratio_[1]*100:.1f}% variance)")
    _add_hover_fields(fig, plot_df)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.caption("How to read this: PCA compresses the selected features into two axes; visible separation is exploratory only and is not proof of a physical class or discovery.")


def make_sky_position_chart(df: pd.DataFrame, height=300, selected_candidate=None):
    ra_col = find_column(df, ["ra", "right_ascension", "ra_deg"])
    dec_col = find_column(df, ["dec", "declination", "dec_deg"])
    if ra_col is None or dec_col is None:
        st.info("Sky-position chart requires RA and Dec columns.")
        return
    plot_df = df.copy()
    plot_df["_ra"] = pd.to_numeric(plot_df[ra_col], errors="coerce")
    plot_df["_dec"] = pd.to_numeric(plot_df[dec_col], errors="coerce")
    plot_df = plot_df[plot_df["_ra"].between(0, 360) & plot_df["_dec"].between(-90, 90)].dropna(subset=["_ra", "_dec"])
    if plot_df.empty:
        st.info("No valid RA/Dec pairs found.")
        return
    fig = go.Figure()
    p = pd.to_numeric(plot_df.get("class_percentile", pd.Series(0, index=plot_df.index)), errors="coerce").fillna(0)
    hover = [c for c in ["candidate_row", "objID", "class", "_ra", "_dec", "anomaly_score", "class_percentile"] if c in plot_df.columns]
    fig.add_trace(go.Scattergl(x=plot_df["_ra"], y=plot_df["_dec"], mode="markers", name="Catalogue objects", marker=dict(size=4, color=p, colorscale="Viridis", cmin=0, cmax=100, opacity=.8, colorbar=dict(title="Within-class<br>percentile")), customdata=plot_df[hover].astype(str).values, hovertemplate="<br>".join([f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(hover)]) + "<extra></extra>"))
    if selected_candidate is not None and "candidate_row" in plot_df:
        chosen = plot_df[plot_df["candidate_row"].astype(str) == str(selected_candidate)]
        if not chosen.empty:
            fig.add_trace(go.Scatter(x=chosen["_ra"], y=chosen["_dec"], mode="markers", name="Selected candidate", marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color=ANOMALY_COLOR, width=2.5))))
    fig.update_layout(title=dict(text="Sky position · catalogue footprint", x=0), height=height, xaxis_title="Right ascension (degrees; reversed)", yaxis_title="Declination (degrees)")
    fig.update_xaxes(autorange="reversed", constrain="domain")
    _add_hover_fields(fig, plot_df)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.caption("How to read this: points show catalogue coordinates, not image detections; RA runs right-to-left by astronomy convention. Spatial patterns can reflect survey coverage.")


def make_colour_magnitude_chart(df: pd.DataFrame, height=420):
    if not {"g_r", "mag_g"}.issubset(df.columns):
        st.info("Colour–magnitude chart needs g-r and g magnitude.")
        return
    plot_df = _valid_plot_rows(df, ["g_r", "mag_g"])
    if plot_df.empty:
        st.info("No valid colour/magnitude measurements to plot.")
        return
    limits = _zoom_controls(plot_df, "g_r", "mag_g", "cm")
    normal = plot_df[pd.to_numeric(plot_df.get("class_percentile", pd.Series(0, index=plot_df.index)), errors="coerce").fillna(0) < 99]
    anomalies = plot_df[pd.to_numeric(plot_df.get("class_percentile", pd.Series(0, index=plot_df.index)), errors="coerce").fillna(0) >= 99]
    fig = go.Figure()
    if len(normal):
        fig.add_trace(go.Histogram2d(x=normal["g_r"], y=normal["mag_g"], colorscale=[[0, "rgba(10,17,32,0)"], [.2, "#1c3d66"], [.55, "#256699"], [1, "#00f3ff"]], nbinsx=45, nbinsy=45, showscale=False, opacity=.85, name="Normal density", hovertemplate="g-r: %{x:.3f}<br>g magnitude: %{y:.3f}<extra>Normal density</extra>"))
    hover = [c for c in ["candidate_row", "objID", "class", "mag_u", "mag_g", "mag_r", "mag_i", "mag_z", "g_r", "anomaly_score", "class_percentile"] if c in anomalies.columns]
    if len(anomalies):
        fig.add_trace(go.Scattergl(x=anomalies["g_r"], y=anomalies["mag_g"], mode="markers", name="≥99th percentile", marker=dict(size=8, color=ANOMALY_COLOR, line=dict(color="#ffe1c4", width=.7)), customdata=anomalies[hover].astype(str).values, hovertemplate="<br>".join([f"{c}: %{{customdata[{i}]}}" for i, c in enumerate(hover)]) + "<extra></extra>"))
    fig.update_layout(title=dict(text="Colour–magnitude diagram · g−r vs g", x=0), height=height, xaxis_title="g − r (magnitude)", yaxis_title="g magnitude (brighter at top)")
    if limits:
        fig.update_xaxes(range=limits[:2])
        fig.update_yaxes(range=limits[2:])
    fig.update_yaxes(autorange="reversed")
    _add_hover_fields(fig, plot_df)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
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
                    figm.add_trace(go.Scatter(x=mag_keys, y=[measurements[k] for k in mag_keys], mode="lines+markers", name="Selected object", line=dict(color=ANOMALY_COLOR, width=2), marker=dict(size=8)))
                    if len(med) == len(mag_keys) and pd.Series(med).notna().any():
                        figm.add_trace(go.Scatter(x=mag_keys, y=med, mode="lines+markers", name=f"{object_class} median", line=dict(color=THEME["cyan"], dash="dash"), marker=dict(size=6)))
                    figm.update_layout(title="u-g-r-i-z magnitude profile", height=300, xaxis_title="Band", yaxis_title="Magnitude (lower = brighter)", yaxis_autorange="reversed")
                    _add_hover_fields(figm, pd.DataFrame())
                    st.plotly_chart(figm, use_container_width=True)
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


@st.cache_data(show_spinner=False)
def load_default_data(path_string: str, modified_time: float):
    df = pd.read_csv(path_string)
    df = preprocess_dataframe(df)
    
    # Automatically compute multi-model scores using teammate's engine if missing
    if "if_percentile" not in df.columns or "lof_percentile" not in df.columns:
        df = score_all(df)
    return df


if "uploaded_analysis" not in st.session_state:
    st.session_state.uploaded_analysis = None

with st.sidebar:
    st.markdown('<div class="brand"><div class="dot"></div>AstroSignal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-hud">Mission Telemetry v2.4</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate", ["Mission overview", "Discover candidates", "Upload & analyze"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("**Model Control Panel**")
    active_model = st.selectbox(
        "Active Detection Engine",
        ["Isolation Forest", "Local Outlier Factor (LOF)", "One-Class SVM"],
        help="Switch between the three ML anomaly models powering metrics, percentiles, and charts."
    )

    st.markdown("---")
    st.markdown("**Project status**")
    st.caption("Prototype · Multi-Model SDSS Photometry")
    st.markdown("---")
    st.caption("Built to prioritize objects for human investigation — not to certify discoveries.")


def render_header():
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Survey data · Multi-Model AI · Human investigation</div>
          <h1>Find the unusual.<br>Investigate the evidence.</h1>
          <p>AstroSignal ranks astronomical objects whose photometric measurements look unusual. Use the sidebar model selector to switch between Isolation Forest, LOF, and One-Class SVM analytics in real-time.</p>
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

# MAP USER'S MODEL SELECTION TO THE CORRESPONDING COLUMNS
if base_df is not None:
    if "if_percentile" not in base_df.columns:
        base_df = score_all(base_df)
        
    if active_model == "Isolation Forest":
        base_df["anomaly_score"] = base_df["if_score"]
        base_df["class_percentile"] = base_df["if_percentile"]
    elif active_model == "Local Outlier Factor (LOF)":
        base_df["anomaly_score"] = base_df["lof_score"]
        base_df["class_percentile"] = base_df["lof_percentile"]
    else:
        base_df["anomaly_score"] = base_df["ocsvm_score"]
        base_df["class_percentile"] = base_df["ocsvm_percentile"]

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
        with c3: metric_card("Top-percentile candidates", f"{n_review:,}" if isinstance(n_review, int) else str(n_review), f"At/above 99th %ile ({active_model})")
        with c4: metric_card("QC flags", f"{flags:,}", "Review indicators, not automatic rejections")
        st.markdown("### Featured candidates")
        st.markdown(f'<p class="small-muted">Showing top shortlist ranked by <b>{active_model}</b>.</p>', unsafe_allow_html=True)
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
        st.write("Models evaluate objects that differ from others in their class across selected photometric metrics.")
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
        st.markdown(f"### Candidate search ({active_model})")
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
        st.markdown(f'<div class="hud-status">Telemetry Match: {len(filtered):,} / {len(df):,} objects ({active_model})</div>', unsafe_allow_html=True)
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
        st.markdown(f"### Scientific visual exploration ({active_model})")
        st.caption(f"Visualizing results using **{active_model}**. Orange rings highlight objects at or above the 99th percentile.")
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
        st.download_button(f"Download filtered candidates ({active_model}) as CSV", data=filtered.to_csv(index=False).encode("utf-8"), file_name=f"astrosignal_filtered_{active_model.lower().replace(' ', '_')}.csv", mime="text/csv")

else:  # Upload & analyze
    render_header()
    st.markdown("### Analyze another astronomy CSV (Multi-Model Engine)")
    st.markdown('<div class="info-panel"><b>Supported input</b><br><span class="small-muted">Best supported: SDSS-style photometry with mag_u, mag_g, mag_r, mag_i, mag_z and optional z, class, ra, dec. When you run analysis, all three models (Isolation Forest, LOF, and One-Class SVM) will be computed simultaneously.</span></div>', unsafe_allow_html=True)
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
                rename_map = {source: target for source, target in manual_map.items() if target not in uploaded_df.columns}
                uploaded_df = uploaded_df.rename(columns=rename_map)
                if rename_map:
                    st.info("Applied column mappings: " + ", ".join(f"{src} → {dst}" for src, dst in rename_map.items()))
            
            # Basic validation check using helper from teammate's anomaly models framework
            try:
                build_feature_matrix(uploaded_df)
                valid = len(uploaded_df) >= 5
                validation_message = "Recognized valid photometry features for multi-model evaluation." if valid else "Please provide at least 5 rows."
            except Exception as e:
                valid = False
                validation_message = str(e)

            if valid:
                st.success(validation_message)
            else:
                st.warning(validation_message)

            if st.button("Run Multi-Model Analysis (IF, LOF, OCSVM)", type="primary", disabled=not valid):
                with st.spinner("Fitting Isolation Forest, Local Outlier Factor, and One-Class SVM models per class…"):
                    try:
                        # Preprocess uploaded dataset completely using pipeline first
                        processed_df = preprocess_dataframe(uploaded_df)
                        result = score_all(processed_df)
                        st.session_state.uploaded_analysis = result
                        st.success("Multi-model analysis complete! All detectors have scored the dataset.")
                    except Exception as exc:
                        st.error(f"Could not analyze this CSV: {exc}")
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")

    result = st.session_state.uploaded_analysis
    if result is not None:
        # Dynamically map active model view for uploaded results too
        if active_model == "Isolation Forest":
            result["anomaly_score"] = result["if_score"]
            result["class_percentile"] = result["if_percentile"]
        elif active_model == "Local Outlier Factor (LOF)":
            result["anomaly_score"] = result["lof_score"]
            result["class_percentile"] = result["lof_percentile"]
        else:
            result["anomaly_score"] = result["ocsvm_score"]
            result["class_percentile"] = result["ocsvm_percentile"]

        st.markdown("---")
        st.markdown(f"### Analysis results ({active_model})")
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Objects analyzed", f"{len(result):,}")
        with m2: st.metric("Classes", str(result["class"].nunique()) if "class" in result.columns else "1")
        with m3: st.metric("Objects ≥ 99th percentile", f"{int(result['class_percentile'].ge(99).sum()):,}")
        display_cols = [c for c in ["candidate_row", "class", "class_percentile", "anomaly_score", "ra", "dec", "dq_extreme_color", "dq_negative_z"] if c in result.columns]
        st.dataframe(result[display_cols].head(100), hide_index=True, use_container_width=True)
        st.caption("The percentile is a within-class ranking for this uploaded dataset, not a probability.")
        col_a, col_b = st.columns(2)
        with col_a: make_colour_colour_chart(result)
        with col_b: make_score_histogram(result)
        col_c, col_d = st.columns(2)
        with col_c: make_colour_magnitude_chart(result)
        with col_d: make_pca_chart(result)
        make_sky_position_chart(result)
        candidate_idx = st.selectbox("Inspect an analyzed candidate", result.index.tolist(), format_func=lambda i: f"Row {result.loc[i].get('candidate_row', i)} · {result.loc[i].get('class', 'Unknown')} · percentile {result.loc[i].get('class_percentile', 0):.2f}")
        show_candidate_inspector(result.loc[candidate_idx], source_label="Uploaded CSV")
        st.download_button("Download full multi-model results", data=result.to_csv(index=False).encode("utf-8"), file_name="astrosignal_multimodel_ranked.csv", mime="text/csv")
        st.markdown('<p class="small-muted">Model note: models are fit separately within each object class.</p>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p class="small-muted">AstroSignal is an exploratory research prototype. Image cutouts are retrieved from SDSS SkyServer when available. Anomaly rankings prioritize review; they do not establish that an object is new, invalid, or scientifically novel.</p>', unsafe_allow_html=True)
