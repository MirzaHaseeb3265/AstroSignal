from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from io import BytesIO
import textwrap

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

      /* ============================================================
         Base gradient + animated nebula / accretion disk background
         ============================================================ */
      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(ellipse 1200px 800px at 15% 10%, rgba(60,110,220,.16), transparent 60%),
          radial-gradient(ellipse 1000px 700px at 85% 85%, rgba(140,80,220,.14), transparent 60%),
          radial-gradient(ellipse 900px 600px at 70% 30%, rgba(0,180,220,.10), transparent 65%),
          linear-gradient(160deg, #03050b 0%, #050814 55%, #03050b 100%);
        position:relative;
      }
      /* Slowly rotating accretion-disk glow, sits under the star canvas */
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

      /* Canvas star-field layer from components.html */
      [data-testid="stAppViewContainer"] .stHtml:first-of-type {
        position:fixed !important; inset:0 !important;
        z-index:0; pointer-events:none;
      }

      @media (prefers-reduced-motion: reduce) {
        [data-testid="stAppViewContainer"]::after { animation:none; }
        * { transition:none !important; animation:none !important; }
      }

      /* ============================================================
         Header / Sidebar / Containers
         ============================================================ */
      [data-testid="stHeader"] { background:rgba(3,5,11,.7); backdrop-filter:blur(6px); }
      [data-testid="stMainBlockContainer"], [data-testid="stSidebar"] { position:relative; z-index:1; }

      [data-testid="stSidebar"] {
        background:
          linear-gradient(180deg, rgba(3,6,14,.94), rgba(6,12,24,.94));
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

      /* ============================================================
         Hero
         ============================================================ */
      .hero {
        position:relative; padding:2rem 2.1rem 1.9rem;
        border:1px solid var(--line-hi);
        border-radius:14px;
        background:
          linear-gradient(125deg, rgba(14,26,48,.9), rgba(6,12,26,.86));
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

      /* ============================================================
         Metric / HUD cards
         ============================================================ */
      [data-testid="stMetric"], .metric-card {
        position:relative;
        background:
          linear-gradient(160deg, rgba(14,24,44,.85), rgba(6,12,26,.75));
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

      /* ============================================================
         Panels / Info boxes
         ============================================================ */
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

      /* ============================================================
         Buttons
         ============================================================ */
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

      /* ============================================================
         Inputs
         ============================================================ */
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

      /* ============================================================
         Tables
         ============================================================ */
      div[data-testid="stDataFrame"] {
        border:1px solid var(--line);
        border-radius:10px; overflow:hidden;
        background:rgba(6,12,26,.6);
      }

      /* ============================================================
         Badges / Status chips
         ============================================================ */
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

      /* Sidebar brand */
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
# Rendered behind the app. Runs once via components.html.
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

      // Three star layers (parallax drift)
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

      // Shooting stars
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

        // Draw stars
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

        // Shooting stars
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
# BACKEND LOGIC — unchanged from original
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

def validate_photometry_input(df: pd.DataFrame) -> tuple[bool, str]:
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
        out = out[out[col].
