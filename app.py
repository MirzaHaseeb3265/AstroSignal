
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AstroSignal | Cosmic Anomaly Explorer",
    page_icon="🔭",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("astrosignal_ranked_candidates.csv")

df = load_data()

st.title("🔭 AstroSignal")
st.subheader("Explore unusual astronomical objects in SDSS data")

st.write(
    "AstroSignal ranks potential anomaly candidates using "
    "unsupervised machine learning. These are candidates for "
    "further investigation, not confirmed discoveries."
)

# Overview metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Objects analyzed", f"{len(df):,}")

with col2:
    st.metric("Object classes", df["class"].nunique())

with col3:
    st.metric(
        "Quality flags",
        int((df["dq_extreme_color"] | df["dq_negative_z"]).sum())
    )

st.divider()

# Filters
st.header("Candidate Explorer")

classes = sorted(df["class"].dropna().unique().tolist())
selected_classes = st.multiselect(
    "Filter by object class",
    options=classes,
    default=classes
)

min_percentile = st.slider(
    "Minimum within-class anomaly percentile",
    min_value=0,
    max_value=100,
    value=95
)

filtered = df[
    df["class"].isin(selected_classes)
    & (df["class_percentile"] >= min_percentile)
].copy()

st.caption(
    "Higher anomaly scores indicate greater model-relative unusualness. "
    "Percentiles are calculated within each object class, not as probabilities."
)

st.write(f"Showing **{len(filtered):,}** candidates")

display_cols = [
    "candidate_row", "class", "ra", "dec", "z",
    "anomaly_score", "class_percentile",
    "dq_extreme_color", "dq_negative_z"
]

st.dataframe(
    filtered[display_cols].sort_values(
        "anomaly_score", ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

st.divider()

# Candidate details
st.header("Candidate Details")

if not filtered.empty:
    candidate_rows = filtered["candidate_row"].tolist()

    chosen = st.selectbox(
        "Choose a candidate row",
        options=candidate_rows
    )

    candidate = filtered[
        filtered["candidate_row"] == chosen
    ].iloc[0]

    a, b, c = st.columns(3)

    a.metric("Object class", str(candidate["class"]))
    b.metric("Anomaly score", f'{candidate["anomaly_score"]:.4f}')
    c.metric(
        "Within-class percentile",
        f'{candidate["class_percentile"]:.2f}'
    )

    st.write("**Sky coordinates**")
    st.write(f'RA: {candidate["ra"]}°')
    st.write(f'Dec: {candidate["dec"]}°')

    st.write("**Measurements**")
    st.json({
        "Redshift (z)": float(candidate["z"]),
        "u-g": float(candidate["u_g"]),
        "g-r": float(candidate["g_r"]),
        "r-i": float(candidate["r_i"]),
        "i-z": float(candidate["i_z"]),
        "g magnitude": float(candidate["mag_g"])
    })

    flags = []

    if bool(candidate["dq_extreme_color"]):
        flags.append("Extreme colour detected by the project's screening rule.")

    if bool(candidate["dq_negative_z"]):
        flags.append("Negative redshift flagged for review.")

    if flags:
        st.warning("Quality review indicators:\n\n- " + "\n- ".join(flags))
    else:
        st.success("No quality flags from the two current screening rules.")

    ra = float(candidate["ra"])
    dec = float(candidate["dec"])

    sdss_url = (
        "https://skyserver.sdss.org/dr17/VisualTools/quickobj"
        f"?ra={ra}&dec={dec}"
    )

    st.link_button("Investigate coordinates in SDSS", sdss_url)

else:
    st.info("No candidates match these filters.")

st.divider()
st.caption(
    "AstroSignal is an exploratory research prototype. "
    "Anomaly scores do not establish scientific novelty or data validity."
)
