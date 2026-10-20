
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="AstroSignal | Cosmic Anomaly Explorer",
    page_icon="🔭",
    layout="wide"
)

REQUIRED_MAGNITUDES = [
    "mag_u", "mag_g", "mag_r", "mag_i", "mag_z"
]
OPTIONAL_COLUMNS = ["class", "z", "ra", "dec", "objID"]
COLOUR_COLUMNS = ["u_g", "g_r", "r_i", "i_z"]


# --------------------------------------------------
# Load the existing AstroSignal results
# --------------------------------------------------
@st.cache_data
def load_existing_data():
    return pd.read_csv("astrosignal_ranked_candidates.csv")


def add_colour_features(data):
    data = data.copy()

    for col in REQUIRED_MAGNITUDES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["u_g"] = data["mag_u"] - data["mag_g"]
    data["g_r"] = data["mag_g"] - data["mag_r"]
    data["r_i"] = data["mag_r"] - data["mag_i"]
    data["i_z"] = data["mag_i"] - data["mag_z"]

    return data


def truthy(value):
    if pd.isna(value):
        return False
    return str(value).strip().lower() in ("true", "1", "yes")


def analyze_uploaded_data(data):
    """Run a fresh anomaly analysis on uploaded astronomical data."""
    data = data.copy().reset_index(drop=True)
    data["candidate_row"] = np.arange(len(data))

    data = add_colour_features(data)

    # Use the same basic photometric features for every upload.
    # Redshift is displayed when available, but is not used as a
    # model feature because unreliable redshift values can dominate
    # the ranking.
    features = COLOUR_COLUMNS + ["mag_g"]

    # Flag rows that cannot be analyzed reliably.
    finite = np.isfinite(data[features]).all(axis=1)
    valid = data.loc[finite].copy()
    invalid = data.loc[~finite].copy()

    if len(valid) < 10:
        raise ValueError(
            "At least 10 rows with valid numeric magnitude values "
            "are required for analysis."
        )

    # Avoid fitting a separate model to tiny object classes.
    if "class" in valid.columns:
        valid["class"] = (
            valid["class"].fillna("UNKNOWN").astype(str).str.upper()
        )
        class_counts = valid["class"].value_counts()
        valid["model_class"] = valid["class"].where(
            valid["class"].map(class_counts) >= 20,
            "ALL OBJECTS"
        )
    else:
        valid["class"] = "UNKNOWN"
        valid["model_class"] = "ALL OBJECTS"

    groups = []

    for model_class, group in valid.groupby("model_class", sort=False):
        group = group.copy()

        # A model needs enough records to learn a useful comparison.
        if len(group) < 10:
            group["anomaly_score"] = np.nan
            group["class_percentile"] = np.nan
            group["is_anomaly"] = False
            group["analysis_note"] = (
                "Not enough records for reliable model fitting."
            )
            groups.append(group)
            continue

        model = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=42
        )

        X = group[features]
        model.fit(X)

        group["anomaly_score"] = -model.score_samples(X)
        group["class_percentile"] = (
            group["anomaly_score"].rank(pct=True) * 100
        )
        group["is_anomaly"] = model.predict(X) == -1
        group["analysis_note"] = (
            "Ranked relative to the comparison group."
        )

        groups.append(group)

    results = pd.concat(groups, ignore_index=True)
    results = results.sort_values(
        "anomaly_score", ascending=False, na_position="last"
    )

    # Data-quality indicators are separate from model anomalies.
    results["dq_extreme_color"] = (
        results[COLOUR_COLUMNS].abs().gt(4).any(axis=1)
    )

    if "z" in results.columns:
        results["z"] = pd.to_numeric(results["z"], errors="coerce")
        results["dq_negative_z"] = results["z"] < 0
    else:
        results["dq_negative_z"] = False

    # Keep rows with invalid magnitudes visible in the exported file.
    if not invalid.empty:
        invalid["model_class"] = "NOT ANALYZED"
        invalid["anomaly_score"] = np.nan
        invalid["class_percentile"] = np.nan
        invalid["is_anomaly"] = False
        invalid["analysis_note"] = (
            "Not analyzed: missing, non-numeric, or infinite "
            "magnitude values."
        )
        invalid["dq_extreme_color"] = False
        invalid["dq_negative_z"] = False
        results = pd.concat(
            [results, invalid], ignore_index=True
        )

    return results


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🔭 AstroSignal")
st.subheader("Explore unusual astronomical objects in SDSS data")

st.write(
    "AstroSignal uses unsupervised machine learning to rank "
    "astronomical anomaly candidates and help users investigate "
    "their measurements. Candidates are not confirmed discoveries."
)

st.info(
    "An anomaly score measures how unusual an object looks to the "
    "model. It is not a probability that the object is a discovery. "
    "Data-quality warnings are separate from anomaly rankings."
)

# --------------------------------------------------
# Main sections
# --------------------------------------------------
existing_tab, upload_tab = st.tabs([
    "🔭 Existing SDSS Explorer",
    "📂 Upload & Analyze CSV"
])

# ==================================================
# TAB 1: EXISTING FEATURES
# ==================================================
with existing_tab:
    try:
        df = load_existing_data()
    except Exception as exc:
        st.error(
            "Could not load astrosignal_ranked_candidates.csv. "
            "Make sure it is in the same GitHub repository as app.py."
        )
        st.exception(exc)
        df = pd.DataFrame()

    if not df.empty:
        flagged = (
            df.get("dq_extreme_color", pd.Series(False, index=df.index))
            .map(truthy)
            |
            df.get("dq_negative_z", pd.Series(False, index=df.index))
            .map(truthy)
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Objects analyzed", f"{len(df):,}")
        c2.metric("Object classes", df["class"].nunique())
        c3.metric("Quality review flags", f"{int(flagged.sum()):,}")

        st.divider()
        st.header("Candidate Explorer")

        classes = sorted(df["class"].dropna().unique().tolist())

        selected_classes = st.multiselect(
            "Filter by object class",
            options=classes,
            default=classes,
            key="existing_classes"
        )

        min_percentile = st.slider(
            "Minimum within-class anomaly percentile",
            min_value=0,
            max_value=100,
            value=95,
            key="existing_percentile"
        )

        filtered = df[
            df["class"].isin(selected_classes)
            & (df["class_percentile"] >= min_percentile)
        ].copy()

        st.caption(
            "Percentiles are calculated within each class. "
            "They are not probabilities and should not be compared "
            "as calibrated scores between classes."
        )

        st.write(f"Showing **{len(filtered):,}** candidates")

        display_cols = [
            c for c in [
                "candidate_row", "class", "ra", "dec", "z",
                "anomaly_score", "class_percentile",
                "dq_extreme_color", "dq_negative_z"
            ] if c in filtered.columns
        ]

        st.dataframe(
            filtered[display_cols].sort_values(
                "anomaly_score", ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.header("Candidate Details")

        if not filtered.empty:
            selected_row = st.selectbox(
                "Choose a candidate row",
                filtered["candidate_row"].tolist(),
                key="existing_candidate"
            )

            candidate = filtered[
                filtered["candidate_row"] == selected_row
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

            measurement_cols = [
                c for c in [
                    "z", "mag_u", "mag_g", "mag_r", "mag_i", "mag_z",
                    "u_g", "g_r", "r_i", "i_z"
                ] if c in candidate.index
            ]

            st.write("**Measurements**")
            st.dataframe(
                pd.DataFrame({
                    "Measurement": measurement_cols,
                    "Value": [
                        candidate[c] for c in measurement_cols
                    ]
                }),
                hide_index=True,
                use_container_width=True
            )

            flags = []
            if truthy(candidate.get("dq_extreme_color", False)):
                flags.append(
                    "At least one colour difference exceeds 4 magnitudes."
                )
            if truthy(candidate.get("dq_negative_z", False)):
                flags.append(
                    "Negative redshift flagged for review."
                )

            if flags:
                st.warning("\n\n".join(flags))
            else:
                st.info(
                    "No flags from the two current screening rules. "
                    "This does not guarantee the measurements are valid."
                )

            ra = pd.to_numeric(candidate.get("ra"), errors="coerce")
            dec = pd.to_numeric(candidate.get("dec"), errors="coerce")

            if pd.notna(ra) and pd.notna(dec):
                sdss_url = (
                    "https://skyserver.sdss.org/dr17/VisualTools/quickobj"
                    f"?ra={ra}&dec={dec}"
                )
                st.link_button(
                    "Investigate coordinates in SDSS",
                    sdss_url
                )

# ==================================================
# TAB 2: CSV UPLOAD AND LIVE ANALYSIS
# ==================================================
with upload_tab:
    st.header("Upload your astronomy dataset")

    st.write(
        "Upload a CSV to run a fresh anomaly analysis. "
        "Your original SDSS results remain available in the other tab."
    )

    with st.expander("📋 Which columns should my CSV contain?", expanded=True):
        st.markdown(
            """
            **Required columns** — five numeric magnitude measurements:

            - `mag_u` — u-band magnitude
            - `mag_g` — g-band magnitude
            - `mag_r` — r-band magnitude
            - `mag_i` — i-band magnitude
            - `mag_z` — z-band magnitude

            **Optional columns:**

            - `class` — object class, such as `GALAXY`, `STAR`, or `QSO`
            - `z` — redshift
            - `ra`, `dec` — sky coordinates
            - `objID` — source identifier

            Column names must match exactly. Each row should represent
            one astronomical object. The five magnitude columns must
            contain numeric values. Rows with missing or invalid
            magnitude values will be reported and excluded from fitting.
            """
        )

        example = pd.DataFrame({
            "mag_u": [20.1, 21.3],
            "mag_g": [19.8, 20.7],
            "mag_r": [19.5, 20.1],
            "mag_i": [19.3, 19.9],
            "mag_z": [19.2, 19.8],
            "class": ["GALAXY", "QSO"],
            "z": [0.12, 1.25],
            "ra": [150.1, 151.2],
            "dec": [1.1, -0.3]
        })

        st.write("Example CSV layout:")
        st.dataframe(example, hide_index=True, use_container_width=True)

        st.download_button(
            "Download example CSV template",
            data=example.to_csv(index=False).encode("utf-8"),
            file_name="astrosignal_csv_template.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Use an astronomy dataset with the required magnitude columns."
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.success(
                f"File loaded: {len(uploaded_df):,} rows and "
                f"{len(uploaded_df.columns)} columns."
            )

            st.write("**Columns detected in your file:**")
            st.code(", ".join(map(str, uploaded_df.columns)))

            missing_cols = [
                col for col in REQUIRED_MAGNITUDES
                if col not in uploaded_df.columns
            ]

            if missing_cols:
                st.error(
                    "This CSV is missing required columns: "
                    + ", ".join(missing_cols)
                )
                st.info(
                    "Please correct the column names or download "
                    "the example template above. Analysis has not run."
                )

            elif len(uploaded_df) < 10:
                st.error(
                    "Please upload at least 10 rows for anomaly analysis."
                )

            else:
                st.write("**Preview of uploaded data:**")
                st.dataframe(
                    uploaded_df.head(10),
                    use_container_width=True,
                    hide_index=True
                )

                saved_results_available = (
                    "anomaly_score" in uploaded_df.columns
                    and "class_percentile" in uploaded_df.columns
                )

                if saved_results_available:
                    analysis_mode = st.radio(
                        "How would you like to use this file?",
                        [
                            "Run fresh anomaly analysis",
                            "Explore saved anomaly results"
                        ],
                        help=(
                            "Choose fresh analysis to fit a new model. "
                            "Choose saved results to inspect the existing "
                            "scores in this file."
                        )
                    )
                else:
                    analysis_mode = "Run fresh anomaly analysis"

                if st.button(
                    "Analyze uploaded CSV",
                    type="primary",
                    key="analyze_upload"
                ):
                    st.session_state["upload_analysis_mode"] = analysis_mode
                    st.session_state["upload_analysis_name"] = uploaded_file.name

                    if analysis_mode == "Explore saved anomaly results":
                        st.session_state["upload_results"] = uploaded_df.copy()
                        st.session_state["upload_results_are_fresh"] = False
                    else:
                        try:
                            results = analyze_uploaded_data(uploaded_df)
                            st.session_state["upload_results"] = results
                            st.session_state["upload_results_are_fresh"] = True
                        except Exception as exc:
                            st.error(f"Analysis could not run: {exc}")
                            st.stop()

                results = st.session_state.get("upload_results")
                same_file = (
                    st.session_state.get("upload_analysis_name")
                    == uploaded_file.name
                )

                if results is not None and same_file:
                    results = results.copy()
                    fresh = st.session_state.get(
                        "upload_results_are_fresh", False
                    )

                    st.divider()
                    st.header("Analysis results")

                    if fresh:
                        st.success(
                            "Fresh anomaly analysis completed on your uploaded data."
                        )
                    else:
                        st.info(
                            "Showing saved scores from the uploaded file. "
                            "The model has not been rerun."
                        )

                    if "is_anomaly" in results.columns:
                        anomaly_mask = (
                            results["is_anomaly"].fillna(False).astype(bool)
                        )
                    elif "anomaly_score" in results.columns:
                        anomaly_mask = pd.Series(
                            False, index=results.index
                        )
                        st.caption(
                            "This file has saved scores but no anomaly labels; "
                            "scores are shown without inventing new labels."
                        )
                    else:
                        anomaly_mask = pd.Series(
                            False, index=results.index
                        )

                    valid_scores = pd.to_numeric(
                        results.get(
                            "anomaly_score",
                            pd.Series(np.nan, index=results.index)
                        ),
                        errors="coerce"
                    )

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Rows in result", f"{len(results):,}")
                    m2.metric(
                        "Flagged by model",
                        f"{int(anomaly_mask.sum()):,}"
                    )
                    m3.metric(
                        "Rows with a score",
                        f"{int(valid_scores.notna().sum()):,}"
                    )

                    if "analysis_note" in results.columns:
                        excluded = results["analysis_note"].astype(str).str.startswith(
                            "Not analyzed"
                        ).sum()
                        if excluded:
                            st.warning(
                                f"{excluded} rows were not analyzed because "
                                "their magnitude measurements were invalid "
                                "or insufficient for fitting."
                            )

                    st.subheader("Candidate results")

                    available_classes = (
                        sorted(results["class"].dropna().astype(str).unique())
                        if "class" in results.columns else []
                    )

                    if available_classes:
                        chosen_classes = st.multiselect(
                            "Filter result classes",
                            available_classes,
                            default=available_classes,
                            key="upload_classes"
                        )
                        view = results[
                            results["class"].astype(str).isin(chosen_classes)
                        ].copy()
                    else:
                        view = results.copy()

                    if "class_percentile" in view.columns:
                        min_upload_percentile = st.slider(
                            "Minimum anomaly percentile",
                            min_value=0,
                            max_value=100,
                            value=95,
                            key="upload_percentile"
                        )
                        percentile_values = pd.to_numeric(
                            view["class_percentile"], errors="coerce"
                        )
                        view = view[
                            percentile_values >= min_upload_percentile
                        ]

                    show_cols = [
                        col for col in [
                            "candidate_row", "objID", "class", "ra", "dec",
                            "z", "anomaly_score", "class_percentile",
                            "is_anomaly", "dq_extreme_color",
                            "dq_negative_z", "analysis_note"
                        ] if col in view.columns
                    ]

                    if "anomaly_score" in view.columns:
                        view = view.sort_values(
                            "anomaly_score",
                            ascending=False,
                            na_position="last"
                        )

                    st.dataframe(
                        view[show_cols] if show_cols else view,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Interactive colour-colour scatter plot
                    if all(col in view.columns for col in ["g_r", "r_i"]):
                        st.subheader("Colour–colour diagram")

                        plot_df = view.copy()
                        plot_df["g_r"] = pd.to_numeric(
                            plot_df["g_r"], errors="coerce"
                        )
                        plot_df["r_i"] = pd.to_numeric(
                            plot_df["r_i"], errors="coerce"
                        )

                        if "is_anomaly" in plot_df.columns:
                            plot_df["Model result"] = np.where(
                                plot_df["is_anomaly"].fillna(False).astype(bool),
                                "Model-flagged candidate",
                                "Other analyzed object"
                            )
                        else:
                            plot_df["Model result"] = "Saved result (label unavailable)"

                        plot_df = plot_df.replace(
                            [np.inf, -np.inf], np.nan
                        ).dropna(subset=["g_r", "r_i"])

                        if not plot_df.empty:
                            hover_cols = [
                                col for col in [
                                    "candidate_row", "class", "z",
                                    "anomaly_score", "class_percentile"
                                ] if col in plot_df.columns
                            ]

                            fig = px.scatter(
                                plot_df,
                                x="g_r",
                                y="r_i",
                                color="Model result",
                                symbol="Model result",
                                hover_data=hover_cols,
                                title="Compare g-r and r-i colours",
                                labels={
                                    "g_r": "g − r colour (magnitudes)",
                                    "r_i": "r − i colour (magnitudes)"
                                }
                            )

                            fig.update_layout(
                                legend_title_text="",
                                xaxis_title="g − r colour (magnitudes)",
                                yaxis_title="r − i colour (magnitudes)"
                            )
                            st.plotly_chart(
                                fig, use_container_width=True
                            )

                            st.caption(
                                "Points farther from the main concentration "
                                "may warrant review, but the graph alone does "
                                "not prove a physical anomaly. Extreme values "
                                "can stretch the axes."
                            )

                    # Detailed selected candidate
                    st.subheader("Inspect a result")

                    if not view.empty:
                        if "candidate_row" in view.columns:
                            candidate_options = view["candidate_row"].tolist()
                            selected = st.selectbox(
                                "Select candidate row",
                                candidate_options,
                                key="upload_selected_candidate"
                            )
                            detail = view[
                                view["candidate_row"] == selected
                            ].iloc[0]
                        else:
                            selected_idx = st.selectbox(
                                "Select result",
                                view.index.tolist(),
                                key="upload_selected_index"
                            )
                            detail = view.loc[selected_idx]

                        details = {
                            col: detail[col]
                            for col in [
                                "class", "ra", "dec", "z",
                                *REQUIRED_MAGNITUDES, *COLOUR_COLUMNS,
                                "anomaly_score", "class_percentile",
                                "is_anomaly", "dq_extreme_color",
                                "dq_negative_z", "analysis_note"
                            ]
                            if col in detail.index
                            and pd.notna(detail[col])
                        }

                        st.json(details)

                        # Compare the selected colour with the dataset
                        if (
                            "g_r" in results.columns
                            and "g_r" in detail.index
                            and pd.notna(detail["g_r"])
                        ):
                            compare = results.copy()
                            compare["g_r"] = pd.to_numeric(
                                compare["g_r"], errors="coerce"
                            )

                            if (
                                "class" in compare.columns
                                and "class" in detail.index
                                and pd.notna(detail["class"])
                            ):
                                peers = compare[
                                    compare["class"].astype(str)
                                    == str(detail["class"])
                                ]
                                if len(peers) < 10:
                                    peers = compare
                            else:
                                peers = compare

                            peers = peers.dropna(subset=["g_r"])

                            if not peers.empty:
                                fig_compare = px.histogram(
                                    peers,
                                    x="g_r",
                                    nbins=30,
                                    title="g-r colour distribution of comparison objects",
                                    labels={"g_r": "g − r colour (magnitudes)"}
                                )
                                fig_compare.add_vline(
                                    x=float(detail["g_r"]),
                                    line_dash="dash",
                                    annotation_text="Selected candidate"
                                )
                                st.plotly_chart(
                                    fig_compare,
                                    use_container_width=True
                                )
                                st.caption(
                                    "This graph compares one measurement "
                                    "with the available dataset or class. "
                                    "It is descriptive, not a significance test."
                                )

                    # Download the results
                    st.download_button(
                        "Download analysis results as CSV",
                        data=results.to_csv(index=False).encode("utf-8"),
                        file_name="astrosignal_analysis_results.csv",
                        mime="text/csv"
                    )

        except pd.errors.EmptyDataError:
            st.error("This CSV file is empty. Please upload a file with data.")
        except UnicodeDecodeError:
            st.error(
                "The file could not be read as UTF-8. "
                "Please save the CSV as UTF-8 and upload it again."
            )
        except Exception as exc:
            st.error(
                "Something went wrong while reading or processing this CSV. "
                "Check the file format and column names, then try again."
            )
            st.exception(exc)

st.divider()
st.caption(
    "AstroSignal is an exploratory research prototype. "
    "Anomaly rankings are model-relative and require scientific verification."
)
