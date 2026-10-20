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
