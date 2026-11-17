
import json

import streamlit as st

from big_data_analytics.llm_insights import generate_business_insights


st.set_page_config(
    page_title="AI Business Insights",
    page_icon="🤖",
    layout="wide",
)

st.title("AI Business Insights")
st.caption(
    "AI-generated interpretation of calculated business analytics."
)

st.info(
    "The AI receives aggregated business metrics, category performance, monthly "
    "results, customer segment summaries, sentiment statistics, the LSTM forecast "
    "and the evaluation metrics of all forecasting models. "
    "Raw customer names, review texts and personal information are not sent. "
    "The AI does not replace the calculations: every number in its report is "
    "checked against the calculated results."
)

if st.button("Generate Business Insights", type="primary"):
    with st.spinner("Analyzing business performance... this can take up to a minute."):
        try:
            st.session_state["business_insights"] = generate_business_insights()
            st.success("Business insights generated successfully.")
        except Exception as error:
            st.error(f"Could not generate insights: {error}")

if "business_insights" in st.session_state:
    result = st.session_state["business_insights"]
    verification = result["verification"]
    unverified = verification[~verification["verified"]]

    st.divider()

    columns = st.columns(3)
    columns[0].metric("Model", result["model"].split("/")[-1], border=True)
    columns[1].metric("Numbers in the report", len(verification), border=True)
    columns[2].metric(
        "Matching the calculated data",
        f"{int(verification['verified'].sum())} of {len(verification)}",
        border=True,
    )

    if unverified.empty:
        st.success("All numbers in the report match the calculated results.")
    else:
        st.warning(
            f"{len(unverified)} number(s) in the report were not found in the "
            "calculated data. They may be derived by the model or incorrect, "
            "so check them before using the conclusions."
        )
        st.dataframe(
            unverified[["number", "text"]],
            hide_index=True,
            column_config={
                "number": "Number",
                "text": st.column_config.TextColumn("Statement", width="large"),
            },
        )

    with st.container(border=True):
        st.markdown(result["report"].replace("$", "\\$"))

    with st.expander("Verification of all numbers"):
        st.dataframe(
            verification,
            hide_index=True,
            column_config={
                "number": "Number",
                "verified": st.column_config.CheckboxColumn("Found in data"),
                "text": st.column_config.TextColumn("Statement", width="large"),
            },
        )

    with st.expander("Data sent to the model"):
        st.json(result["context"], expanded=False)

    st.download_button(
        "Download Business Report",
        data=result["report"],
        file_name="business_insights.md",
        mime="text/markdown",
    )
    st.download_button(
        "Download Input Data",
        data=json.dumps(result["context"], ensure_ascii=False, indent=2),
        file_name="llm_input_data.json",
        mime="application/json",
    )
