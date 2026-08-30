"""Interactive Streamlit interface for SupportFlow AI."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import (
    DEPARTMENTS,
    FEEDBACK_DATA_DIR,
    METADATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
)
from src.feedback import save_correction
from src.inference import load_router, predict_ticket


FEEDBACK_PATH = FEEDBACK_DATA_DIR / "corrections.csv"

st.set_page_config(
    page_title="SupportFlow AI",
    page_icon="🎫",
    layout="wide",
)


@st.cache_resource
def get_router():
    return load_router(MODEL_PATH, METADATA_PATH)


@st.cache_data
def get_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text())


def probability_chart(probabilities: dict[str, float]):
    frame = pd.DataFrame(
        {
            "Department": list(probabilities),
            "Probability": list(probabilities.values()),
        }
    ).sort_values("Probability")
    figure = px.bar(
        frame,
        x="Probability",
        y="Department",
        orientation="h",
        text_auto=".1%",
        range_x=[0, 1],
        color="Probability",
        color_continuous_scale="Blues",
    )
    figure.update_layout(coloraxis_showscale=False, height=330, margin=dict(l=0, r=0, t=10, b=0))
    figure.update_xaxes(tickformat=".0%")
    return figure


def explanation_chart(explanation: list[dict]):
    frame = pd.DataFrame(explanation).sort_values("contribution")
    figure = px.bar(
        frame,
        x="contribution",
        y="feature",
        orientation="h",
        labels={"contribution": "Positive contribution", "feature": "Word or phrase"},
        color="contribution",
        color_continuous_scale="Teal",
    )
    figure.update_layout(coloraxis_showscale=False, height=330, margin=dict(l=0, r=0, t=10, b=0))
    return figure


st.title("🎫 SupportFlow AI")
st.caption("Interpretable customer-support ticket routing with human review")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("The trained router is missing. Run `python scripts/train.py` first.")
    st.stop()

model, metadata = get_router()

with st.form("ticket_form"):
    ticket_text = st.text_area(
        "Describe the customer’s issue",
        placeholder="Example: I was charged twice and need the duplicate payment refunded.",
        height=150,
        help="Do not include passwords, full card numbers, or other sensitive information.",
    )
    route_submitted = st.form_submit_button("Route ticket", type="primary")

if route_submitted:
    try:
        st.session_state["last_result"] = predict_ticket(ticket_text, model, metadata)
    except ValueError as error:
        st.session_state.pop("last_result", None)
        st.warning(str(error))

result = st.session_state.get("last_result")
if result:
    if result["needs_review"]:
        st.warning(
            "Human review required — prediction confidence is below the "
            f"{result['review_threshold']:.0%} automation threshold."
        )
        route_label = "Human Review"
    else:
        st.success(f"Auto-route to {result['department']}")
        route_label = result["department"]

    route_col, priority_col, confidence_col = st.columns(3)
    route_col.metric("Routing decision", route_label)
    priority_col.metric("Priority", result["priority"])
    confidence_col.metric("Prediction confidence", f"{result['confidence']:.1%}")

    if result["priority_terms"]:
        st.caption(
            "Priority rule matched: " + ", ".join(f"“{term}”" for term in result["priority_terms"])
        )
    else:
        st.caption(f"Priority assigned by transparent `{result['priority_method']}` policy.")

    probability_col, explanation_col = st.columns(2)
    with probability_col:
        st.subheader("Department probabilities")
        st.plotly_chart(
            probability_chart(result["probabilities"]),
            width="stretch",
            config={"displayModeBar": False},
        )

    with explanation_col:
        st.subheader("Words influencing this prediction")
        if result["explanation"]:
            st.plotly_chart(
                explanation_chart(result["explanation"]),
                width="stretch",
                config={"displayModeBar": False},
            )
        else:
            st.info("No positive word contribution was available for this input.")

    st.caption(
        "Word contributions show associations learned by the linear model. "
        "They do not prove causation or guarantee that a route is correct."
    )

    st.divider()
    st.subheader("Correct the route")
    st.write("Corrections are saved for later review; they do not retrain the live model automatically.")
    default_index = list(DEPARTMENTS).index(result["department"])
    with st.form("correction_form", clear_on_submit=False):
        corrected_department = st.selectbox(
            "Correct department",
            DEPARTMENTS,
            index=default_index,
        )
        correction_submitted = st.form_submit_button("Save correction")

    if correction_submitted:
        save_correction(
            FEEDBACK_PATH,
            ticket_text=result["cleaned_text"],
            predicted_department=result["department"],
            corrected_department=corrected_department,
            confidence=result["confidence"],
            priority=result["priority"],
            model_version=result["model_version"],
        )
        st.success("Correction saved for the next data-review cycle.")

if FEEDBACK_PATH.exists():
    st.download_button(
        "Download collected corrections",
        data=FEEDBACK_PATH.read_bytes(),
        file_name="supportflow_corrections.csv",
        mime="text/csv",
    )

with st.expander("Model performance and responsible-use notes"):
    metrics = get_metrics()
    test_metrics = metrics["test"]
    metric_col, f1_col, threshold_col = st.columns(3)
    metric_col.metric("Test accuracy", f"{test_metrics['accuracy']:.1%}")
    f1_col.metric("Test macro-F1", f"{test_metrics['macro_f1']:.3f}")
    threshold_col.metric("Review threshold", f"{metadata['review_threshold']:.0%}")

    comparison = pd.DataFrame(
        [
            {
                "Model": name.replace("_", " ").title(),
                "Validation accuracy": values["accuracy"],
                "Validation macro-F1": values["macro_f1"],
            }
            for name, values in metrics["validation"].items()
        ]
    )
    st.dataframe(
        comparison.style.format(
            {"Validation accuracy": "{:.3f}", "Validation macro-F1": "{:.3f}"}
        ),
        hide_index=True,
        width="stretch",
    )

    confusion = pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=metadata["departments"],
        columns=metadata["departments"],
    )
    confusion_figure = px.imshow(
        confusion,
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual", "color": "Tickets"},
        color_continuous_scale="Blues",
        aspect="auto",
    )
    confusion_figure.update_layout(height=430)
    st.plotly_chart(confusion_figure, width="stretch", config={"displayModeBar": False})

    st.warning(
        "The evaluation data contains generated, strongly templated requests. "
        "The high test score does not establish equivalent performance on real customer traffic."
    )
    st.markdown(
        "- Use this tool for routing assistance, not high-impact decisions.\n"
        "- Review low-confidence predictions and sensitive incidents.\n"
        "- Audit collected corrections before adding them to training data.\n"
        "- Do not enter passwords, full payment-card numbers, or confidential data."
    )

st.caption(f"Model version: {metadata['model_version']} · Selected model: logistic regression")
