
from pathlib import Path
from datetime import datetime
import io
import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.prediction import predict_machine


# ---------------------------------------------------------------------
# 1. APPLICATION CONFIGURATION
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
DATASET_PATH = BASE_DIR / "dataset" / "ai4i2020.csv"

st.set_page_config(
    page_title="Predictive Maintenance | Machine Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design colours
BG = "#08111f"
PANEL = "#111f31"
PANEL_ALT = "#15263b"
BORDER = "#2a405b"
TEXT = "#f4f7fc"
MUTED = "#c0ccdc"
ACCENT = "#56c2f2"
GREEN = "#43d6a1"
AMBER = "#f4c56a"
RED = "#ff7887"


# ---------------------------------------------------------------------
# 2. PROFESSIONAL DASHBOARD STYLING
# ---------------------------------------------------------------------

st.markdown(
    f"""
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap'
    );

    :root {{
        color-scheme: dark;
    }}

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
    }}

    .stApp {{
        background: radial-gradient(
            ellipse at 85% 0%,
            #142b43 0%,
            {BG} 48%
        );
        color: {TEXT};
    }}

    header[data-testid="stHeader"] {{
        background: rgba(8,17,31,.96);
    }}

    .block-container {{
        max-width: 1680px;
        padding: 1.4rem clamp(1rem, 3vw, 3rem) 3rem;
    }}

    /* Main headings */

    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {{
        color: {TEXT} !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 800 !important;
        line-height: 1.4 !important;
        opacity: 1 !important;
    }}

    p, li, label {{
        color: {MUTED};
    }}

    [data-testid="stCaptionContainer"] p {{
        color: #afbed1 !important;
        font-size: 12px;
    }}

    /* Sidebar */

    [data-testid="stSidebar"] {{
        background: linear-gradient(
            180deg,
            #101f32,
            #0b1625
        );
        border-right: 1px solid {BORDER};
    }}

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        color: {TEXT} !important;
        font-weight: 700 !important;
        opacity: 1 !important;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: {MUTED} !important;
    }}

    /* Section headings */

    .section-label {{
        display: block;
        color: {ACCENT} !important;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.8px;
        line-height: 1.8;
        margin: 5px 0 7px;
        visibility: visible;
    }}

    .section-heading {{
        display: block;
        color: {TEXT} !important;
        font-family: 'Manrope', sans-serif;
        font-size: 29px;
        font-weight: 800;
        line-height: 1.35;
        margin: 0 0 8px;
        visibility: visible;
    }}

    .section-description {{
        display: block;
        color: {MUTED} !important;
        font-size: 14px;
        line-height: 1.75;
        max-width: 1050px;
        margin-bottom: 22px;
        visibility: visible;
    }}

    /* Metric cards */

    .metric-card {{
        background: linear-gradient(
            145deg,
            {PANEL_ALT},
            {PANEL}
        );
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 19px 20px;
        min-height: 132px;
        box-shadow: 0 10px 28px rgba(0,0,0,.16);
    }}

    .metric-label {{
        color: #c2d1e2 !important;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }}

    .metric-value {{
        color: {TEXT} !important;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(23px, 2vw, 31px);
        font-weight: 800;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }}

    .metric-caption {{
        color: #afbed1 !important;
        font-size: 11px;
        margin-top: 9px;
        line-height: 1.5;
    }}

    /* Information cards */

    .info-card {{
        background: linear-gradient(
            145deg,
            #14263b,
            #101d2e
        );
        border: 1px solid {BORDER};
        border-radius: 13px;
        padding: 17px 19px;
        margin-bottom: 12px;
        min-height: 90px;
    }}

    .info-title {{
        color: {TEXT} !important;
        font-size: 14px;
        font-weight: 800;
        margin-bottom: 7px;
    }}

    .info-text {{
        color: {MUTED} !important;
        font-size: 13px;
        line-height: 1.7;
    }}

    /* Risk badges */

    .status-pill {{
        display: inline-block;
        border-radius: 30px;
        padding: 7px 13px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .7px;
    }}

    .status-low {{
        background: #103b34;
        color: #70e7bd;
        border: 1px solid #267b68;
    }}

    .status-medium {{
        background: #493719;
        color: #ffda83;
        border: 1px solid #88672e;
    }}

    .status-high {{
        background: #4a202b;
        color: #ff9aa5;
        border: 1px solid #8c3c4b;
    }}

    /* Navigation tabs */

    [data-testid="stTabs"] {{
        margin-top: 12px;
    }}

    [data-testid="stTabs"] [role="tablist"] {{
        gap: 7px;
        border-bottom: 1px solid {BORDER};
    }}

    [data-testid="stTabs"] button {{
        color: #d4deeb !important;
        font-size: 13px !important;
        font-weight: 800 !important;
        min-height: 52px;
        padding: 11px 13px;
        white-space: nowrap;
        opacity: 1 !important;
    }}

    [data-testid="stTabs"] button p {{
        color: inherit !important;
        font-weight: 800 !important;
    }}

    [data-testid="stTabs"] button[aria-selected="true"] {{
        color: {ACCENT} !important;
        border-bottom: 3px solid {ACCENT} !important;
    }}

    [data-testid="stTabs"] button:hover {{
        color: #ffffff !important;
    }}

    /* Buttons */

    div.stButton > button {{
        background: #087fbd;
        color: white !important;
        border: 1px solid #169bdc;
        border-radius: 9px;
        min-height: 44px;
        font-weight: 800;
    }}

    div.stButton > button:hover {{
        background: #1199d7;
        border-color: #65cafa;
        color: white !important;
    }}

    .stDownloadButton > button {{
        background: #172b42;
        color: #f2f7ff !important;
        border: 1px solid #42617f;
        border-radius: 9px;
        min-height: 42px;
        font-weight: 700;
    }}

    .stDownloadButton > button:hover {{
        background: #203d5b;
        color: white !important;
    }}

    /* Expanders and data tables */

    [data-testid="stExpander"] {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {{
        color: {TEXT} !important;
        font-weight: 700;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        overflow: hidden;
    }}

    /* Charts */

    [data-testid="stPlotlyChart"] {{
        background: rgba(14,27,43,.5);
        border: 1px solid rgba(91,133,174,.22);
        border-radius: 13px;
        padding: 6px;
    }}

    /* Alerts */

    [data-testid="stAlert"] p {{
        color: {TEXT} !important;
    }}

    hr {{
        border-color: {BORDER};
    }}

    .muted-note {{
        color: #afbed1;
        font-size: 12px;
        line-height: 1.7;
    }}

    @media(max-width:900px) {{
        .section-heading {{
            font-size: 24px;
        }}

        .metric-card {{
            min-height: 115px;
        }}
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# 3. REUSABLE HELPER FUNCTIONS
# ---------------------------------------------------------------------

def section_header(label, heading, description):

    st.markdown(
        f'<div class="section-label">{html.escape(label)}</div>'
        f'<div class="section-heading">{html.escape(heading)}</div>'
        f'<div class="section-description">{html.escape(description)}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value, caption=""):

    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{html.escape(str(label))}</div>'
        f'<div class="metric-value">{html.escape(str(value))}</div>'
        f'<div class="metric-caption">{html.escape(str(caption))}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def info_card(title, description):

    st.markdown(
        f'<div class="info-card">'
        f'<div class="info-title">{html.escape(str(title))}</div>'
        f'<div class="info-text">{html.escape(str(description))}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def load_csv(path):

    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


def load_dataset():

    return load_csv(DATASET_PATH)


def get_probability(result):

    """
    Prediction module returns percentage points from 0 to 100.
    """

    try:
        return max(
            0.0,
            min(float(result.get("failure_probability", 0.0)), 100.0)
        )

    except (TypeError, ValueError):
        return 0.0


def risk_class(risk):

    value = str(risk).strip().lower()

    if "high" in value or "critical" in value:
        return "status-high"

    if "medium" in value or "moderate" in value:
        return "status-medium"

    return "status-low"


def style_figure(fig, height=380):

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color=TEXT,
            family="DM Sans, sans-serif"
        ),
        margin=dict(
            l=18,
            r=18,
            t=45,
            b=25
        ),
        legend=dict(
            font=dict(color=MUTED)
        ),
        title_font=dict(
            color=TEXT,
            size=16
        ),
        xaxis=dict(
            title_font=dict(color=MUTED),
            tickfont=dict(color=MUTED),
            gridcolor="#26394f"
        ),
        yaxis=dict(
            title_font=dict(color=MUTED),
            tickfont=dict(color=MUTED),
            gridcolor="#26394f"
        ),
    )

    return fig


def normalize_comparison(df):

    if df.empty:
        return df

    out = df.copy()

    out.columns = [
        str(c).strip()
        for c in out.columns
    ]

    rename = {
        "F1": "F1 Score",
        "F1-score": "F1 Score",
        "PR AUC": "PR-AUC",
        "PR_AUC": "PR-AUC",
    }

    return out.rename(columns=rename)


def to_csv_bytes(df):

    return df.to_csv(index=False).encode("utf-8-sig")


def safe_int(value, default=0):

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------
# 4. SESSION STATE AND DATA LOADING
# ---------------------------------------------------------------------

for key, default in {
    "latest_result": None,
    "latest_sensors": None,
    "latest_analysis_time": None,
    "analysis_history": [],
}.items():

    if key not in st.session_state:
        st.session_state[key] = default


comparison_df = normalize_comparison(
    load_csv(RESULTS_DIR / "model_comparison.csv")
)

importance_df = load_csv(
    RESULTS_DIR / "feature_importance.csv"
)

dataset_df = load_dataset()


# ---------------------------------------------------------------------
# 5. SIDEBAR - MACHINE CONFIGURATION
# ---------------------------------------------------------------------

with st.sidebar:

    st.markdown("## ⚙️ Machine Configuration")

    st.markdown(
        '<p class="muted-note">'
        'Enter the current operating readings, then run the machine health assessment.'
        '</p>',
        unsafe_allow_html=True
    )

    st.divider()

    machine_type = st.selectbox(
        "Machine type",
        ["L", "M", "H"],
        index=1,
        help="Product quality category used by the AI4I 2020 dataset.",
    )

    st.markdown("### Thermal Parameters")

    air_temp = st.slider(
        "Air temperature (K)",
        295.0,
        305.0,
        300.0,
        0.1
    )

    process_temp = st.slider(
        "Process temperature (K)",
        305.0,
        314.0,
        310.0,
        0.1
    )

    st.markdown("### Mechanical Parameters")

    rotational_speed = st.slider(
        "Rotational speed (rpm)",
        1000,
        3000,
        1500,
        10
    )

    torque = st.slider(
        "Torque (Nm)",
        3.0,
        77.0,
        40.0,
        0.5
    )

    tool_wear = st.slider(
        "Tool wear (min)",
        0,
        253,
        100,
        1
    )

    st.divider()

    analyze_button = st.button(
        "▶  Run Machine Analysis",
        use_container_width=True,
        type="primary"
    )

    st.markdown("### System Information")

    st.caption("Prediction model · XGBoost")
    st.caption("Anomaly detection · Isolation Forest")
    st.caption("Reference dataset · AI4I 2020")
    st.caption("Input source · Configured sensor readings")


# ---------------------------------------------------------------------
# 6. RUN MACHINE PREDICTION
# ---------------------------------------------------------------------

if analyze_button:

    sensor_data = {

        "Type": machine_type,

        "Air temperature [K]": float(air_temp),

        "Process temperature [K]": float(process_temp),

        "Rotational speed [rpm]": int(rotational_speed),

        "Torque [Nm]": float(torque),

        "Tool wear [min]": int(tool_wear),

    }

    try:

        with st.spinner(
            "Evaluating sensor readings and generating machine health assessment..."
        ):

            prediction = predict_machine(sensor_data)

        st.session_state.latest_result = prediction

        st.session_state.latest_sensors = sensor_data

        st.session_state.latest_analysis_time = (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        probability = get_probability(prediction)

        history_entry = {

            "Analysis Time": st.session_state.latest_analysis_time,

            "Machine Type": machine_type,

            "Failure Probability (%)": round(probability, 2),

            "Risk Level": prediction.get("risk_level", "Unknown"),

            "Failure Status": prediction.get("failure_status", "Unknown"),

            "Anomaly Status": prediction.get("anomaly_status", "Unknown"),

            "Rotational Speed (rpm)": rotational_speed,

            "Torque (Nm)": torque,

            "Tool Wear (min)": tool_wear,

        }

        st.session_state.analysis_history.insert(
            0,
            history_entry
        )

        st.session_state.analysis_history = (
            st.session_state.analysis_history[:50]
        )

        st.success("Machine analysis completed.")

    except Exception as exc:

        st.error(
            f"Machine analysis could not be completed: {exc}"
        )


# ---------------------------------------------------------------------
# 7. MAIN DASHBOARD NAVIGATION
# ---------------------------------------------------------------------

tabs = st.tabs([
    "Executive Overview",
    "Sensor Diagnostics",
    "Model Intelligence",
    "Maintenance Intelligence",
    "Reports & Insights",
])


# ---------------------------------------------------------------------
# 8. EXECUTIVE OVERVIEW
# ---------------------------------------------------------------------

with tabs[0]:

    section_header(
        "01 / SYSTEM SUMMARY",
        "Executive Overview",
        "A clear summary of the latest machine assessment, predicted failure risk and operating condition.",
    )

    if st.session_state.latest_result is None:

        info_card(
            "Ready for machine assessment",
            "Set the operating parameters in the left panel and select Run Machine Analysis. The results will appear here.",
        )

        if not dataset_df.empty:

            total_records = len(dataset_df)

            if "Machine failure" in dataset_df.columns:

                failure_count = safe_int(
                    pd.to_numeric(
                        dataset_df["Machine failure"],
                        errors="coerce"
                    ).fillna(0).sum()
                )

                failure_share = (
                    100 * failure_count / total_records
                    if total_records else 0
                )

            else:

                failure_count = 0
                failure_share = 0.0

            a, b, c = st.columns(3)

            with a:
                metric_card(
                    "DATASET RECORDS",
                    f"{total_records:,}",
                    "Historical observations"
                )

            with b:
                metric_card(
                    "LABELLED FAILURES",
                    f"{failure_count:,}",
                    "Historical dataset labels"
                )

            with c:
                metric_card(
                    "HISTORICAL FAILURE SHARE",
                    f"{failure_share:.2f}%",
                    "Dataset proportion, not live machine risk"
                )

            if "Machine failure" in dataset_df.columns:

                st.markdown("### Historical Failure Distribution")

                counts = (
                    pd.to_numeric(
                        dataset_df["Machine failure"],
                        errors="coerce"
                    )
                    .value_counts()
                    .reindex([0, 1], fill_value=0)
                )

                dist = pd.DataFrame({
                    "Condition": ["No failure", "Failure"],
                    "Records": counts.values
                })

                fig = px.pie(
                    dist,
                    names="Condition",
                    values="Records",
                    hole=0.62,
                    color="Condition",
                    color_discrete_map={
                        "No failure": GREEN,
                        "Failure": RED
                    }
                )

                style_figure(fig, 340)

                fig.update_layout(
                    legend=dict(
                        orientation="h",
                        y=-0.08
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.caption(
                    "This chart describes historical dataset labels; it is not a live prediction."
                )

        else:

            st.warning(
                "The reference dataset was not found. "
                "Expected location: dataset/ai4i2020.csv"
            )

    else:

        result = st.session_state.latest_result

        sensors = st.session_state.latest_sensors or {}

        probability = get_probability(result)

        risk = str(
            result.get("risk_level", "Unknown")
        )

        failure_prediction = safe_int(
            result.get("failure_prediction", 0)
        )

        is_anomaly = bool(
            result.get("is_anomaly", False)
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            metric_card(
                "FAILURE PROBABILITY",
                f"{probability:.2f}%",
                "Model-estimated probability"
            )

        with c2:

            metric_card(
                "RISK LEVEL",
                risk.upper(),
                "Model-defined risk category"
            )

        with c3:

            metric_card(
                "FAILURE CLASS",
                "FAILURE" if failure_prediction == 1 else "NORMAL",
                str(result.get("failure_status", "Unknown"))
            )

        with c4:

            metric_card(
                "ANOMALY CHECK",
                "ANOMALY" if is_anomaly else "NORMAL",
                str(result.get("anomaly_status", "Unknown"))
            )

        left, right = st.columns([1.05, 1])

        with left:

            st.markdown("### Predicted Failure Probability")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",

                    value=probability,

                    number={
                        "suffix": "%",
                        "font": {
                            "color": TEXT,
                            "size": 38
                        }
                    },

                    title={
                        "text": "Model estimate",
                        "font": {
                            "color": MUTED,
                            "size": 14
                        }
                    },

                    gauge={

                        "axis": {
                            "range": [0, 100],
                            "tickcolor": MUTED,
                            "tickfont": {
                                "color": MUTED
                            }
                        },

                        "bar": {
                            "color": ACCENT
                        },

                        "bgcolor": PANEL,

                        "borderwidth": 0,

                        "steps": [
                            {
                                "range": [0, 30],
                                "color": "#164237"
                            },
                            {
                                "range": [30, 60],
                                "color": "#51401e"
                            },
                            {
                                "range": [60, 100],
                                "color": "#512631"
                            },
                        ],
                    },
                )
            )

            gauge.update_layout(
                height=310,
                paper_bgcolor="rgba(0,0,0,0)",
                font_color=TEXT,
                margin=dict(
                    l=25,
                    r=25,
                    t=55,
                    b=15
                )
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

            st.caption(
                "The value is the model's estimated probability for the supplied readings, not a measured failure rate."
            )

        with right:

            st.markdown("### Machine Condition")

            st.markdown(
                f'<span class="status-pill {risk_class(risk)}">'
                f'{html.escape(risk.upper())} RISK</span>',
                unsafe_allow_html=True
            )

            st.markdown("")

            info_card(
                "Failure prediction",
                str(
                    result.get(
                        "failure_status",
                        "No status returned by model."
                    )
                )
            )

            info_card(
                "Anomaly detection",
                str(
                    result.get(
                        "anomaly_status",
                        "No anomaly status returned by model."
                    )
                )
            )

            info_card(
                "How to interpret",
                "Failure prediction and anomaly detection are separate model outputs. An unusual pattern does not, by itself, confirm that a failure will occur."
            )

        st.markdown("### Sensor Snapshot")

        snapshot = pd.DataFrame(
            [
                (
                    "Machine type",
                    sensors.get("Type", "—")
                ),
                (
                    "Air temperature",
                    f'{sensors.get("Air temperature [K]", "—")} K'
                ),
                (
                    "Process temperature",
                    f'{sensors.get("Process temperature [K]", "—")} K'
                ),
                (
                    "Rotational speed",
                    f'{sensors.get("Rotational speed [rpm]", "—")} rpm'
                ),
                (
                    "Torque",
                    f'{sensors.get("Torque [Nm]", "—")} Nm'
                ),
                (
                    "Tool wear",
                    f'{sensors.get("Tool wear [min]", "—")} min'
                ),
            ],
            columns=[
                "Operating parameter",
                "Input used for prediction"
            ]
        )

        st.dataframe(
            snapshot,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Analysis timestamp: {st.session_state.latest_analysis_time or 'Not available'}"
        )


# ---------------------------------------------------------------------
# 9. SENSOR DIAGNOSTICS
# ---------------------------------------------------------------------

with tabs[1]:

    section_header(
        "02 / OPERATING PARAMETERS",
        "Sensor Diagnostics",
        "Inspect the five operating inputs used by the prediction model and review their current configured values.",
    )

    sensors = st.session_state.latest_sensors

    if sensors is None:

        st.info(
            "Sensor diagnostics will be populated after the first machine analysis. Current values can be configured in the sidebar."
        )

        sensors = {

            "Type": machine_type,

            "Air temperature [K]": air_temp,

            "Process temperature [K]": process_temp,

            "Rotational speed [rpm]": rotational_speed,

            "Torque [Nm]": torque,

            "Tool wear [min]": tool_wear,

        }

        source_label = (
            "Current sidebar settings — analysis not yet run"
        )

    else:

        source_label = (
            "Last analyzed sensor readings"
        )

    st.caption(source_label)

    sensor_specs = [

        (
            "Air temperature",
            "Air temperature [K]",
            "K",
            "Ambient temperature around the machine."
        ),

        (
            "Process temperature",
            "Process temperature [K]",
            "K",
            "Temperature measured during the machine process."
        ),

        (
            "Rotational speed",
            "Rotational speed [rpm]",
            "rpm",
            "Rotational speed of the machine's rotating component."
        ),

        (
            "Torque",
            "Torque [Nm]",
            "Nm",
            "Rotational force or load applied to the machine."
        ),

        (
            "Tool wear",
            "Tool wear [min]",
            "min",
            "Accumulated tool operating time represented in minutes."
        ),

    ]

    sensor_cols = st.columns(5)

    for col, (label, key, unit, description) in zip(
        sensor_cols,
        sensor_specs
    ):

        with col:

            metric_card(
                label.upper(),
                f'{sensors.get(key, "—")}',
                unit
            )

    st.markdown("### Parameter Reference")

    reference_df = pd.DataFrame(
        [
            {
                "Parameter": label,
                "Current reading": sensors.get(key, "—"),
                "Unit": unit,
                "Description": description
            }

            for label, key, unit, description in sensor_specs
        ]
    )

    st.dataframe(
        reference_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Operating Readings")

    readings = pd.DataFrame({

        "Parameter": [
            x[0] for x in sensor_specs
        ],

        "Value": [
            float(sensors.get(x[1], 0))
            for x in sensor_specs
        ],

        "Unit": [
            x[2] for x in sensor_specs
        ],

    })

    fig = px.bar(
        readings,
        x="Parameter",
        y="Value",
        color="Parameter",
        text="Value"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
        cliponaxis=False
    )

    style_figure(fig, 390)

    fig.update_layout(
        showlegend=False,
        xaxis_title="Sensor",
        yaxis_title="Recorded numerical value"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "The sensors use different physical units and scales. The chart is a display of raw readings, not a comparison of relative importance."
    )


# ---------------------------------------------------------------------
# 10. MODEL INTELLIGENCE
# ---------------------------------------------------------------------

with tabs[2]:

    section_header(
        "03 / MODEL PERFORMANCE",
        "Model Intelligence",
        "Review the prediction workflow, training evaluation results and the feature-importance information available in the project.",
    )

    st.markdown("### Prediction Workflow")

    workflow = st.columns(4)

    workflow_items = [

        (
            "01 · INPUT",
            "Sensor parameters",
            "Five numeric readings and the machine type are prepared for inference."
        ),

        (
            "02 · CLASSIFICATION",
            "XGBoost model",
            "The trained classifier estimates the machine-failure probability."
        ),

        (
            "03 · ANOMALY",
            "Isolation Forest",
            "A separate model checks whether the input pattern is unusual."
        ),

        (
            "04 · EXPLANATION",
            "Feature importance",
            "Saved feature-importance results show model-level input influence."
        ),

    ]

    for col, (eyebrow, title, desc) in zip(
        workflow,
        workflow_items
    ):

        with col:

            info_card(
                eyebrow + "  |  " + title,
                desc
            )

    st.divider()

    st.markdown("### Model Evaluation Results")

    if not comparison_df.empty:

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        metric_candidates = [

            m

            for m in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "PR-AUC"
            ]

            if m in comparison_df.columns

        ]

        if "Model" in comparison_df.columns and metric_candidates:

            default_metrics = [

                m

                for m in [
                    "PR-AUC",
                    "Recall",
                    "F1 Score"
                ]

                if m in metric_candidates

            ]

            selected_metrics = st.multiselect(
                "Metrics to display",
                metric_candidates,
                default=default_metrics or metric_candidates
            )

            if selected_metrics:

                chart_data = comparison_df.melt(
                    id_vars="Model",
                    value_vars=selected_metrics,
                    var_name="Metric",
                    value_name="Score"
                )

                chart_data["Score"] = pd.to_numeric(
                    chart_data["Score"],
                    errors="coerce"
                )

                fig = px.bar(
                    chart_data,
                    x="Model",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    text_auto=".3f"
                )

                style_figure(fig, 420)

                fig.update_layout(
                    yaxis_title="Evaluation score",
                    xaxis_title="Model",
                    yaxis=dict(
                        range=[0, 1.05],
                        gridcolor="#26394f"
                    )
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        st.markdown("### Metric Guide")

        metric_help = {

            "Accuracy":
                "Share of all evaluated predictions that were correct.",

            "Precision":
                "Share of predicted failures that were actual failures.",

            "Recall":
                "Share of actual failures correctly identified by the model.",

            "F1 Score":
                "Harmonic mean of precision and recall.",

            "PR-AUC":
                "Area under the precision-recall curve; useful when positive failure cases are relatively uncommon.",

        }

        for metric in metric_candidates:

            info_card(
                metric,
                metric_help.get(
                    metric,
                    "Evaluation metric recorded in the model comparison file."
                )
            )

        st.caption(
            "These are saved evaluation results from model training/testing, not a live performance measurement."
        )

    else:

        st.warning(
            "Model comparison file is unavailable. Expected: results/model_comparison.csv"
        )

    st.divider()

    st.markdown("### Feature Importance")

    if (
        not importance_df.empty
        and {"Feature", "Importance"}.issubset(importance_df.columns)
    ):

        imp = importance_df.copy()

        imp["Importance"] = pd.to_numeric(
            imp["Importance"],
            errors="coerce"
        )

        imp = imp.dropna(
            subset=["Importance"]
        ).sort_values(
            "Importance",
            ascending=True
        )

        if not imp.empty:

            fig = px.bar(
                imp,
                x="Importance",
                y="Feature",
                orientation="h",
                text="Importance"
            )

            fig.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside",
                cliponaxis=False
            )

            style_figure(
                fig,
                max(360, 48 * len(imp))
            )

            fig.update_layout(
                xaxis_title="Importance value",
                yaxis_title="Feature"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "Feature importance reflects the saved model-level importance values. It does not establish causation."
            )

    elif not importance_df.empty:

        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "The feature-importance file does not contain the expected Feature and Importance columns, so it is shown as a table."
        )

    else:

        st.warning(
            "Feature importance file is unavailable. Expected: results/feature_importance.csv"
        )


# ---------------------------------------------------------------------
# 11. MAINTENANCE INTELLIGENCE
# ---------------------------------------------------------------------

with tabs[3]:

    section_header(
        "04 / MAINTENANCE PLANNING",
        "Maintenance Intelligence",
        "Translate the latest model outputs and sensor readings into practical inspection and monitoring priorities.",
    )

    result = st.session_state.latest_result

    if result is None:

        info_card(
            "No completed assessment",
            "Run Machine Analysis from the sidebar to generate maintenance insights for the configured readings."
        )

    else:

        probability = get_probability(result)

        risk = str(
            result.get("risk_level", "Unknown")
        )

        recommendations = result.get(
            "recommendations",
            []
        )

        if isinstance(recommendations, str):
            recommendations = [recommendations]

        if not isinstance(recommendations, (list, tuple)):
            recommendations = []

        a, b = st.columns(2)

        with a:

            st.markdown("### Assessment Summary")

            st.markdown(
                f'<span class="status-pill {risk_class(risk)}">'
                f'{html.escape(risk.upper())} MODEL RISK</span>',
                unsafe_allow_html=True
            )

            st.markdown("")

            metric_card(
                "FAILURE PROBABILITY",
                f"{probability:.2f}%",
                "Model-estimated probability"
            )

            st.markdown("")

            info_card(
                "Failure classification",
                str(
                    result.get(
                        "failure_status",
                        "Unknown"
                    )
                )
            )

            info_card(
                "Anomaly assessment",
                str(
                    result.get(
                        "anomaly_status",
                        "Unknown"
                    )
                )
            )

        with b:

            st.markdown("### Operational Priority")

            if (
                "high" in risk.lower()
                or "critical" in risk.lower()
            ):

                st.error(
                    "Elevated model risk. Arrange a qualified inspection and review the machine's operating conditions."
                )

            elif (
                "medium" in risk.lower()
                or "moderate" in risk.lower()
            ):

                st.warning(
                    "Moderate model risk. Increase monitoring and consider a planned maintenance inspection."
                )

            else:

                st.success(
                    "Low model risk. Continue routine monitoring and follow the established maintenance schedule."
                )

            info_card(
                "Decision support",
                "Use these outputs alongside physical inspection, maintenance records and site safety procedures. The model does not replace an engineer's assessment."
            )

        st.divider()

        st.markdown("### Recommended Maintenance Actions")

        if recommendations:

            for index, recommendation in enumerate(
                recommendations,
                start=1
            ):

                info_card(
                    f"ACTION {index:02d}",
                    str(recommendation)
                )

        else:

            info_card(
                "Routine monitoring",
                "No additional recommendation was returned by the prediction module. Continue the approved maintenance schedule."
            )

    st.divider()

    st.markdown("### Failure Categories in the AI4I Dataset")

    failure_modes = [

        (
            "TWF",
            "Tool Wear Failure",
            "Failure category associated with tool wear."
        ),

        (
            "HDF",
            "Heat Dissipation Failure",
            "Failure category associated with thermal conditions."
        ),

        (
            "PWF",
            "Power Failure",
            "Failure category associated with power-related operating conditions."
        ),

        (
            "OSF",
            "Overstrain Failure",
            "Failure category associated with excessive mechanical strain."
        ),

        (
            "RNF",
            "Random Failure",
            "Random failure category recorded in the dataset."
        ),

    ]

    mode_cols = st.columns(5)

    for col, (code, title, desc) in zip(
        mode_cols,
        failure_modes
    ):

        with col:

            info_card(
                code + " · " + title,
                desc
            )

    st.caption(
        "These categories are represented in the reference dataset. The current prediction function does not return an individual failure-mode prediction."
    )


# ---------------------------------------------------------------------
# 12. REPORTS AND INSIGHTS
# ---------------------------------------------------------------------

with tabs[4]:

    section_header(
        "05 / REPORTING AND ANALYTICS",
        "Reports & Insights",
        "Review the current machine assessment, export reports and inspect analysis history and reference dataset characteristics.",
    )

    result = st.session_state.latest_result

    sensors = st.session_state.latest_sensors

    st.markdown("### Current Machine Analysis")

    if result is not None and sensors is not None:

        probability = get_probability(result)

        analysis_time = (
            st.session_state.latest_analysis_time
            or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        report_rows = [

            ("Analysis timestamp", analysis_time),

            ("Machine type", sensors.get("Type", "—")),

            ("Air temperature (K)", sensors.get("Air temperature [K]", "—")),

            ("Process temperature (K)", sensors.get("Process temperature [K]", "—")),

            ("Rotational speed (rpm)", sensors.get("Rotational speed [rpm]", "—")),

            ("Torque (Nm)", sensors.get("Torque [Nm]", "—")),

            ("Tool wear (min)", sensors.get("Tool wear [min]", "—")),

            ("Failure probability (%)", round(probability, 2)),

            ("Failure prediction", result.get("failure_prediction", "—")),

            ("Failure status", result.get("failure_status", "Unknown")),

            ("Risk level", result.get("risk_level", "Unknown")),

            ("Anomaly status", result.get("anomaly_status", "Unknown")),

        ]

        report_df = pd.DataFrame(
            report_rows,
            columns=["Report field", "Value"]
        )

        st.dataframe(
            report_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Download Machine Analysis (CSV)",
            data=to_csv_bytes(report_df),
            file_name="machine_analysis_report.csv",
            mime="text/csv"
        )

        recommendations = result.get(
            "recommendations",
            []
        )

        if isinstance(recommendations, str):
            recommendations = [recommendations]

        recommendations_text = (
            "\n".join(
                f"- {item}"
                for item in recommendations
            )
            if recommendations
            else "- Continue routine monitoring and follow the approved maintenance schedule."
        )

        detailed_report = (

            "INDUSTRIAL PREDICTIVE MAINTENANCE\n"
            "MACHINE ANALYSIS REPORT\n"

            + "=" * 42 + "\n\n"

            + "ANALYSIS DETAILS\n"

            + f"Timestamp: {analysis_time}\n\n"

            + "MACHINE PARAMETERS\n"

            + f"Machine type: {sensors.get('Type', '—')}\n"

            + f"Air temperature: {sensors.get('Air temperature [K]', '—')} K\n"

            + f"Process temperature: {sensors.get('Process temperature [K]', '—')} K\n"

            + f"Rotational speed: {sensors.get('Rotational speed [rpm]', '—')} rpm\n"

            + f"Torque: {sensors.get('Torque [Nm]', '—')} Nm\n"

            + f"Tool wear: {sensors.get('Tool wear [min]', '—')} min\n\n"

            + "MODEL OUTPUTS\n"

            + f"Failure probability: {probability:.2f}%\n"

            + f"Failure prediction: {result.get('failure_prediction', '—')}\n"

            + f"Failure status: {result.get('failure_status', 'Unknown')}\n"

            + f"Risk level: {result.get('risk_level', 'Unknown')}\n"

            + f"Anomaly status: {result.get('anomaly_status', 'Unknown')}\n\n"

            + "MAINTENANCE RECOMMENDATIONS\n"

            + recommendations_text

            + "\n\n"

            + "Note: This report contains model-generated estimates based on the supplied sensor readings. It is decision support and does not replace physical inspection or qualified engineering judgement.\n"

        )

        st.download_button(
            "⬇ Download Detailed Report (TXT)",
            data=detailed_report.encode("utf-8"),
            file_name="machine_maintenance_report.txt",
            mime="text/plain"
        )

    else:

        st.info(
            "No machine assessment is available yet. Run Machine Analysis to create downloadable reports."
        )

    st.divider()

    st.markdown("### Analysis History")

    if st.session_state.analysis_history:

        history_df = pd.DataFrame(
            st.session_state.analysis_history
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇ Download Analysis History (CSV)",
            data=to_csv_bytes(history_df),
            file_name="machine_analysis_history.csv",
            mime="text/csv"
        )

    else:

        st.caption(
            "Completed machine assessments will be listed here during this session."
        )

    st.divider()

    st.markdown("### Reference Dataset Overview")

    if not dataset_df.empty:

        missing_count = int(
            dataset_df.isna().sum().sum()
        )

        d1, d2, d3 = st.columns(3)

        with d1:

            metric_card(
                "TOTAL RECORDS",
                f"{len(dataset_df):,}",
                "Dataset observations"
            )

        with d2:

            metric_card(
                "TOTAL COLUMNS",
                str(len(dataset_df.columns)),
                "Available fields"
            )

        with d3:

            metric_card(
                "MISSING VALUES",
                f"{missing_count:,}",
                "Across all columns"
            )

        if "Machine failure" in dataset_df.columns:

            st.markdown(
                "### Historical Machine Failure Distribution"
            )

            labels = (
                pd.to_numeric(
                    dataset_df["Machine failure"],
                    errors="coerce"
                )
                .value_counts()
                .reindex([0, 1], fill_value=0)
            )

            distribution = pd.DataFrame({

                "Condition": [
                    "No failure",
                    "Failure"
                ],

                "Records": labels.values

            })

            fig = px.pie(
                distribution,
                names="Condition",
                values="Records",
                hole=0.58,
                color="Condition",
                color_discrete_map={
                    "No failure": GREEN,
                    "Failure": RED
                }
            )

            style_figure(fig, 360)

            fig.update_layout(
                legend=dict(
                    orientation="h",
                    y=-0.08
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "Historical labels from the supplied dataset. They are not the current machine's predicted probability."
            )

    else:

        st.warning(
            "Reference dataset was not found. Expected location: dataset/ai4i2020.csv"
        )


# ---------------------------------------------------------------------
# 13. DASHBOARD FOOTER
# ---------------------------------------------------------------------

st.divider()

st.markdown(
    '<p style="text-align:center;color:#9aabc0;'
    'font-size:11px;letter-spacing:.7px;">'
    'PREDICTIVE MAINTENANCE · MACHINE HEALTH ANALYTICS'
    '</p>',
    unsafe_allow_html=True,
)
