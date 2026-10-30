from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "housing_price_model.joblib"

st.set_page_config(
    page_title="HouseWise | Housing Price Predictor",
    page_icon="H",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
        --ink: #edf4f5;
        --muted: #9aadb2;
        --panel: #111d23;
        --line: rgba(175, 205, 207, .15);
        --accent: #52d3bd;
    }
    html, body, [class*="css"] { font-family: Inter, "Segoe UI", sans-serif; }
    .stApp { background: radial-gradient(ellipse at 50% 0%, #17343a, #0b151a 48%, #091115); color: var(--ink); }
    [data-testid="stHeader"], [data-testid="stSidebar"], #MainMenu, footer { visibility: hidden; }
    .block-container { max-width: 860px; padding: 3.2rem 1.5rem 4rem; }
    h1, h2, h3 { letter-spacing: -.04em; }
    .brand { color: var(--accent); font-size: .76rem; font-weight: 750; letter-spacing: .2em; }
    .hero { text-align: center; padding: 1.1rem 0 2.1rem; }
    .hero h1 { color: #f4f8f8; font-size: clamp(2.6rem, 8vw, 4.2rem); margin: .65rem 0 .35rem; }
    .hero .subtitle { color: #bdd0d1; font-size: 1.02rem; margin-bottom: .8rem; }
    .hero .tagline { color: var(--muted); font-size: 1rem; }
    div[data-testid="stForm"] {
        background: rgba(17, 29, 35, .9); border: 1px solid var(--line);
        border-radius: 20px; padding: 1.45rem 1.6rem 1.25rem;
    }
    div[data-testid="stForm"] label { color: #e2ecec; font-weight: 600; }
    div[data-testid="stForm"] [data-testid="stCaptionContainer"] { color: var(--muted); }
    .stNumberInput input { background: #0c171c; border-color: var(--line); border-radius: 9px; }
    div[data-testid="stFormSubmitButton"] button {
        min-height: 3.2rem; border: 0; border-radius: 10px;
        background: linear-gradient(100deg, #42cbb6, #79dfbb);
        color: #071917; font-size: .86rem; font-weight: 800; letter-spacing: .08em;
    }
    div[data-testid="stFormSubmitButton"] button:hover { color: #071917; filter: brightness(1.06); }
    .result {
        margin-top: 1.5rem; padding: 2rem 1.5rem; text-align: center;
        background: linear-gradient(145deg, rgba(25, 55, 59, .96), rgba(16, 31, 36, .98));
        border: 1px solid rgba(82, 211, 189, .42); border-radius: 20px;
    }
    .result-label { color: #a9c4c4; font-size: .76rem; font-weight: 750; letter-spacing: .16em; }
    .result-price { color: #6de0ca; font-size: clamp(2.2rem, 8vw, 3.7rem); font-weight: 800; letter-spacing: -.05em; margin: .5rem 0; }
    .result-note, .disclaimer { color: var(--muted); font-size: .86rem; }
    .disclaimer { margin: 1rem auto 0; max-width: 650px; text-align: center; line-height: 1.6; }
    @media (max-width: 600px) {
        .block-container { padding: 2rem 1rem 3rem; }
        div[data-testid="stForm"] { padding: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifact():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run python train_model.py first."
        )
    return joblib.load(MODEL_PATH)


artifact = load_artifact()
model = artifact["pipeline"]
metrics = artifact["metrics"]
features = metrics["features"]
expected_features = [
    "Overall Qual",
    "Gr Liv Area",
    "Bedroom AbvGr",
    "Full Bath",
    "Year Built",
    "Garage Cars",
    "Neighborhood",
]
if features != expected_features:
    raise ValueError(
        "The saved model must use the seven prediction features in the expected order. "
        "Run python train_model.py to rebuild it."
    )

defaults = artifact["defaults"]
neighborhoods = artifact["options"]["Neighborhood"]

st.markdown(
    """
    <header class="hero">
      <div class="brand">HOUSEWISE</div>
      <h1>Housing Price Predictor</h1>
      <div class="subtitle">Smart Housing Price Prediction</div>
      <div class="tagline">Find out what your home could be worth.</div>
    </header>
    """,
    unsafe_allow_html=True,
)

with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        overall_quality = st.number_input(
            "Overall Quality",
            min_value=1,
            max_value=10,
            value=int(defaults["Overall Qual"]),
            step=1,
            help="Overall material and finish quality, rated from 1 to 10.",
        )
        living_area = st.number_input(
            "Living Area (sq ft)",
            min_value=334,
            max_value=5642,
            value=int(defaults["Gr Liv Area"]),
            step=50,
            help="Above-ground finished living area in square feet.",
        )
        bedrooms = st.number_input(
            "Total Bedrooms",
            min_value=0,
            max_value=8,
            value=int(defaults["Bedroom AbvGr"]),
            step=1,
            help="Number of bedrooms above ground.",
        )
    with right:
        full_bathrooms = st.number_input(
            "Full Bathrooms",
            min_value=0,
            max_value=4,
            value=int(defaults["Full Bath"]),
            step=1,
            help="Number of full bathrooms.",
        )
        year_built = st.number_input(
            "Year Built",
            min_value=1872,
            max_value=2010,
            value=int(defaults["Year Built"]),
            step=1,
            help="Year the home was originally constructed.",
        )
        garage_capacity = st.number_input(
            "Garage Capacity (cars)",
            min_value=0,
            max_value=5,
            value=int(defaults["Garage Cars"]),
            step=1,
            help="Number of vehicles the garage can accommodate.",
        )
        neighborhood = st.selectbox(
            "Locality / Area",
            options=neighborhoods,
            index=neighborhoods.index(defaults["Neighborhood"]),
            help="Choose the home's neighborhood from areas represented in the training dataset.",
        )
    previous_owners = st.number_input(
        "Number of Previous Owners",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
        help="Informational only. The training dataset does not record previous-owner counts.",
    )
    st.caption(
        "Previous-owner count is not used in the price prediction because it is not available in the training data."
    )
    submitted = st.form_submit_button(
        "PREDICT HOUSE PRICE", width="stretch"
    )

if submitted:
    values = [
        overall_quality,
        living_area,
        bedrooms,
        full_bathrooms,
        year_built,
        garage_capacity,
        neighborhood,
    ]
    if (
        len(values) != len(features)
        or any(value is None for value in values)
        or previous_owners < 0
    ):
        st.error("Please provide a valid value for each house detail.")
    else:
        input_data = pd.DataFrame([values], columns=features)
        prediction = float(model.predict(input_data)[0])
        if not pd.notna(prediction):
            raise ValueError("The trained model returned a non-finite prediction.")
        st.markdown(
            f"""
            <section class="result">
              <div class="result-label">YOUR ESTIMATED HOUSE PRICE</div>
              <div class="result-price">${prediction:,.0f}</div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="disclaimer">This is an estimated market value based on historical Ames, Iowa home sales. '
            'It is not a guaranteed selling price or a formal appraisal.</p>',
            unsafe_allow_html=True,
        )
