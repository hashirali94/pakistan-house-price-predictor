import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

st.set_page_config(page_title="Pakistan House Price Predictor", page_icon="🏠")

CAT = ["property_type", "location", "city"]
NUM = ["baths", "bedrooms", "Area_in_Marla"]


def fmt_pkr(x):
    """Format a price in crore / lakh."""
    if x >= 1e7:
        return f"PKR {x / 1e7:.2f} crore"
    return f"PKR {x / 1e5:.1f} lakh"


@st.cache_resource(show_spinner="Training the model on first load (about 30 seconds)...")
def load_model():
    df = pd.read_csv("Cleaned_data_for_model.csv")
    df = df[df["purpose"] == "For Sale"].drop(columns=["Unnamed: 0", "purpose"])
    d = df[
        (df.price >= 500000)
        & (df.Area_in_Marla >= 1)
        & (df.Area_in_Marla <= 100)
        & (df.bedrooms >= 1)
    ].copy()

    X = d[CAT + NUM]
    y = np.log1p(d["price"])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    pre = ColumnTransformer(
        [("c", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)],
        remainder="passthrough",
    )
    model = Pipeline(
        [("pre", pre), ("gb", HistGradientBoostingRegressor(max_iter=300, random_state=42))]
    )
    model.fit(Xtr, ytr)

    pred = np.expm1(model.predict(Xte))
    actual = np.expm1(yte)
    stats = {
        "r2": r2_score(actual, pred),
        "mae": mean_absolute_error(actual, pred),
        "median_ape": float(np.median(np.abs(pred - actual) / actual)),
        "rows": len(d),
    }
    return model, d, stats


model, data, stats = load_model()

st.title("🏠 Pakistan House Price Predictor")
st.write(
    "Estimate the sale price of a property in five Pakistani cities. "
    "The model was trained on Zameen.com listings."
)

cities = sorted(data["city"].unique())
city = st.selectbox("City", cities, index=cities.index("Karachi") if "Karachi" in cities else 0)
locations = sorted(data.loc[data["city"] == city, "location"].unique())
location = st.selectbox("Area / Society", locations)
types = sorted(data["property_type"].unique())
ptype = st.selectbox("Property type", types, index=types.index("House") if "House" in types else 0)
area = st.number_input("Size (marla)", min_value=1.0, max_value=100.0, value=5.0, step=0.5)

c1, c2 = st.columns(2)
beds = c1.number_input("Bedrooms", min_value=1, max_value=10, value=3)
baths = c2.number_input("Bathrooms", min_value=1, max_value=10, value=2)

if st.button("Estimate price", type="primary"):
    row = pd.DataFrame(
        [
            {
                "property_type": ptype,
                "location": location,
                "city": city,
                "baths": baths,
                "bedrooms": beds,
                "Area_in_Marla": area,
            }
        ]
    )
    price = float(np.expm1(model.predict(row))[0])
    low = price * (1 - stats["median_ape"])
    high = price * (1 + stats["median_ape"])
    st.success(f"Estimated price: **{fmt_pkr(price)}**")
    st.write(
        f"Typical range: {fmt_pkr(low)} to {fmt_pkr(high)} "
        f"(based on the model's median error of about {stats['median_ape'] * 100:.0f}% on test data)."
    )

    # Reliability check: how many similar listings did the model learn from?
    same = data[
        (data["city"] == city)
        & (data["location"] == location)
        & (data["property_type"] == ptype)
    ]
    similar = same[same["Area_in_Marla"].between(area * 0.8, area * 1.2)]
    if len(same) < 20 or len(similar) < 5:
        st.warning(
            f"Lower confidence: the data has only {len(similar)} similar listings "
            f"({len(same)} for this area and property type). "
            "Treat this estimate as a rough guide."
        )
    else:
        st.caption(
            f"Based on {len(similar)} similar listings in this area "
            f"(median asking price of those listings: {fmt_pkr(similar['price'].median())})."
        )

st.divider()
st.subheader("About this model")
st.write(
    f"- Trained on **{stats['rows']:,}** cleaned sale listings (Zameen.com data from Kaggle)\n"
    f"- Model: Histogram Gradient Boosting Regressor\n"
    f"- Test R² score: **{stats['r2']:.3f}**\n"
    f"- Mean absolute error: about **{fmt_pkr(stats['mae'])}**"
)
st.caption(
    "Limitations: the data contains asking prices, not final sale prices, and prices change over time. "
    "This is a portfolio project, not valuation or investment advice."
)
st.markdown("[Code and notebook on GitHub](https://github.com/hashirali94/pakistan-house-price-predictor)")
