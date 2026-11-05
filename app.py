import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="centered",
)

# ---------------------------------------------------------------------
# Load the model and recreate the scaler used during training.
#
# In the notebook, RobustScaler was fitted on X_train after this exact
# train_test_split. The scaler itself was not saved, so we recreate it
# from the same Iris dataset and split to keep preprocessing consistent.
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("iris_model.pkl")

    iris = load_iris()
    X_train, _, _, _ = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    scaler = RobustScaler()
    scaler.fit(X_train)

    return model, scaler, iris


try:
    model, scaler, iris = load_artifacts()
except FileNotFoundError:
    st.error(
        "The model file `iris_model.pkl` was not found. "
        "Place it in the same folder as `app.py`."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("Iris Species Predictor")
st.write(
    "Enter the flower measurements below and the trained Logistic "
    "Regression model will predict its Iris species."
)

st.caption("Model: Logistic Regression · Preprocessing: RobustScaler")


# ---------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------
feature_config = [
    {
        "name": "Sepal length",
        "unit": "cm",
        "min": 4.0,
        "max": 8.0,
        "default": 5.8,
        "step": 0.1,
    },
    {
        "name": "Sepal width",
        "unit": "cm",
        "min": 2.0,
        "max": 4.5,
        "default": 3.0,
        "step": 0.1,
    },
    {
        "name": "Petal length",
        "unit": "cm",
        "min": 1.0,
        "max": 7.0,
        "default": 4.0,
        "step": 0.1,
    },
    {
        "name": "Petal width",
        "unit": "cm",
        "min": 0.1,
        "max": 2.5,
        "default": 1.2,
        "step": 0.1,
    },
]

with st.form("prediction_form"):
    st.subheader("Flower measurements")

    col1, col2 = st.columns(2)
    values = []

    for index, feature in enumerate(feature_config):
        target_col = col1 if index % 2 == 0 else col2
        with target_col:
            value = st.slider(
                f"{feature['name']} ({feature['unit']})",
                min_value=float(feature["min"]),
                max_value=float(feature["max"]),
                value=float(feature["default"]),
                step=float(feature["step"]),
            )
            values.append(value)

    submitted = st.form_submit_button(
        "Predict species",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
if submitted:
    input_data = pd.DataFrame(
        [values],
        columns=iris.feature_names,
    )

    scaled_input = scaler.transform(input_data)
    prediction = int(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]

    predicted_species = iris.target_names[prediction]
    confidence = float(probabilities[prediction])

    st.divider()
    st.subheader("Prediction")

    st.success(f"Predicted species: **{predicted_species.title()}**")
    st.metric("Model confidence", f"{confidence:.1%}")

    probability_df = pd.DataFrame(
        {
            "Species": [name.title() for name in iris.target_names],
            "Probability": probabilities,
        }
    ).sort_values("Probability", ascending=False)

    st.write("### Class probabilities")
    st.bar_chart(
        probability_df.set_index("Species"),
        y="Probability",
        use_container_width=True,
    )

    st.write("### Submitted measurements")
    display_df = input_data.T.reset_index()
    display_df.columns = ["Measurement", "Value (cm)"]
    st.dataframe(display_df, hide_index=True, use_container_width=True)

    st.caption(
        "The confidence value is the model's predicted class probability. "
        "It should not be interpreted as a guarantee that the prediction is correct."
    )
else:
    st.info("Adjust the measurements, then select **Predict species**.")


# ---------------------------------------------------------------------
# Optional dataset information
# ---------------------------------------------------------------------
with st.expander("About the Iris dataset"):
    st.write(
        "The Iris dataset contains measurements from three Iris species: "
        "Setosa, Versicolor, and Virginica. Each sample has four features: "
        "sepal length, sepal width, petal length, and petal width."
    )
    st.write(f"Number of samples: **{len(iris.data)}**")
    st.write(f"Number of features: **{iris.data.shape[1]}**")
