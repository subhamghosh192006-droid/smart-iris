
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Smart Iris",
    page_icon="🌸",
    layout="centered"
)

@st.cache_resource
def train_model():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, iris, accuracy

model, iris, accuracy = train_model()

st.title("🌸 Smart Iris")
st.subheader("AI-Powered Iris Flower Classifier")
st.write(
    "Predict the species of an Iris flower using "
    "a machine learning model."
)

st.metric("Model Test Accuracy", f"{accuracy:.1%}")
st.divider()

st.header("🔍 Predict a Flower")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)

with col2:
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict Iris Species", type="primary"):
    features = [[
        sepal_length, sepal_width, petal_length, petal_width
    ]]
    prediction = model.predict(features)[0]
    species = iris.target_names[prediction]

    st.success(f"Predicted species: {species.title()}")

    probabilities = model.predict_proba(features)[0]
    st.subheader("Prediction probabilities")
    chart_data = pd.DataFrame({
        "Species": [name.title() for name in iris.target_names],
        "Probability": probabilities
    }).set_index("Species")
    st.bar_chart(chart_data)

st.divider()
st.caption("Built with Python, Streamlit and scikit-learn.")
