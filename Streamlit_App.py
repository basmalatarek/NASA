import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="AstroVision 🚀", layout="wide")

# ================== BACKGROUND CSS ==================
page_bg = """
<style>
@keyframes moveSpace {
    from {background-position: 0 0;}
    to {background-position: 10000px 5000px;}
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at bottom, #0d1b2a 0%, #000000 100%);
    background-image: url('https://www.transparenttextures.com/patterns/stardust.png');
    animation: moveSpace 300s linear infinite;
    color: white;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
h1 {
    text-align: center;
    font-size: 4em;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: glow 3s ease-in-out infinite alternate;
}
@keyframes glow {
    from { text-shadow: 0 0 10px #00c6ff; }
    to { text-shadow: 0 0 20px #0072ff; }
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ================== APP TITLE ==================
st.markdown("<h1>🚀 AstroVision</h1>", unsafe_allow_html=True)
st.markdown("### 🌌 Exploring Exoplanets with Machine Learning")

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    df = pd.read_csv("df_new.csv")
    return df

df = load_data()

# ================== LOAD MODEL ==================
@st.cache_data
def load_model():
    model = joblib.load("best_model.pkl")
    return model

model = load_model()

# ================== NAVIGATION ==================
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🔍 Visualization", "🤖 Prediction"])

# ================== TAB 1 - DATA OVERVIEW ==================
with tab1:
    st.subheader("Dataset Overview")
    st.dataframe(df.head(20))
    st.write("### Dataset Summary")
    st.write(df.describe())

# ================== TAB 2 - VISUALIZATION ==================
with tab2:
    st.subheader("Feature Distribution")
    feature = st.selectbox("Select a Feature to Visualize:", [col for col in df.columns if col != "koi_disposition"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[feature], bins=30, color="skyblue", edgecolor="white")
    ax.set_title(f"Distribution of {feature}", color="white")
    ax.set_xlabel(feature)
    ax.set_ylabel("Frequency")
    fig.patch.set_alpha(0)
    st.pyplot(fig)

# ================== TAB 3 - MODEL & PREDICTION ==================
with tab3:
    st.subheader("HistGradientBoosting Model")

    if "koi_disposition" in df.columns:
        # ================== Prepare Target ==================
        y_raw = df["koi_disposition"].astype(str)
        le = LabelEncoder()
        y = le.fit_transform(y_raw)  # convert labels to numbers

        # ================== Prepare Features ==================
        X_full = pd.get_dummies(df.drop(columns=["koi_disposition"]))
        X_full = X_full[model.feature_names_in_]  # match model features

        X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.write(f"### ✅ Model Accuracy: {acc:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=le.classes_))

    st.write("---")
    st.subheader("🔮 Try Custom Prediction (Top Features Only)")

    # ================== TOP FEATURES INPUT ==================
    top_features = [
        "koi_score", "koi_max_mult_ev", "dec", "koi_count", 
        "koi_fpflag_nt", "koi_duration", "koi_fpflag_co", 
        "koi_fpflag_ss", "ra", "koi_dicco_msky"
    ]

    input_data = []
    for col in top_features:
        val = st.number_input(f"Enter {col}:", value=float(df[col].mean()))
        input_data.append(val)

    # ================== CREATE INPUT DF FOR PREDICTION ==================
    input_df = pd.DataFrame([input_data], columns=top_features)

    # Add remaining columns from model as 0
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match model
    input_df = input_df[model.feature_names_in_]

    if st.button("🚀 Predict"):
        prediction_encoded = model.predict(input_df)
        prediction_label = le.inverse_transform(prediction_encoded)  # convert back to original label
        st.success(f"🌠 Predicted: {prediction_label[0]}")
