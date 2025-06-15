import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from src.preprocessing import preprocess

# --- Page Config ---
st.set_page_config(page_title="Spot the Scam", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main-title {
        font-size: 48px;
        font-weight: bold;
        color: #0e1117;
        margin-bottom: 5px;
    }
    .tagline {
        font-size: 20px;
        color: #6c757d;
        margin-top: 0px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 14px;
        color: #888;
    }
    </style>
""", unsafe_allow_html=True)

# --- Hero Header ---
st.markdown("<div class='main-title'>🛡️ Spot the Scam</div>", unsafe_allow_html=True)
st.markdown("<div class='tagline'>Protecting job seekers from fraud using AI</div>", unsafe_allow_html=True)
st.markdown("---")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    threshold = st.slider("🔍 Fraud Probability Threshold", 0.0, 1.0, 0.5, 0.01)
    show_wordcloud = st.checkbox("☁️ Show Word Cloud", value=True)
    show_shap = st.checkbox("📊 Show SHAP Explainability", value=False)
    st.markdown("---")
    st.markdown("Made with ❤️ for the Hackathon")

# --- File Upload ---
uploaded_file = st.file_uploader("📤 Upload a job postings CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    raw_df = df.copy()
    df = preprocess(df)

    model = joblib.load("model.pkl")
    tfidf = joblib.load("tfidf.pkl")

    X = tfidf.transform(df["combined_text"])
    probs = model.predict_proba(X)[:, 1]
    preds = (probs > threshold).astype(int)

    df["Fraud Probability"] = probs
    df["Predicted Class"] = preds
    df["Label"] = np.where(preds == 1, "🚨 Fraud", "✅ Genuine")
    df["Fraud Probability %"] = (probs * 100).round(2).astype(str) + "%"

    st.success(f"✅ Processed {len(df)} job postings")

    export_df = raw_df.copy()
    export_df["Fraud Probability"] = probs
    export_df["Predicted Class"] = preds
    export_df["Label"] = df["Label"]
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results as CSV", csv, "scam_predictions.csv", "text/csv")

    st.subheader("📋 Enhanced Job Listings Review")
    for i, row in df.iterrows():
        with st.expander(f"🔎 {row['title']}"):
            st.markdown(f"""
            **📌 Status:** {'`FRAUD 🚨`' if row['Predicted Class'] == 1 else '`GENUINE ✅`'}  
            **📊 Probability of Fraud:** `{row['Fraud Probability']:.2%}`  
            **📍 Location:** {raw_df.loc[i, 'location'] if 'location' in raw_df.columns else 'N/A'}  
            **📝 Description:**  
            {raw_df.loc[i, 'description'] if 'description' in raw_df.columns else '*No description provided.*'}
            """)

    st.subheader("📊 Summary Table (Sorted by Fraud Score)")
    styled_df = df[["title", "Label", "Fraud Probability %"]].sort_values("Fraud Probability %", ascending=False).reset_index(drop=True)
    st.dataframe(styled_df, use_container_width=True)

    st.subheader("🚨 Top 10 Most Suspicious Jobs")
    top10 = df.sort_values("Fraud Probability", ascending=False).head(10)
    for _, row in top10.iterrows():
        st.markdown(f"**{row['title']}** — {row['Fraud Probability']:.2%}")
        color = "red" if row['Fraud Probability'] > 0.7 else "orange" if row['Fraud Probability'] > 0.4 else "green"
        filled = int(row['Fraud Probability'] * 100)
        components.html(f"""
        <div style='background:#eee;width:100%;border-radius:5px'>
            <div style='width:{filled}%;background:{color};padding:4px;color:white;border-radius:5px;text-align:center;'>
                {filled}%
            </div>
        </div>
        """, height=35)

    st.subheader("📈 Fraud Probability Distribution")
    st.plotly_chart(px.histogram(df, x="Fraud Probability", nbins=20, title="Probability Histogram"), use_container_width=True)

    st.subheader("🎯 Fake vs Genuine Breakdown")
    st.plotly_chart(px.pie(df, names="Label", title="Class Breakdown"), use_container_width=True)

    if show_wordcloud:
        st.subheader("☁️ Word Cloud of Job Descriptions")
        text = " ".join(raw_df["description"].dropna().astype(str).values)
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)


    st.markdown("<div class='footer'>🚀 Built by Data Whisperers for the ANVESHAN Hackathon 2025 • IITG 🧠</div>", unsafe_allow_html=True)

else:
    st.info("👈 Please upload a CSV file to begin fraud detection.")
