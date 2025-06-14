# src/explainability.py
import shap
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import pandas as pd
import joblib


def get_wordcloud(df, label=1):
    text = " ".join(
        df[df["Predicted Class"] == label]["combined_text"].fillna("No text")
    )
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wc

def get_shap_plot(df, model_path="model.pkl", tfidf_path="tfidf.pkl"):
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)

    X = tfidf.transform(df["combined_text"])
    X_dense = X.toarray()
    X_subset = X_dense[:50]  # Limit for speed

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_subset, check_additivity=False)

    # Class-based check for binary classifier
    if isinstance(shap_values, list) and len(shap_values) > 1:
        values_to_plot = shap_values[1]
    else:
        values_to_plot = shap_values

    feature_names = tfidf.get_feature_names_out()
    X_df = pd.DataFrame(X_subset, columns=feature_names)

    plt.figure(figsize=(10, 5))
    shap.summary_plot(values_to_plot, X_df, show=False)
    st.pyplot(plt.gcf())  # ✅ This is enough to render the SHAP plot