# src/explainability.py
import shap
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import pandas as pd
import numpy as np
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

    if df.empty or "combined_text" not in df:
        st.warning("No input data to explain.")
        return

    X = tfidf.transform(df["combined_text"])
    X_dense = X.toarray()
    X_subset = X_dense[:50]

    feature_names = tfidf.get_feature_names_out()
    X_df = pd.DataFrame(X_subset, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_subset, check_additivity=False)

    if isinstance(shap_values, list) and len(shap_values) > 1:
        values_to_plot = shap_values[1]
    else:
        values_to_plot = shap_values

    # Debugging info
    st.write("SHAP shape:", np.array(values_to_plot).shape)
    st.write("Non-zero SHAP values (sample):", np.count_nonzero(values_to_plot[0]))

    fig = plt.figure(figsize=(12, 6))
    shap.summary_plot(values_to_plot, X_df, show=False)
    plt.tight_layout()
    return fig
