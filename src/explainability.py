# src/explainability.py
import shap
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import joblib
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

def get_wordcloud(df, label=1):
    text = " ".join(
        df[df["Predicted Class"] == label]["combined_text"].fillna("No text")
    )
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wc

def get_shap_plot(df, sample_index=0):
    """Robust SHAP explanations with error handling for text classification"""
    try:
        # Load model and vectorizer
        model = joblib.load("model.pkl")
        tfidf = joblib.load("tfidf.pkl")
        
        # Prepare sample
        sample_text = df.iloc[sample_index]["combined_text"]
        X_sample = tfidf.transform([sample_text])
        
        # Create explainer with optimal parameters
        explainer = shap.TreeExplainer(
            model,
            data=None,
            feature_perturbation="interventional",  # Works better for text
            model_output="probability"  # Changed to probability
        )
        
        # Get SHAP values with all checks disabled
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = explainer.shap_values(
                X_sample,
                check_additivity=False,  # Disable problematic check
                approximate=True  # Use approximation for faster results
            )
        
        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Get fraud class values
        
        # Convert to dense array if sparse
        if hasattr(shap_values, 'toarray'):
            shap_values = shap_values.toarray()
        
        # Get top features
        feature_names = tfidf.get_feature_names_out()
        top_n = min(20, len(feature_names))  # Ensure we don't exceed features
        sorted_idx = np.argsort(-np.abs(shap_values[0]))[:top_n]
        top_features = [feature_names[i] for i in sorted_idx]
        top_values = [shap_values[0][i] for i in sorted_idx]
        
        # Create visualizations
        st.markdown("### 🔍 Top Predictive Words")
        
        # Bar plot of SHAP values
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['red' if v > 0 else 'green' for v in top_values]
        ax.barh(top_features[::-1], top_values[::-1], color=colors[::-1])
        ax.set_xlabel("SHAP Value (Impact on Fraud Probability)")
        proba = model.predict_proba(X_sample)[0][1]
        ax.set_title(f"Predicted Fraud Probability: {proba:.1%}")
        st.pyplot(fig)
        plt.close()
        
        # Highlight important words in text
        st.markdown("### 📝 Key Words in Context")
        text_display = sample_text
        for feat, val in zip(top_features[:10], top_values[:10]):
            highlight_color = "#ff9999" if val > 0 else "#99ff99"
            text_display = text_display.replace(
                feat,
                f"<span style='background-color: {highlight_color}; font-weight: bold'>{feat}</span>"
            )
        st.markdown(f"<div style='border:1px solid #eee; padding:10px;'>{text_display}</div>", 
                    unsafe_allow_html=True)
        
    except Exception as e:
        st.error("Couldn't generate full SHAP explanation. Showing simplified view:")
        
        # Fallback visualization
        model = joblib.load("model.pkl")
        tfidf = joblib.load("tfidf.pkl")
        X_sample = tfidf.transform([df.iloc[sample_index]["combined_text"]])
        
        # Get feature importance the simple way
        coefs = model.feature_importances_
        top_idx = np.argsort(-coefs)[:20]
        features = tfidf.get_feature_names_out()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(features[top_idx][::-1], coefs[top_idx][::-1])
        ax.set_xlabel("Feature Importance Score")
        st.pyplot(fig)
        plt.close()
        
        st.info("Top words from model's perspective (not sample-specific):")

def get_wordcloud(df, label=1):
    """Generate word cloud for specified class"""
    text = " ".join(df[df["Predicted Class"] == label]["combined_text"].fillna(""))
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    return wc
