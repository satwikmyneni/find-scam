# shap_explainer.py
import shap
import pandas as pd
import joblib

def explain_shap(X, model, tfidf, idx=0):
    explainer = shap.Explainer(model, tfidf.transform)
    shap_values = explainer(X["combined_text"])
    return shap_values[idx]  # explain the nth job posting
