import shap

# Fit SHAP explainer only once (after training)
explainer = shap.Explainer(model.predict_proba, X_vectorized)

# For new prediction (e.g. user-uploaded row)
shap_values = explainer(X_vectorized)

# Visualize
shap.plots.waterfall(shap_values[0])  # Explains one example
