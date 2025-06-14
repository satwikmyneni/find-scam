import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
import joblib
from src.preprocessing import preprocess

# Load data
df = pd.read_csv("data/train.csv")
df = preprocess(df)

X = df["combined_text"]
y = df["fraudulent"]

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Handle imbalance
ros = RandomOverSampler()
X_res, y_res = ros.fit_resample(X_tfidf, y)

# Split
X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_val)
f1 = f1_score(y_val, y_pred)
print(f"✅ Model trained with F1 Score: {f1:.4f}")

# Save model and vectorizer
joblib.dump(clf, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
