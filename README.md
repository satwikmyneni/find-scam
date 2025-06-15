# 🕵️ Spot the Scam

This project aims to detect **fraudulent job postings** using machine learning. It was built as part of a data science internship/minor to learn how to work with real-world datasets, handle text data, and build interpretable ML models.

We trained a binary classifier using job descriptions from a Kaggle dataset and deployed an interactive Streamlit dashboard to test new inputs in real-time and visualize feature importance using SHAP plots.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## 🔍 What it Does

- Classifies job descriptions as **fraudulent** or **genuine**
- Lets you input job details or upload a CSV
- Shows **SHAP-based** explanation of model predictions
- Achieved **F1 score: 0.9993** on validation data

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📁 Folder Structure
spot-the-scam/
│
├── app.py # Streamlit app
├── train_model.py # Model training script
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── model.pkl # Trained model
├── tfidf.pkl # Trained TF-IDF vectorizer
│
├── data/
│ ├── train.csv # Dataset (Kaggle)
│ └── test.csv # Optional
│
├── src/
│ ├── preprocessing.py # Cleaning and preprocessing logic
│ ├── explainability.py # SHAP explainability code
│ └── model.py # (Optional) model utilities

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📊 Dataset Info

- **Source**: [Kaggle - Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Total records**: 17,680
- **Label**: `fraudulent` (0 for real, 1 for fake)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🧪 Model Training

- Vectorized job descriptions using **TF-IDF** (max 5000 features)
- Addressed class imbalance using **RandomOverSampler**
- Used **RandomForestClassifier** with class weights
- Achieved an F1 Score of **0.9993**
- Saved the model and vectorizer using `joblib`

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Run training manually with:**
#In the terminal
python train_model.py

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**💻 How to Run the App**
bash
Copy
Edit

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Clone the repo
git clone https://github.com/your-username/spot-the-scam.git
cd spot-the-scam

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create and activate virtual environment
python -m venv virtual
virtual\Scripts\activate  # On Windows

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Install dependencies
pip install -r requirements.txt

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Start the Streamlit app
streamlit run app.py
Once running, you can:
Paste or upload job descriptions
Instantly detect if a job is fake or not
View a SHAP summary of important words contributing to the decision

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**⚙️ Tech Used**
•Python
•Streamlit
•Scikit-learn
•SHAP
•Imbalanced-learn
•Numpy
•Pandas, Matplotlib

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Future Scope**
•Classify job roles like internship, marketing, IT, etc.
•Build a retraining option in UI
•Add word clouds and advanced visualizations
•Turn the model into a real-time API for integration


