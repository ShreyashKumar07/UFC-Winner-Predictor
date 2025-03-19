# UFC-Winner-Predictor

A machine learning tool to predict UFC fight winners using a Random Forest Classifier and historical fighter stats. Built with a Jupyter notebook and deployed as an interactive Streamlit app.

Overview
This project predicts UFC fight outcomes based on fighter statistics, processed from the UFC Data GitHub repository by jansen88. The dataset was cleaned and modeled in Google Colab (UFC_Experiment_Final.ipynb), producing a trained model (model.pkl) and fighter data (fighters_df.csv). The Streamlit app (ufc_predictor.py) lets users select fighters, compare stats, and get predictions.

Files
complete_ufc_data.csv: Original UFC dataset from jansen88/ufc-data.
fighters_df.csv: Processed fighter stats (height, reach, strikes, etc.).
model.pkl: Trained Random Forest Classifier for predictions.
UFC_Experiment_Final.ipynb: Notebook for data processing and model training.
ufc_predictor.py: Streamlit app for fighter selection and prediction.
README.md: This file.
Credits
Dataset: jansen88/ufc-data by jansen88.

Prerequisites
Python 3.8+
pip

Setup
1. Clone Repository
Copy:
git clone https://github.com/[Your-GitHub-Username]/UFC-Fight-Forecast.git
cd UFC-Fight-Forecast

2. Virtual Environment (Optional)
Copy:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3. Install Dependencies
Copy:
pip install streamlit pandas numpy pickle5 plotly scikit-learn

4. Run the App
Copy:
streamlit run ufc_predictor.py
(Opens in browser at http://localhost:8501)

Usage
Select two fighters via dropdowns.
View stats comparison (tables, bar/radar charts).
Get predicted winner and confidence score (if available).
Note: Based on data as of March 18, 2025.

Details
Data: Processed in UFC_Experiment_Final.ipynb with feature engineering (e.g., stat differences).
Model: Random Forest Classifier predicts outcomes using fighter stats.
App: Streamlit UI with Plotly visualizations.

Limitations
Predictions reflect historical data up to March 18, 2025.
Excludes real-time factors (e.g., injuries).
