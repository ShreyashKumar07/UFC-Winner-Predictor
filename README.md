# UFC-Winner-Predictor

A machine learning tool to predict UFC fight winners using a Decision Tree Classifier and historical fighter stats. Built with a Jupyter notebook and deployed as an interactive Streamlit app with enhanced features for fighter analysis.

## Overview

This project predicts UFC fight outcomes based on fighter statistics, processed from the UFC Data GitHub repository by jansen88. The dataset was cleaned and modeled in Google Colab (`UFC_Experiment_Final.ipynb`), producing a trained model (`model.pkl`) and fighter data (`fighters_df.csv`). The Streamlit app (`ufc_predictor_new.py`) allows users to select fighters, compare stats, view historical fight data, analyze betting odds, and get predictions with visualizations.

## Files

- `complete_ufc_data.csv`: Original UFC dataset from jansen88/ufc-data.
- `fighters_df.csv`: Processed fighter stats (height, reach, strikes, etc.).
- `df_binary.csv`: Processed fight data with outcomes, methods, and betting odds.
- `model.pkl.zip`: Trained Decision Tree Classifier model (compressed; unzip to get `model.pkl`).
- `UFC_Experiment_Final.ipynb`: Notebook for data processing and model training.
- `ufc_predictor_new.py`: Streamlit app for fighter selection, analysis, and prediction.
- `README.md`

## Credits

- **Dataset**: [jansen88/ufc-data](https://github.com/jansen88/ufc-data) by jansen88.

## Prerequisites

- Python 3.8+
- pip or conda (recommended for dependency management)

## Setup

1. Download The Repository and save it in your pc in a desired folder name

2. Open Anaconda Navigator and paste and run the following codes  
      
3. Create a Virtual Environment (Optional but Recommended)

Using venv:
  python -m venv venv
  source venv/bin/activate  # Windows: venv\Scripts\activate

Or, if using Conda:
  conda create -n ufc_predictor_env python=3.11
  conda activate ufc_predictor_env

4. Install Dependencies

If using pip:
  pip install streamlit pandas numpy scikit-learn plotly

If using conda:
  conda install pandas numpy scikit-learn
  conda install -c conda-forge streamlit
  conda install -c conda-forge plotly

5. Unzip the Model
  Unzip model.pkl.zip to extract model.pkl into the project directory

6. Navigate to your directory
   example: cd C:\Users\your_username\OneDrive\Desktop\Sports Analytics Project\UFC_Predictor

5. Run the App
  streamlit run ufc_predictor_new.py
The app will open in your browser at http://localhost:8501.

Usage

Select Weight Class (Optional): Filter fighters by weight class using the dropdown.
Select Fighters: Use the search bars and dropdowns to select two fighters.
Explore Fighter Data:
View stats comparison (tables).
Analyze historical fight data, including past fights, head-to-head history, and recent form (timelines, pie charts).
Check average historical betting odds and implied probabilities.
Visualize Stats: Expand the "Visual Comparison" section to see bar charts and radar charts comparing fighter stats.
Get Prediction: View the predicted winner and confidence score (if available).
Export Results: Download the stats comparison and differences as a CSV file.
Note: Predictions are based on data as of March 18, 2025.
