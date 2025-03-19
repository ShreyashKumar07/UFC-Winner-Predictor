import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Load the datasets and model
fighters_df = pd.read_csv('fighters_df.csv')
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the features expected by the model
FEATURES = [
    'height_diff', 'reach_diff', 'sig_strikes_landed_pm_diff', 'sig_strikes_accuracy_diff',
    'sig_strikes_absorbed_pm_diff', 'sig_strikes_defended_diff', 'takedown_avg_per15m_diff',
    'takedown_accuracy_diff', 'takedown_defence_diff', 'submission_avg_attempted_per15m_diff',
    'curr_weight_diff', 'fighter1_height', 'fighter1_curr_weight', 'fighter1_reach',
    'fighter1_sig_strikes_landed_pm', 'fighter1_sig_strikes_accuracy', 'fighter1_sig_strikes_absorbed_pm',
    'fighter1_sig_strikes_defended', 'fighter1_takedown_avg_per15m', 'fighter1_takedown_accuracy',
    'fighter1_takedown_defence', 'fighter1_submission_avg_attempted_per15m', 'fighter1_age',
    'fighter2_height', 'fighter2_curr_weight', 'fighter2_reach', 'fighter2_sig_strikes_landed_pm',
    'fighter2_sig_strikes_accuracy', 'fighter2_sig_strikes_absorbed_pm', 'fighter2_sig_strikes_defended',
    'fighter2_takedown_avg_per15m', 'fighter2_takedown_accuracy', 'fighter2_takedown_defence',
    'fighter2_submission_avg_attempted_per15m', 'fighter2_age'
]

# Define key stats for display (excluding fighter_name)
KEY_STATS = [
    'height', 'reach', 'sig_strikes_landed_pm', 'sig_strikes_accuracy',
    'sig_strikes_absorbed_pm', 'sig_strikes_defended', 'takedown_avg_per15m',
    'takedown_accuracy', 'takedown_defence', 'submission_avg_attempted_per15m',
    'curr_weight', 'age'
]

# Readable labels for stats
STAT_LABELS = {
    'height': 'Height (cm)',
    'reach': 'Reach (cm)',
    'sig_strikes_landed_pm': 'Sig. Strikes Landed/Min',
    'sig_strikes_accuracy': 'Sig. Strikes Accuracy (%)',
    'sig_strikes_absorbed_pm': 'Sig. Strikes Absorbed/Min',
    'sig_strikes_defended': 'Sig. Strikes Defended (%)',
    'takedown_avg_per15m': 'Takedowns/15min',
    'takedown_accuracy': 'Takedown Accuracy (%)',
    'takedown_defence': 'Takedown Defence (%)',
    'submission_avg_attempted_per15m': 'Submissions Attempted/15min',
    'curr_weight': 'Current Weight (kg)',
    'age': 'Age (years)'
}

# Function to compute differences between two fighters
def compute_differences(fighter1_data, fighter2_data):
    diffs = {}
    for stat in KEY_STATS:
        diffs[f'{stat}_diff'] = fighter1_data[stat] - fighter2_data[stat]
    return diffs

# Function to prepare input for the model
def prepare_model_input(fighter1_data, fighter2_data):
    input_dict = {}
    # Add difference features
    diffs = compute_differences(fighter1_data, fighter2_data)
    input_dict.update(diffs)
    # Add Fighter 1 stats
    for stat in fighters_df.columns.drop('fighter_name'):
        input_dict[f'fighter1_{stat}'] = fighter1_data[stat]
    # Add Fighter 2 stats
    for stat in fighters_df.columns.drop('fighter_name'):
        input_dict[f'fighter2_{stat}'] = fighter2_data[stat]
    return pd.DataFrame([input_dict], columns=FEATURES)

# Streamlit app layout
st.title("UFC Fight Predictor")
st.write("Select two fighters to predict the winner based on their stats!")

# Fighter selection
col1, col2 = st.columns(2)

with col1:
    fighter1_name = st.selectbox("Select Fighter 1", options=fighters_df['fighter_name'].unique(), key="fighter1")
with col2:
    fighter2_name = st.selectbox("Select Fighter 2", options=fighters_df['fighter_name'].unique(), key="fighter2")

# Ensure two different fighters are selected
if fighter1_name == fighter2_name:
    st.error("Please select two different fighters!")
else:
    # Get fighter data
    fighter1_data = fighters_df[fighters_df['fighter_name'] == fighter1_name].iloc[0]
    fighter2_data = fighters_df[fighters_df['fighter_name'] == fighter2_name].iloc[0]

    # Fighter Stats Comparison
    st.subheader("Fighter Stats Comparison")
    stats_df = pd.DataFrame({
        'Stat': [STAT_LABELS[stat] for stat in KEY_STATS],
        fighter1_name: [fighter1_data[stat] for stat in KEY_STATS],
        fighter2_name: [fighter2_data[stat] for stat in KEY_STATS]
    })
    st.dataframe(stats_df.set_index('Stat'), use_container_width=True)

    # Stat Differences
    st.subheader("Stat Differences (Fighter 1 - Fighter 2)")
    diffs = compute_differences(fighter1_data, fighter2_data)
    diffs_df = pd.DataFrame({
        'Stat': [STAT_LABELS[stat] for stat in KEY_STATS],
        'Difference': [diffs[f'{stat}_diff'] for stat in KEY_STATS]
    })
    st.dataframe(diffs_df.set_index('Stat'), use_container_width=True)

    # Visualizations
    st.subheader("Visual Comparison")

    # Bar Chart for Key Stats
    bar_stats = ['height', 'reach', 'sig_strikes_landed_pm', 'takedown_avg_per15m']
    bar_df = pd.DataFrame({
        'Stat': [STAT_LABELS[stat] for stat in bar_stats for _ in range(2)],
        'Value': [fighter1_data[stat] for stat in bar_stats] + [fighter2_data[stat] for stat in bar_stats],
        'Fighter': [fighter1_name] * len(bar_stats) + [fighter2_name] * len(bar_stats)
    })
    fig_bar = px.bar(bar_df, x='Stat', y='Value', color='Fighter', barmode='group', title="Key Stats Comparison")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Radar Chart for Overall Profile
    radar_stats = [
        'sig_strikes_landed_pm', 'sig_strikes_accuracy', 'sig_strikes_defended',
        'takedown_avg_per15m', 'takedown_accuracy', 'takedown_defence'
    ]
    # Normalize stats to 0-1 for radar chart
    max_values = fighters_df[radar_stats].max()
    fighter1_values = [fighter1_data[stat] / max_values[stat] for stat in radar_stats]
    fighter2_values = [fighter2_data[stat] / max_values[stat] for stat in radar_stats]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=fighter1_values + [fighter1_values[0]],  # Close the loop
        theta=[STAT_LABELS[stat] for stat in radar_stats] + [STAT_LABELS[radar_stats[0]]],
        fill='toself',
        name=fighter1_name
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=fighter2_values + [fighter2_values[0]],  # Close the loop
        theta=[STAT_LABELS[stat] for stat in radar_stats] + [STAT_LABELS[radar_stats[0]]],
        fill='toself',
        name=fighter2_name
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Fighter Profile (Normalized Stats)"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Make prediction
    input_df = prepare_model_input(fighter1_data, fighter2_data)
    prediction = model.predict(input_df)[0]
    winner = fighter1_name if prediction == 1 else fighter2_name

    # Get prediction probability
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(input_df)[0]
        confidence = max(probas) * 100
        st.subheader("Prediction")
        st.write(f"**Winner**: {winner}")
        st.write(f"**Confidence**: {confidence:.2f}%")
    else:
        st.subheader("Prediction")
        st.write(f"**Winner**: {winner}")
        st.write("Confidence score not available with this model.")

    st.write(f"Note: Prediction is based on historical stats as of March 18, 2025, using a trained model.")