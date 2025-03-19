import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Load the datasets and model
fighters_df = pd.read_csv('fighters_df.csv')
df_binary = pd.read_csv('df_binary.csv')
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

# Define key stats for display
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
    diffs = compute_differences(fighter1_data, fighter2_data)
    input_dict.update(diffs)
    for stat in fighters_df.columns.drop('fighter_name'):
        input_dict[f'fighter1_{stat}'] = fighter1_data[stat]
        input_dict[f'fighter2_{stat}'] = fighter2_data[stat]
    return pd.DataFrame([input_dict], columns=FEATURES)

# Function to get historical fights for a fighter
def get_fighter_history(fighter_name):
    fighter_fights = df_binary[(df_binary['fighter1'] == fighter_name) | (df_binary['fighter2'] == fighter_name)]
    history = []
    for _, row in fighter_fights.iterrows():
        opponent = row['fighter2'] if row['fighter1'] == fighter_name else row['fighter1']
        result = 'Win' if (row['outcome'] == 'fighter1' and row['fighter1'] == fighter_name) or (row['outcome'] == 'fighter2' and row['fighter2'] == fighter_name) else 'Loss'
        history.append({
            'Event': row['event_name'],
            'Date': row['event_date'],
            'Opponent': opponent,
            'Result': result,
            'Method': row['method'],
            'Round': row['round']
        })
    return pd.DataFrame(history)

# Function to get head-to-head history
def get_head_to_head(fighter1_name, fighter2_name):
    matches = df_binary[((df_binary['fighter1'] == fighter1_name) & (df_binary['fighter2'] == fighter2_name)) |
                        ((df_binary['fighter1'] == fighter2_name) & (df_binary['fighter2'] == fighter1_name))]
    return matches

# Function to calculate implied probability from odds
def implied_probability(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

# Streamlit app layout
st.title("UFC Fight Predictor MVP")
st.write("Predict UFC fight outcomes with detailed fighter insights, historical data, and visualizations!")

# Weight class filter
weight_classes = ['All'] + sorted(df_binary['weight_class'].unique())
selected_weight_class = st.selectbox("Filter by Weight Class", options=weight_classes)

# Filter fighters based on weight class
if selected_weight_class == 'All':
    available_fighters = fighters_df['fighter_name'].unique()
else:
    weight_class_fights = df_binary[df_binary['weight_class'] == selected_weight_class]
    fighters_in_class = set(weight_class_fights['fighter1']).union(set(weight_class_fights['fighter2']))
    available_fighters = [f for f in fighters_df['fighter_name'].unique() if f in fighters_in_class]

# Fighter selection with search
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    fighter1_search = st.text_input("Search Fighter 1", key="fighter1_search")
    fighter1_options = [f for f in available_fighters if fighter1_search.lower() in f.lower()]
    fighter1_name = st.selectbox("Select Fighter 1", options=fighter1_options, key="fighter1")
with col2:
    fighter2_search = st.text_input("Search Fighter 2", key="fighter2_search")
    fighter2_options = [f for f in available_fighters if fighter2_search.lower() in f.lower()]
    fighter2_name = st.selectbox("Select Fighter 2", options=fighter2_options, key="fighter2")
with col3:
    if st.button("Clear Selection"):
        st.session_state.fighter1 = None
        st.session_state.fighter2 = None
        st.session_state.fighter1_search = ""
        st.session_state.fighter2_search = ""
        st.rerun()

# Ensure two different fighters are selected
if fighter1_name and fighter2_name and fighter1_name == fighter2_name:
    st.error("Please select two different fighters!")
elif fighter1_name and fighter2_name:
    # Get fighter data
    fighter1_data = fighters_df[fighters_df['fighter_name'] == fighter1_name].iloc[0]
    fighter2_data = fighters_df[fighters_df['fighter_name'] == fighter2_name].iloc[0]

    # Fighter Stats Comparison
    with st.expander("Fighter Stats Comparison", expanded=True):
        stats_df = pd.DataFrame({
            'Stat': [STAT_LABELS[stat] for stat in KEY_STATS],
            fighter1_name: [fighter1_data[stat] for stat in KEY_STATS],
            fighter2_name: [fighter2_data[stat] for stat in KEY_STATS]
        })
        st.dataframe(stats_df.set_index('Stat'), use_container_width=True)

    # Stat Differences
    with st.expander("Stat Differences (Fighter 1 - Fighter 2)"):
        diffs = compute_differences(fighter1_data, fighter2_data)
        diffs_df = pd.DataFrame({
            'Stat': [STAT_LABELS[stat] for stat in KEY_STATS],
            'Difference': [diffs[f'{stat}_diff'] for stat in KEY_STATS]
        })
        st.dataframe(diffs_df.set_index('Stat'), use_container_width=True)

    # Historical Fight Data
    with st.expander("Historical Fight Data"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{fighter1_name}'s Fight History")
            history1 = get_fighter_history(fighter1_name)
            if not history1.empty:
                st.dataframe(history1, use_container_width=True)
                # Win/Loss Pie Chart
                win_loss1 = history1['Result'].value_counts()
                fig_pie1 = px.pie(values=win_loss1.values, names=win_loss1.index, title=f"{fighter1_name} Win/Loss Record")
                st.plotly_chart(fig_pie1, use_container_width=True)
            else:
                st.write("No fight history available.")
        with col2:
            st.subheader(f"{fighter2_name}'s Fight History")
            history2 = get_fighter_history(fighter2_name)
            if not history2.empty:
                st.dataframe(history2, use_container_width=True)
                # Win/Loss Pie Chart
                win_loss2 = history2['Result'].value_counts()
                fig_pie2 = px.pie(values=win_loss2.values, names=win_loss2.index, title=f"{fighter2_name} Win/Loss Record")
                st.plotly_chart(fig_pie2, use_container_width=True)
            else:
                st.write("No fight history available.")

        # Head-to-Head History
        st.subheader("Head-to-Head History")
        head_to_head = get_head_to_head(fighter1_name, fighter2_name)
        if not head_to_head.empty:
            st.dataframe(head_to_head[['event_name', 'event_date', 'outcome', 'method', 'round']], use_container_width=True)
        else:
            st.write("These fighters have not fought each other before.")

        # Recent Form Timeline
        st.subheader("Recent Form Timeline")
        col1, col2 = st.columns(2)
        with col1:
            if not history1.empty:
                history1['Date'] = pd.to_datetime(history1['Date'])
                history1 = history1.sort_values('Date')
                fig_timeline1 = px.scatter(history1, x='Date', y='Result', color='Result', title=f"{fighter1_name} Recent Form",
                                           hover_data=['Event', 'Opponent', 'Method'])
                st.plotly_chart(fig_timeline1, use_container_width=True)
        with col2:
            if not history2.empty:
                history2['Date'] = pd.to_datetime(history2['Date'])
                history2 = history2.sort_values('Date')
                fig_timeline2 = px.scatter(history2, x='Date', y='Result', color='Result', title=f"{fighter2_name} Recent Form",
                                           hover_data=['Event', 'Opponent', 'Method'])
                st.plotly_chart(fig_timeline2, use_container_width=True)

    # Betting Odds Context
    with st.expander("Historical Betting Odds Context"):
        st.subheader("Average Betting Odds from Past Fights")
        col1, col2 = st.columns(2)
        with col1:
            odds1 = df_binary[(df_binary['fighter1'] == fighter1_name) | (df_binary['fighter2'] == fighter1_name)][['favourite', 'underdog', 'favourite_odds', 'underdog_odds']]
            odds1 = odds1.replace([np.inf, -np.inf], np.nan).dropna()
            if not odds1.empty:
                avg_fav_odds1 = odds1[odds1['favourite'] == fighter1_name]['favourite_odds'].mean()
                avg_und_odds1 = odds1[odds1['underdog'] == fighter1_name]['underdog_odds'].mean()
                st.write(f"{fighter1_name} Average Favourite Odds: {avg_fav_odds1:.2f}" if not np.isnan(avg_fav_odds1) else f"{fighter1_name} never a favourite.")
                st.write(f"{fighter1_name} Average Underdog Odds: {avg_und_odds1:.2f}" if not np.isnan(avg_und_odds1) else f"{fighter1_name} never an underdog.")
            else:
                st.write(f"No betting odds data for {fighter1_name}.")
        with col2:
            odds2 = df_binary[(df_binary['fighter1'] == fighter2_name) | (df_binary['fighter2'] == fighter2_name)][['favourite', 'underdog', 'favourite_odds', 'underdog_odds']]
            odds2 = odds2.replace([np.inf, -np.inf], np.nan).dropna()
            if not odds2.empty:
                avg_fav_odds2 = odds2[odds2['favourite'] == fighter2_name]['favourite_odds'].mean()
                avg_und_odds2 = odds2[odds2['underdog'] == fighter2_name]['underdog_odds'].mean()
                st.write(f"{fighter2_name} Average Favourite Odds: {avg_fav_odds2:.2f}" if not np.isnan(avg_fav_odds2) else f"{fighter2_name} never a favourite.")
                st.write(f"{fighter2_name} Average Underdog Odds: {avg_und_odds2:.2f}" if not np.isnan(avg_und_odds2) else f"{fighter2_name} never an underdog.")
            else:
                st.write(f"No betting odds data for {fighter2_name}.")

    # Visual Comparison
    with st.expander("Visual Comparison"):
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
        max_values = fighters_df[radar_stats].max()
        fighter1_values = [fighter1_data[stat] / max_values[stat] for stat in radar_stats]
        fighter2_values = [fighter2_data[stat] / max_values[stat] for stat in radar_stats]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=fighter1_values + [fighter1_values[0]],
            theta=[STAT_LABELS[stat] for stat in radar_stats] + [STAT_LABELS[radar_stats[0]]],
            fill='toself',
            name=fighter1_name
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=fighter2_values + [fighter2_values[0]],
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
    with st.spinner("Generating Prediction..."):
        input_df = prepare_model_input(fighter1_data, fighter2_data)
        prediction = model.predict(input_df)[0]
        winner = fighter1_name if prediction == 1 else fighter2_name
        confidence = max(model.predict_proba(input_df)[0]) * 100 if hasattr(model, 'predict_proba') else None

    st.subheader("Prediction")
    st.write(f"**Winner**: {winner}")
    if confidence:
        st.write(f"**Confidence**: {confidence:.2f}%")
        # Compare with implied probabilities from odds
        if not odds1.empty and not odds2.empty:
            avg_odds1 = avg_fav_odds1 if not np.isnan(avg_fav_odds1) else avg_und_odds1
            avg_odds2 = avg_fav_odds2 if not np.isnan(avg_fav_odds2) else avg_und_odds2
            if not np.isnan(avg_odds1) and not np.isnan(avg_odds2):
                prob1 = implied_probability(avg_odds1) * 100
                prob2 = implied_probability(avg_odds2) * 100
                st.write(f"**Implied Probability from Historical Odds**: {fighter1_name}: {prob1:.2f}%, {fighter2_name}: {prob2:.2f}%")

    st.write(f"Note: Prediction is based on historical stats as of March 18, 2025, using a trained model.")

    # Export Options (CSV only)
    with st.expander("Export Results"):
        export_df = stats_df.copy()
        export_df['Difference'] = diffs_df['Difference'].values
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download Stats as CSV",
            data=csv,
            file_name=f"{fighter1_name}_vs_{fighter2_name}_stats.csv",
            mime="text/csv"
        )

else:
    st.info("Please select two fighters to generate a prediction.")