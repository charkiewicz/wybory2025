import pandas as pd
import os
import statsmodels.api as sm

# --- 0. Setup: Define file paths ---
data_dir = 'data' 
files = {
    'r1_votes': 'polska_prezydent2025_obkw_kandydaci_NATIONAL_FINAL.csv',
    'r1_summary': 'polska_prezydent2025_obkw_podsumowanie_NATIONAL_FINAL.csv',
    'r2_votes': 'polska_prezydent2025_tura2_obkw_kandydaci_NATIONAL_FINAL.csv',
    'r2_summary': 'polska_prezydent2025_tura2_obkw_podsumowanie_NATIONAL_FINAL.csv'
}

# --- 1. Load all four CSV files into pandas DataFrames ---
try:
    r1_votes_df = pd.read_csv(os.path.join(data_dir, files['r1_votes']))
    r1_summary_df = pd.read_csv(os.path.join(data_dir, files['r1_summary']))
    r2_votes_df = pd.read_csv(os.path.join(data_dir, files['r2_votes']))
    r2_summary_df = pd.read_csv(os.path.join(data_dir, files['r2_summary']))
    print("All four files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("\nPlease ensure your directory structure is correct.")
    exit()

# --- 2. Create the URL_ID mapping (Requirement 1) ---
print("\n--- Step 1: Creating URL_ID mapping ---")

# --- MODIFICATION START ---
# Define columns that MUST have a value for a row to be valid.
# We are REMOVING 'Gmina_Name' from this list as per your instruction.
essential_cols = ['Wojewodztwo_Name', 'Powiat_MnpP_Name', 'Commission_Address_Raw']

# Define columns for sorting. We STILL include 'Gmina_Name' here to ensure
# that nulls are matched with nulls correctly.
sort_cols = ['Wojewodztwo_Name', 'Powiat_MnpP_Name', 'Gmina_Name', 'Commission_Address_Raw']
# --- MODIFICATION END ---


# Clean the dataframes by dropping rows where ESSENTIAL identifiers are null
r1_summary_clean = r1_summary_df.dropna(subset=essential_cols).copy()
r2_summary_clean = r2_summary_df.dropna(subset=essential_cols).copy()

# Sort both dataframes to ensure corresponding rows are aligned
r1_summary_sorted = r1_summary_clean.sort_values(by=sort_cols).reset_index(drop=True)
r2_summary_sorted = r2_summary_clean.sort_values(by=sort_cols).reset_index(drop=True)

# Verify that the number of polling stations matches
if len(r1_summary_sorted) != len(r2_summary_sorted):
    print(f"Warning: The number of polling stations in summary files do not match after cleaning.")
    print(f"Round 1 valid stations: {len(r1_summary_sorted)}")
    print(f"Round 2 valid stations: {len(r2_summary_sorted)}")
    print("Cannot proceed with a reliable mapping.")
    exit()

# Create the mapping from Round 1 URL_ID to Round 2 URL_ID
url_id_mapping = dict(zip(r1_summary_sorted['URL_ID'], r2_summary_sorted['URL_ID']))

print(f"Successfully created a mapping for {len(url_id_mapping)} polling stations.")
print("Sample of URL_ID mapping (R1_URL_ID -> R2_URL_ID):")
for i, (k, v) in enumerate(url_id_mapping.items()):
    if i >= 5: break
    print(f"  {k} -> {v}")


# --- 3. Prepare and combine data for the final DataFrame (Requirement 2 & 3) ---
# (The rest of the script remains the same)
print("\n--- Step 2: Preparing and combining data ---")

r1_summary_subset = r1_summary_sorted[[
    'URL_ID', 'Wojewodztwo_Name', 'Powiat_MnpP_Name', 'Gmina_Name', 'Commission_Address_Raw',
    'liczba_uprawnionych', 'liczba_glosow_niewaznych', 'liczba_glosow_waznych_na_kandydatow'
]].copy()

r2_summary_subset = r2_summary_sorted[[
    'URL_ID', 'liczba_uprawnionych', 'liczba_glosow_niewaznych', 'liczba_glosow_waznych_na_kandydatow'
]].copy()

r2_summary_subset = r2_summary_subset.rename(columns={
    'URL_ID': 'R2_URL_ID',
    'liczba_uprawnionych': 'R2_liczba_uprawnionych',
    'liczba_glosow_niewaznych': 'R2_liczba_glosow_niewaznych',
    'liczba_glosow_waznych_na_kandydatow': 'R2_liczba_glosow_waznych_na_kandydatow'
})

r1_votes_pivot = r1_votes_df.pivot_table(
    index='URL_ID', columns='Candidate', values='Votes', fill_value=0
).add_prefix('R1_VOTES_')

r2_votes_pivot = r2_votes_df.pivot_table(
    index='URL_ID', columns='Candidate', values='Votes', fill_value=0
).add_prefix('R2_VOTES_')

combined_summary = pd.concat([r1_summary_subset, r2_summary_subset], axis=1)

final_df = pd.merge(combined_summary, r1_votes_pivot, on='URL_ID', how='left')

final_df = pd.merge(final_df, r2_votes_pivot, left_on='R2_URL_ID', right_index=True, how='left')

print("Successfully merged all data into a single DataFrame.")


# --- 4. Display the final result ---
print("\n--- Step 3: Final Combined DataFrame ---")
print(f"Shape of the final DataFrame: {final_df.shape}")
print("\nColumns of the final DataFrame:")
print(final_df.columns.tolist())

print("\nHead of the final DataFrame:")
print(final_df.head().T)

# --- 5. Save the final results to a new CSV file ---
output_filename = 'election_results_R1_R2_combined.csv'
final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"\n--- Step 4: Saving Results ---")
print(f"Final combined data successfully saved to '{output_filename}'")

# --- 6. Analyze Turnout ---

# Calculate total votes cast (valid + invalid) for each round
final_df['R1_Total_Votes_Cast'] = final_df['liczba_glosow_waznych_na_kandydatow'] + final_df['liczba_glosow_niewaznych']
final_df['R2_Total_Votes_Cast'] = final_df['R2_liczba_glosow_waznych_na_kandydatow'] + final_df['R2_liczba_glosow_niewaznych']

# Calculate turnout percentage for each round
# We use .fillna(0) in case liczba_uprawnionych is zero to avoid division errors
final_df['Turnout_R1_pct'] = (final_df['R1_Total_Votes_Cast'] / final_df['liczba_uprawnionych']).fillna(0) * 100
final_df['Turnout_R2_pct'] = (final_df['R2_Total_Votes_Cast'] / final_df['R2_liczba_uprawnionych']).fillna(0) * 100

# Calculate the change in turnout in percentage points
final_df['Turnout_Change_pp'] = final_df['Turnout_R2_pct'] - final_df['Turnout_R1_pct']

print("\n--- Step 5: Turnout Analysis ---")
print("Turnout columns calculated. Here's a summary:")
print(final_df[['Turnout_R1_pct', 'Turnout_R2_pct', 'Turnout_Change_pp']].describe())

# Show the 5 polling stations with the biggest INCREASE in turnout
print("\nTop 5 Polling Stations by Turnout INCREASE:")
print(final_df.sort_values(by='Turnout_Change_pp', ascending=False)[['Wojewodztwo_Name', 'Powiat_MnpP_Name', 'Gmina_Name', 'Commission_Address_Raw', 'Turnout_Change_pp']].head())

# Show the 5 polling stations with the biggest DECREASE in turnout
print("\nTop 5 Polling Stations by Turnout DECREASE:")
print(final_df.sort_values(by='Turnout_Change_pp', ascending=True)[['Wojewodztwo_Name', 'Powiat_MnpP_Name', 'Gmina_Name', 'Commission_Address_Raw', 'Turnout_Change_pp']].head())

# --- 7. Prepare for Voter Flow Analysis ---

# Calculate the change in absolute votes for the two R2 candidates
final_df['gain_NAWROCKI'] = final_df['R2_VOTES_NAWROCKI Karol Tadeusz'] - final_df['R1_VOTES_NAWROCKI Karol Tadeusz']
final_df['gain_TRZASKOWSKI'] = final_df['R2_VOTES_TRZASKOWSKI Rafa Kazimierz'] - final_df['R1_VOTES_TRZASKOWSKI Rafa Kazimierz']

# Identify all R1 candidates who were eliminated
r1_candidates = [col for col in final_df.columns if col.startswith('R1_VOTES_')]
eliminated_candidates = [c for c in r1_candidates if 'NAWROCKI' not in c and 'TRZASKOWSKI' not in c]

# Calculate the total pool of votes from eliminated candidates in each polling station
final_df['votes_pool_eliminated'] = final_df[eliminated_candidates].sum(axis=1)

print("\n--- Step 6: Voter Flow Analysis ---")
print("Data prepared for voter flow analysis.")

# --- Option 1: Simple Proportional Model ---
total_gain_nawrocki = final_df['gain_NAWROCKI'].sum()
total_gain_trzaskowski = final_df['gain_TRZASKOWSKI'].sum()
total_gains = total_gain_nawrocki + total_gain_trzaskowski

prop_nawrocki = total_gain_nawrocki / total_gains
prop_trzaskowski = total_gain_trzaskowski / total_gains

print("\n--- Voter Flow Option 1: Simple Proportional Model (National) ---")
print(f"Total 'new' votes acquired in R2: {total_gains:,.0f}")
print(f"  - Share acquired by NAWROCKI: {prop_nawrocki:.2%}")
print(f"  - Share acquired by TRZASKOWSKI: {prop_trzaskowski:.2%}")

# --- Option 2: Correlation Model ---
# Calculate the proportion of the 'available votes' that each R2 candidate captured in each station
# We add a small number (1e-6) to the denominator to avoid division by zero
final_df['gain_share_NAWROCKI'] = final_df['gain_NAWROCKI'] / (final_df['votes_pool_eliminated'] + 1e-6)
final_df['gain_share_TRZASKOWSKI'] = final_df['gain_TRZASKOWSKI'] / (final_df['votes_pool_eliminated'] + 1e-6)

# Calculate R1 vote share for each eliminated candidate
for cand_col in eliminated_candidates:
    share_col_name = cand_col.replace('R1_VOTES_', 'R1_SHARE_')
    final_df[share_col_name] = final_df[cand_col] / (final_df['R1_Total_Votes_Cast'] + 1e-6)

print("\n--- Voter Flow Option 2: Correlation Model ---")
print("Correlation between R1 candidate's vote share and NAWROCKI's vote gain share:")

correlations = {}
for cand_col in eliminated_candidates:
    share_col_name = cand_col.replace('R1_VOTES_', 'R1_SHARE_')
    # We only consider stations where the eliminated candidate had at least one vote
    subset_df = final_df[final_df[share_col_name] > 0]
    corr = subset_df['gain_share_NAWROCKI'].corr(subset_df[share_col_name])
    correlations[share_col_name] = corr

# Print sorted correlations
sorted_correlations = sorted(correlations.items(), key=lambda item: item[1], reverse=True)
for cand, corr_value in sorted_correlations:
    print(f"  - {cand.replace('R1_SHARE_', ''):<30}: {corr_value: .3f}")


# --- Option 3: Multiple Regression Model ---


print("\n--- Voter Flow Option 3: Multiple Regression Model ---")

# Define the variables for the model
# Y = The proportion of the available votes that Nawrocki captured
y = final_df['gain_share_NAWROCKI']

# X = The R1 vote shares of all eliminated candidates
eliminated_share_cols = [c.replace('R1_VOTES_', 'R1_SHARE_') for c in eliminated_candidates]
X = final_df[eliminated_share_cols]
X = sm.add_constant(X) # Adds an intercept to the model

# Filter out nonsensical data points (e.g., gaining more votes than were available)
# and where no votes were available from eliminated candidates
valid_data_filter = (final_df['votes_pool_eliminated'] > 0) & (y >= 0) & (y <= 1.5) # Allow some noise
X_filtered = X[valid_data_filter]
y_filtered = y[valid_data_filter]

# Fit the Ordinary Least Squares (OLS) model
model = sm.OLS(y_filtered, X_filtered).fit()

# Print the summary
print("The 'coef' column estimates the proportion of a candidate's voters who went to NAWROCKI.")
print(model.summary())

import matplotlib.pyplot as plt
import seaborn as sns

# --- 8. Create and Save Turnout Difference Histogram ---

print("\n--- Step 7: Visualizing Turnout Change ---")

# For a more readable histogram, we can filter out the extreme outliers.
# Let's focus on changes between -25 and +25 percentage points.
# This range will likely contain over 99% of the regular polling stations.
turnout_change_filtered = final_df['Turnout_Change_pp'][
    (final_df['Turnout_Change_pp'] > -25) & (final_df['Turnout_Change_pp'] < 25)
]

# Set the style for the plot
sns.set_style("whitegrid")
plt.figure(figsize=(12, 7)) # Create a figure with a nice size

# Create the histogram
ax = sns.histplot(x=turnout_change_filtered, bins=100, kde=True, color='skyblue')

# Calculate the mean of the filtered data to show on the plot
mean_change = turnout_change_filtered.mean()

# Add a vertical line at the mean
plt.axvline(mean_change, color='red', linestyle='--', linewidth=2, 
            label=f'Mean Change: {mean_change:.2f} pp')

# Add titles and labels for clarity
plt.title('Distribution of Turnout Change Between Round 1 and Round 2', fontsize=16)
plt.xlabel('Change in Turnout (Percentage Points)', fontsize=12)
plt.ylabel('Number of Polling Stations', fontsize=12)
plt.legend()

# Save the plot to a file
histogram_filename = 'turnout_change_histogram.png'
plt.savefig(histogram_filename, dpi=300) # dpi=300 for high quality

print(f"Histogram saved as '{histogram_filename}'")

# Display the plot
plt.show()


import numpy as np

# --- 9. Anomaly Detection based on Regression Model ---

print("\n--- Step 8: Anomaly Detection ---")
print("Calculating theoretical R2 results based on R1 voting patterns...")

# 1. Generate predictions for ALL valid data points using the fitted model
# The model was trained on X_filtered, but we can predict for the full set X
# This gives us the predicted *share* of the available vote pool for Nawrocki
final_df['predicted_gain_share_NAWROCKI'] = model.predict(X)

# 2. Convert the predicted share back into a theoretical number of absolute votes
# Predicted gain = predicted_share * available_votes_pool
predicted_gain_nawrocki = final_df['predicted_gain_share_NAWROCKI'] * final_df['votes_pool_eliminated']

# Theoretical R2 votes = R1 votes + predicted gain
final_df['theoretical_R2_VOTES_NAWROCKI'] = final_df['R1_VOTES_NAWROCKI Karol Tadeusz'] + predicted_gain_nawrocki

# Calculate theoretical votes for Trzaskowski based on the remainder of the pool
predicted_gain_trzaskowski = (1 - final_df['predicted_gain_share_NAWROCKI']) * final_df['votes_pool_eliminated']
final_df['theoretical_R2_VOTES_TRZASKOWSKI'] = final_df['R1_VOTES_TRZASKOWSKI Rafa Kazimierz'] + predicted_gain_trzaskowski

# 3. Calculate the Anomaly Score (the residual: Actual - Theoretical)
final_df['anomaly_score_NAWROCKI'] = final_df['R2_VOTES_NAWROCKI Karol Tadeusz'] - final_df['theoretical_R2_VOTES_NAWROCKI']

print("Anomaly scores calculated.")

# --- 4. Visualize the Anomalies ---
print("Generating anomaly visualization plot...")

plt.figure(figsize=(12, 12))
sns.set_style("whitegrid")

# Create the scatter plot of actual vs. theoretical votes
ax = sns.scatterplot(
    x='R2_VOTES_NAWROCKI Karol Tadeusz',
    y='theoretical_R2_VOTES_NAWROCKI',
    data=final_df,
    alpha=0.3, # Use transparency to see density
    s=15      # Smaller points
)

# Draw the 45-degree "Line of No-Surprise" (y=x)
max_val = final_df[['R2_VOTES_NAWROCKI Karol Tadeusz', 'theoretical_R2_VOTES_NAWROCKI']].max().max()
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')

# Find and label the biggest outliers
# We sort by the absolute value of the anomaly score to find the largest deviations in either direction
final_df['abs_anomaly_score'] = abs(final_df['anomaly_score_NAWROCKI'])
top_outliers = final_df.sort_values(by='abs_anomaly_score', ascending=False).head(10)

# Annotate the plot with the top outliers
for i, row in top_outliers.iterrows():
    plt.text(row['R2_VOTES_NAWROCKI Karol Tadeusz'], row['theoretical_R2_VOTES_NAWROCKI'], 
             f"  URL_ID: {row['URL_ID']}", fontsize=9, color='black', ha='left')
    plt.plot(row['R2_VOTES_NAWROCKI Karol Tadeusz'], row['theoretical_R2_VOTES_NAWROCKI'], 'o', ms=8, mec='k', mfc='none')


# Set titles and labels
plt.title('Anomaly Detection: Actual vs. Theoretical R2 Votes for Nawrocki', fontsize=16)
plt.xlabel('Actual R2 Votes for Nawrocki', fontsize=12)
plt.ylabel('Theoretical R2 Votes for Nawrocki (Predicted from R1)', fontsize=12)
plt.legend()
plt.axis('equal') # Ensure the x and y axes have the same scale for a true 45-degree line
plt.xlim(0, max_val)
plt.ylim(0, max_val)

# Save the plot
anomaly_plot_filename = 'anomaly_detection_plot.png'
plt.savefig(anomaly_plot_filename, dpi=300)
print(f"Anomaly plot saved as '{anomaly_plot_filename}'")
plt.show()

# Print the details of the top 5 most suspicious polling stations
print("\nTop 5 Most Anomalous Polling Stations (Highest Deviation from Prediction):")
print(top_outliers[[
    'URL_ID', 'R2_URL_ID', 'Wojewodztwo_Name', 'Powiat_MnpP_Name', 'Gmina_Name', 
    'R2_VOTES_NAWROCKI Karol Tadeusz', 'theoretical_R2_VOTES_NAWROCKI', 'anomaly_score_NAWROCKI'
]].head())



# --- 10. Generate and Save Outlier Report ---

print("\n--- Step 9: Generating Outlier Report ---")

# We already have the 'abs_anomaly_score' column. Let's sort by it.
outlier_report_df = final_df.sort_values(by='abs_anomaly_score', ascending=False)

# Define the columns we want in our final report for easy analysis
report_columns = [
    # Identifier
    'URL_ID',
    'R2_URL_ID',
    'Wojewodztwo_Name',
    'Powiat_MnpP_Name',
    'Gmina_Name',
    'Commission_Address_Raw',
    
    # The Anomaly Score (The "Evidence")
    'anomaly_score_NAWROCKI',
    
    # Actual R2 Results
    'R2_VOTES_NAWROCKI Karol Tadeusz',
    'R2_VOTES_TRZASKOWSKI Rafa Kazimierz',
    'R2_liczba_glosow_waznych_na_kandydatow',

    # Theoretical R2 Results (The "Expectation")
    'theoretical_R2_VOTES_NAWROCKI',
    'theoretical_R2_VOTES_TRZASKOWSKI',

    # R1 Votes for context
    'R1_VOTES_NAWROCKI Karol Tadeusz',
    'R1_VOTES_TRZASKOWSKI Rafa Kazimierz',
    'votes_pool_eliminated'
]

# Create the final report DataFrame
outlier_report_df = outlier_report_df[report_columns]

# Round the theoretical votes to whole numbers for clarity
outlier_report_df['theoretical_R2_VOTES_NAWROCKI'] = outlier_report_df['theoretical_R2_VOTES_NAWROCKI'].round(0)
outlier_report_df['theoretical_R2_VOTES_TRZASKOWSKI'] = outlier_report_df['theoretical_R2_VOTES_TRZASKOWSKI'].round(0)


# Save the top 100 outliers to a CSV file for detailed review
outlier_filename = 'election_anomalies_report.csv'
outlier_report_df.head(100).to_csv(outlier_filename, index=False, encoding='utf-8-sig')

print(f"Top 100 outliers saved to '{outlier_filename}'")

# Display the top 10 most suspicious polling stations in the console
print("\n--- Top 10 Most Anomalous Polling Stations ---")
print(outlier_report_df.head(10).to_string())


