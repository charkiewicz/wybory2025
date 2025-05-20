import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from collections import Counter # For digit analysis
import json # For saving structured results

# --- 0. Configuration ---
BASE_OUTPUT_DIR = "data/turnout_analysis_results" # From previous script
FURTHER_ANALYSIS_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "further_analysis")
os.makedirs(FURTHER_ANALYSIS_OUTPUT_DIR, exist_ok=True)

# Input file from the previous analysis step
TURNOUT_DATA_FILE = os.path.join(BASE_OUTPUT_DIR, "data_with_turnout.csv")

# Input file with candidate-level votes (needed for digit analysis)
# Replace with your actual path if different
CANDIDATE_VOTES_FILE = 'data/polska_prezydent2025_obkw_kandydaci_NATIONAL_FINAL.csv'

# Thresholds for outlier identification
TURNOUT_UPPER_THRESHOLD = 90.0  # e.g., turnout > 90%
TURNOUT_LOWER_THRESHOLD = 30.0  # e.g., turnout < 30%
SUSPICIOUSLY_HIGH_TURNOUT_FOR_DIGIT_ANALYSIS = 95.0 # e.g. turnout > 95%

# --- 1. Load Data ---
print("--- Loading Data ---")
try:
    df_turnout = pd.read_csv(TURNOUT_DATA_FILE)
    print(f"Successfully loaded turnout data: {TURNOUT_DATA_FILE}")
    print(f"Shape: {df_turnout.shape}")
except FileNotFoundError:
    print(f"ERROR: File not found: {TURNOUT_DATA_FILE}. Please run the previous analysis script first.")
    exit()
except Exception as e:
    print(f"Error loading {TURNOUT_DATA_FILE}: {e}")
    exit()

# For digit analysis, we need the candidate votes data
try:
    df_candidate_votes = pd.read_csv(CANDIDATE_VOTES_FILE)
    print(f"Successfully loaded candidate votes data: {CANDIDATE_VOTES_FILE}")
    print(f"Shape: {df_candidate_votes.shape}")
except FileNotFoundError:
    print(f"WARNING: Candidate votes file not found: {CANDIDATE_VOTES_FILE}. Digit analysis will be skipped.")
    df_candidate_votes = None
except Exception as e:
    print(f"Error loading {CANDIDATE_VOTES_FILE}: {e}")
    df_candidate_votes = None

# --- 2. Outlier Identification ---
print(f"\n--- Outlier Identification (Turnout < {TURNOUT_LOWER_THRESHOLD}% or > {TURNOUT_UPPER_THRESHOLD}%) ---")
df_low_turnout = df_turnout[df_turnout['Turnout_Percentage'] < TURNOUT_LOWER_THRESHOLD]
df_high_turnout = df_turnout[df_turnout['Turnout_Percentage'] > TURNOUT_UPPER_THRESHOLD]

df_outliers = pd.concat([df_low_turnout, df_high_turnout]).sort_values(by='Turnout_Percentage')

print(f"Number of stations with turnout < {TURNOUT_LOWER_THRESHOLD}%: {len(df_low_turnout)}")
print(f"Number of stations with turnout > {TURNOUT_UPPER_THRESHOLD}%: {len(df_high_turnout)}")
print(f"Total outlier stations identified: {len(df_outliers)}")

if not df_outliers.empty:
    outliers_file_path = os.path.join(FURTHER_ANALYSIS_OUTPUT_DIR, "turnout_outliers.csv")
    df_outliers.to_csv(outliers_file_path, index=False)
    print(f"Outlier station data saved to: {outliers_file_path}")
    print("\nSample of outlier stations:")
    print(df_outliers[['URL_ID', 'Wojewodztwo_Name', 'Powiat_MnpP_Name', 'Gmina_Name', 'Turnout_Percentage']].head())
else:
    print("No outliers found based on the defined thresholds.")

# --- 3. Geographic Clustering (Conceptual - Grouping by Admin Units) ---
# Full geographic mapping requires geocoding and libraries like GeoPandas/Folium.
# Here, we'll group outliers by administrative units if columns exist.
print("\n--- Geographic Clustering of Outliers (by Administrative Units) ---")
admin_cols = ['Wojewodztwo_Name', 'Powiat_MnpP_Name', 'Gmina_Name']
if not df_outliers.empty and all(col in df_outliers.columns for col in admin_cols):
    print("Outlier counts per Wojewodztwo:")
    print(df_outliers['Wojewodztwo_Name'].value_counts())

    # You can get more granular:
    # print("\nOutlier counts per Powiat (Top 10):")
    # print(df_outliers['Powiat_MnpP_Name'].value_counts().nlargest(10))

    # List stations in a specific Wojewodztwo with high turnout
    # example_woj = df_outliers['Wojewodztwo_Name'].value_counts().index[0] # Pick the Woj with most outliers
    # print(f"\nHigh turnout stations in {example_woj}:")
    # print(df_high_turnout[df_high_turnout['Wojewodztwo_Name'] == example_woj][['URL_ID', 'Gmina_Name', 'Turnout_Percentage']])
else:
    print("Could not perform administrative unit grouping for outliers (missing columns or no outliers).")
print("For actual map-based geographic clustering, geocoding 'Commission_Address_Raw' and using libraries like GeoPandas/Folium would be necessary.")

# --- 4. Correlation Analysis (Placeholder - Requires Demographic/Other Data) ---
print("\n--- Correlation Analysis (Conceptual) ---")
print("To perform meaningful correlation analysis (e.g., turnout vs. demographics),")
print("you would need an additional dataset with demographic or geographic characteristics")
print("per polling station (URL_ID) or Gmina/Powiat.")
print("Example steps (if you had such data in 'df_demographics' with 'URL_ID'):")
print("# df_merged = pd.merge(df_turnout, df_demographics, on='URL_ID', how='left')")
print("# correlation_matrix = df_merged[['Turnout_Percentage', 'Demographic_Feature1', 'Demographic_Feature2']].corr()")
print("# print(correlation_matrix)")
print("# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')")
print("# plt.title('Correlation Heatmap')")
# heatmap_path = os.path.join(FURTHER_ANALYSIS_OUTPUT_DIR, "correlation_heatmap_example.png")
# print(f"# plt.savefig('{heatmap_path}')")
# print(f"# plt.show()")
print("This section is a placeholder as external demographic data is not loaded.")


# --- 5. Digit Analysis on High-Turnout Stations (Benford's Law - First Significant Digit) ---
print(f"\n--- Digit Analysis for Stations with Turnout > {SUSPICIOUSLY_HIGH_TURNOUT_FOR_DIGIT_ANALYSIS}% ---")

def get_first_digit(n):
    n_abs = abs(n)
    if n_abs == 0:
        return None # Or handle as '0' if that's meaningful in context
    return int(str(n_abs)[0])

def get_second_digit(n):
    s = str(abs(n))
    if len(s) < 2 or '.' in s[1]: # if it's like "3.14", second digit is 1
        return None
    if s[0] == '0' and '.' in s: # e.g. 0.123
        s_after_decimal = s.split('.')[1]
        while s_after_decimal.startswith('0'):
            s_after_decimal = s_after_decimal[1:]
        if len(s_after_decimal) >=2:
            return int(s_after_decimal[1])
        else:
            return None
    return int(s[1])


if df_candidate_votes is not None:
    suspicious_stations_ids = df_turnout[
        df_turnout['Turnout_Percentage'] > SUSPICIOUSLY_HIGH_TURNOUT_FOR_DIGIT_ANALYSIS
    ]['URL_ID'].unique()

    if len(suspicious_stations_ids) > 0:
        print(f"Found {len(suspicious_stations_ids)} stations with turnout > {SUSPICIOUSLY_HIGH_TURNOUT_FOR_DIGIT_ANALYSIS}% for digit analysis.")

        # Filter candidate votes for these suspicious stations
        df_suspicious_votes = df_candidate_votes[df_candidate_votes['URL_ID'].isin(suspicious_stations_ids)]

        if not df_suspicious_votes.empty:
            # Analyze First Significant Digit (FSD) of the 'Votes' column
            # We only consider non-zero votes for Benford's Law
            valid_votes_for_benford = df_suspicious_votes[df_suspicious_votes['Votes'] > 0]['Votes']

            if not valid_votes_for_benford.empty:
                first_digits = valid_votes_for_benford.apply(get_first_digit).dropna()
                digit_counts_fsd = Counter(first_digits)
                total_valid_numbers_fsd = len(first_digits)

                observed_freq_fsd = {digit: count / total_valid_numbers_fsd for digit, count in digit_counts_fsd.items()}
                benford_freq_fsd = {digit: np.log10(1 + 1/digit) for digit in range(1, 10)}

                print("\nFirst Significant Digit (FSD) Analysis (Benford's Law):")
                print(f"Total non-zero vote counts analyzed from suspicious stations: {total_valid_numbers_fsd}")
                print("Digit | Observed Freq. | Benford Freq.")
                print("------|----------------|---------------")
                for d in range(1, 10):
                    obs = observed_freq_fsd.get(d, 0)
                    ben = benford_freq_fsd.get(d, 0)
                    print(f"{d:^6}| {obs:^14.4f} | {ben:^13.4f}")

                # Plotting FSD
                digits = list(range(1, 10))
                observed_plot_fsd = [observed_freq_fsd.get(d, 0) for d in digits]
                benford_plot_fsd = [benford_freq_fsd.get(d, 0) for d in digits]

                plt.figure(figsize=(12, 7))
                bar_width = 0.35
                index = np.arange(len(digits))
                plt.bar(index - bar_width/2, observed_plot_fsd, bar_width, label='Observed FSD', color='skyblue')
                plt.bar(index + bar_width/2, benford_plot_fsd, bar_width, label="Benford's Law FSD", color='salmon', alpha=0.7)
                plt.xlabel('First Significant Digit')
                plt.ylabel('Frequency')
                plt.title(f'First Digit Distribution in Votes from High Turnout Stations (> {SUSPICIOUSLY_HIGH_TURNOUT_FOR_DIGIT_ANALYSIS}%)')
                plt.xticks(index, digits)
                plt.legend()
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                fsd_plot_path = os.path.join(FURTHER_ANALYSIS_OUTPUT_DIR, "benford_fsd_high_turnout_votes.png")
                plt.savefig(fsd_plot_path)
                print(f"Benford FSD plot saved to: {fsd_plot_path}")
                plt.show()

                # Chi-squared test for goodness of fit (FSD)
                observed_counts_fsd_chi = [digit_counts_fsd.get(d, 0) for d in range(1, 10)]
                expected_counts_fsd_chi = [benford_freq_fsd.get(d, 0) * total_valid_numbers_fsd for d in range(1, 10)]
                
                # Filter out expected counts of 0 to avoid division by zero if a digit is truly not expected (though not for Benford 1-9)
                # Also, chi2_contingency requires all expected frequencies to be >= 5 for reliable results.
                # We'll proceed but note this limitation.
                
                if total_valid_numbers_fsd > 0 and sum(expected_counts_fsd_chi) > 0 : # Ensure there are counts to test
                    # For small sample sizes or if expected frequencies are low, the test might not be accurate.
                    # scipy.stats.chisquare needs observed and expected counts.
                    chi2_stat_fsd, p_value_fsd = stats.chisquare(f_obs=observed_counts_fsd_chi, f_exp=expected_counts_fsd_chi)
                    print(f"\nChi-squared test for FSD (Goodness of Fit to Benford's Law):")
                    print(f"Chi2 Statistic: {chi2_stat_fsd:.4f}, P-value: {p_value_fsd:.4f}")
                    alpha = 0.05
                    if p_value_fsd < alpha:
                        print(f"P-value < {alpha}, suggesting observed FSD distribution significantly differs from Benford's Law.")
                    else:
                        print(f"P-value >= {alpha}, cannot reject null hypothesis (observed FSD distribution is consistent with Benford's Law).")
                    print("Note: Chi-squared test reliability depends on sample size and expected frequencies (ideally all expected > 5).")

                # --- Optional: Second Significant Digit (SSD) Analysis ---
                # second_digits = valid_votes_for_benford.apply(get_second_digit).dropna().astype(int) # ensure int for counter
                # if not second_digits.empty:
                #     digit_counts_ssd = Counter(second_digits)
                #     total_valid_numbers_ssd = len(second_digits)
                #     observed_freq_ssd = {digit: count / total_valid_numbers_ssd for digit, count in digit_counts_ssd.items()}
                #     benford_freq_ssd_formula = lambda d: np.sum([np.log10(1 + 1 / (k * 10 + d)) for k in range(1, 10)]) # Benford for SSD
                #     benford_freq_ssd = {digit: benford_freq_ssd_formula(digit) for digit in range(0, 10)}
                #     # ... (plotting and chi-squared for SSD similar to FSD) ...
                #     print("\nSecond Significant Digit analysis would go here...")

            else:
                print("No valid (non-zero) vote counts found in suspicious stations for Benford analysis.")
        else:
            print(f"No candidate vote data found for the {len(suspicious_stations_ids)} suspicious stations.")
    else:
        print(f"No stations found with turnout > {SUSPICIOUSLY_HIGH_TURNOUT_FOR_DIGIT_ANALYSIS}%. Skipping digit analysis.")
else:
    print("Candidate votes data (df_candidate_votes) not loaded. Skipping digit analysis.")

print("\n--- Further Analysis Script Complete ---")
print(f"Outputs saved in: {FURTHER_ANALYSIS_OUTPUT_DIR}")