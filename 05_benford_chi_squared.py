import pandas as pd
import numpy as np
from scipy import stats
import math
import os
from collections import Counter

# --- Configuration ---
CSV_FILE_PATH = 'data/polska_prezydent2025_tura2_obkw_kandydaci_NATIONAL_FINAL.csv'
OUTPUT_RESULTS_FILE = 'data/benford_tests_results.txt' # Renamed for clarity
ALPHA = 0.05 # Significance level
MIN_DATA_POINTS_FOR_BENFORD = 30 # Minimum number of first digits for a somewhat reliable test

# --- Helper Functions ---
def get_first_digit(n):
    """Extracts the first significant digit of a positive number."""
    if pd.isna(n) or n <= 0: # Handle NaN and non-positive
        return None
    n_int = int(n) # Ensure it's an integer for the loop
    while n_int >= 10:
        n_int //= 10
    return n_int

def perform_benford_test_on_series(votes_series, series_label, alpha_level):
    """
    Performs Benford's Law test on a given pandas Series of vote counts.
    Returns a list of strings representing the formatted results.
    """
    test_results_lines = []
    test_results_lines.append(f"\n--- Benford's Law Test for: {series_label} ---\n")

    # Filter for positive votes only for this series
    positive_votes = votes_series[votes_series > 0].dropna()

    if positive_votes.empty:
        test_results_lines.append(f"No positive vote entries found for {series_label}. Skipping Benford's test.\n")
        return test_results_lines

    test_results_lines.append(f"Number of vote entries considered (Votes > 0): {len(positive_votes)}\n")

    first_digits = positive_votes.apply(get_first_digit).dropna().astype(int)

    if len(first_digits) < MIN_DATA_POINTS_FOR_BENFORD:
        test_results_lines.append(
            f"Warning: Insufficient data points ({len(first_digits)}) for a reliable Benford's Law test for {series_label}. "
            f"Minimum recommended: {MIN_DATA_POINTS_FOR_BENFORD}. Results should be treated with extreme caution.\n"
        )
        # We can still proceed to show what the calculation would be, but with a strong warning.
        # Or, uncomment the next line to skip entirely:
        # return test_results_lines

    if len(first_digits) == 0: # If after filtering MIN_DATA_POINTS, nothing is left (or was empty to start)
        test_results_lines.append(f"No valid first digits to analyze for {series_label}.\n")
        return test_results_lines


    observed_counts = Counter(first_digits)
    total_valid_digits = len(first_digits)

    observed_freq_array = np.array([observed_counts.get(d, 0) for d in range(1, 10)])
    benford_expected_props = np.array([math.log10(1 + 1/d) for d in range(1, 10)])
    expected_freq_array = benford_expected_props * total_valid_digits

    test_results_lines.append("Observed vs. Expected Frequencies for First Digits (Benford's Law):")
    test_results_lines.append("Digit | Observed Count | Expected Count (Benford) | Observed Prop. | Expected Prop. (Benford)")
    test_results_lines.append("------|----------------|--------------------------|----------------|--------------------------")
    for i in range(9):
        digit = i + 1
        obs_c = observed_freq_array[i]
        exp_c = expected_freq_array[i]
        obs_p = obs_c / total_valid_digits if total_valid_digits > 0 else 0
        exp_p = benford_expected_props[i]
        test_results_lines.append(f"{digit: <5} | {obs_c: <14} | {exp_c: <24.2f} | {obs_p: <14.4f} | {exp_p: <24.4f}")
    test_results_lines.append("\n")

    # Perform Chi-squared goodness-of-fit test
    warnings_chi2 = []
    if np.any(expected_freq_array == 0):
        warnings_chi2.append("Warning: Some expected frequencies are zero. Chi-squared test might be unreliable or fail.")
    if np.any(expected_freq_array < 1):
         warnings_chi2.append("Warning: Some expected frequencies are less than 1.")
    elif np.any(expected_freq_array < 5):
         warnings_chi2.append("Warning: Some expected frequencies are less than 5. Chi-squared test results should be interpreted with caution.")

    if warnings_chi2:
        test_results_lines.extend(warnings_chi2)
        test_results_lines.append("\n")


    if total_valid_digits > 0 and not np.any(expected_freq_array == 0) and len(np.unique(first_digits)) > 1 : # Avoid test if no data or all expected are zero, or only one digit type
        try:
            chi2_stat, p_value = stats.chisquare(f_obs=observed_freq_array, f_exp=expected_freq_array)
            test_results_lines.append(f"Chi-squared statistic: {chi2_stat:.4f}")
            test_results_lines.append(f"P-value: {p_value:.4g}")
            test_results_lines.append(f"Significance level (alpha): {alpha_level}")

            if p_value < alpha_level:
                test_results_lines.append(f"Result: The p-value ({p_value:.4g}) is less than alpha ({alpha_level}).")
                test_results_lines.append("We reject the null hypothesis. The observed distribution of first digits significantly deviates from Benford's Law for this dataset.\n")
            else:
                test_results_lines.append(f"Result: The p-value ({p_value:.4g}) is not less than alpha ({alpha_level}).")
                test_results_lines.append("We fail to reject the null hypothesis. There is not enough evidence to say the observed distribution of first digits significantly deviates from Benford's Law for this dataset.\n")
        except ValueError as e:
            test_results_lines.append(f"Error during Chi-squared calculation: {e}. This can happen if observed and expected arrays don't align or have issues (e.g. too many zeros).\n")
            test_results_lines.append("Skipping Chi-squared interpretation for this set.\n")

    else:
        test_results_lines.append("Chi-squared test skipped due to insufficient data or problematic expected frequencies.\n")

    return test_results_lines


# --- Main script ---
if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_RESULTS_FILE), exist_ok=True)
    results_output = []

    results_output.append("BENFORD'S LAW TEST RESULTS\n")
    results_output.append("===========================\n")

    try:
        df = pd.read_csv(CSV_FILE_PATH)
        results_output.append(f"Successfully loaded data from '{CSV_FILE_PATH}'.\n")
    except FileNotFoundError:
        results_output.append(f"Error: The file '{CSV_FILE_PATH}' was not found.")
        results_output.append("Please make sure the CSV file is in the correct location.")
        with open(OUTPUT_RESULTS_FILE, 'w', encoding='utf-8') as f:
            f.write("\n".join(results_output))
        print(f"Results saved to {OUTPUT_RESULTS_FILE}")
        exit()

    # Ensure 'Votes' column is numeric
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
    # We don't drop NaNs globally here, as filtering happens within perform_benford_test_on_series

    # 1. Benford's Law Test for ALL CANDIDATES COMBINED
    all_votes_series = df['Votes'].copy() # Use a copy to avoid SettingWithCopyWarning if any modification
    combined_results = perform_benford_test_on_series(all_votes_series, "All Candidates Combined", ALPHA)
    results_output.extend(combined_results)

    # 2. Benford's Law Test for EACH CANDIDATE SEPARATELY
    results_output.append("\n\n--- Benford's Law Tests Per Candidate ---\n")
    unique_candidates = df['Candidate'].unique()

    if len(unique_candidates) == 0:
        results_output.append("No unique candidates found in the 'Candidate' column.\n")
    else:
        results_output.append(f"Found {len(unique_candidates)} unique candidates. Performing tests for each...\n")

    for candidate_name in unique_candidates:
        if pd.isna(candidate_name): # Handle potential NaN candidate names
            candidate_label = "Candidate: Unknown/NaN"
            candidate_votes_series = df[df['Candidate'].isna()]['Votes'].copy()
        else:
            candidate_label = f"Candidate: {candidate_name}"
            candidate_votes_series = df[df['Candidate'] == candidate_name]['Votes'].copy()

        candidate_results = perform_benford_test_on_series(candidate_votes_series, candidate_label, ALPHA)
        results_output.extend(candidate_results)
        results_output.append("-" * 50) # Separator between candidates

    # General explanation of Chi-squared (optional, can be removed if only Benford results are needed)
    results_output.append("\n\n--- Chi-squared Goodness-of-Fit Test (General Concept for Benford's Application) ---\n")
    results_output.append("The Chi-squared goodness-of-fit test, as applied here, determines if the observed frequency")
    results_output.append("distribution of first digits matches the expected frequencies predicted by Benford's Law.")
    results_output.append("\nNull Hypothesis (H0): The observed first-digit frequencies are consistent with Benford's Law.")
    results_output.append("Alternative Hypothesis (Ha): The observed first-digit frequencies are NOT consistent with Benford's Law.")
    results_output.append("\nA small p-value (typically < alpha) leads to rejecting H0, suggesting the data does not follow Benford's Law.\n")


    # Save all results
    try:
        with open(OUTPUT_RESULTS_FILE, 'w', encoding='utf-8') as f:
            f.write("\n".join(results_output))
        print(f"\nAll Benford's test results saved to: {OUTPUT_RESULTS_FILE}")
    except Exception as e:
        print(f"\nError saving results to file: {e}")