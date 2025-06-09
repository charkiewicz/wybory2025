import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration (same as before) ---
FILE_R1_KANDYDACI = "data/polska_prezydent2025_obkw_kandydaci_NATIONAL_FINAL.csv"
FILE_R1_PODSUMOWANIE = "data/polska_prezydent2025_obkw_podsumowanie_NATIONAL_FINAL.csv"
FILE_R2_KANDYDACI = "data/polska_prezydent2025_tura2_obkw_kandydaci_NATIONAL_FINAL.csv"
FILE_R2_PODSUMOWANIE = "data/polska_prezydent2025_tura2_obkw_podsumowanie_NATIONAL_FINAL.csv"

CANDIDATE_NAWROCKI = "NAWROCKI Karol Tadeusz"
CANDIDATE_TRZASKOWSKI = "TRZASKOWSKI Rafa Kazimierz"

TRANSFER_RULES = {
    "MENTZEN Sawomir Jerzy": {CANDIDATE_NAWROCKI: 0.88, CANDIDATE_TRZASKOWSKI: 0.12},
    "BRAUN Grzegorz Micha": {CANDIDATE_NAWROCKI: 0.925, CANDIDATE_TRZASKOWSKI: 0.075},
    "HOOWNIA Szymon Franciszek": {CANDIDATE_TRZASKOWSKI: 0.862, CANDIDATE_NAWROCKI: 0.138},
    "BIEJAT Magdalena Agnieszka": {CANDIDATE_TRZASKOWSKI: 0.902, CANDIDATE_NAWROCKI: 0.098},
    "ZANDBERG Adrian Tadeusz": {CANDIDATE_TRZASKOWSKI: 0.838, CANDIDATE_NAWROCKI: 0.162}
}
SPECIFIED_TRANSFER_CANDIDATES_R1 = list(TRANSFER_RULES.keys())
R2_FINALISTS_FROM_R1 = [CANDIDATE_NAWROCKI, CANDIDATE_TRZASKOWSKI]
NEW_VOTER_DISTRIBUTION = {
    CANDIDATE_TRZASKOWSKI: 0.60,
    CANDIDATE_NAWROCKI: 0.40
}
GEO_COLS_FOR_AGG = ['Wojewodztwo_Name', 'Powiat_MnpP_Name', 'Gmina_Name'] # Key for aggregation
R1_SUMMARY_VOTES_COL = 'liczba_glosow_waznych_na_kandydatow'
R2_SUMMARY_VOTES_COL = 'liczba_glosow_waznych_na_kandydatow'

# --- Helper Functions ---

def load_and_prepare_data():
    # ... (same as before)
    try:
        r1_kandydaci_df = pd.read_csv(FILE_R1_KANDYDACI)
        r1_podsumowanie_df = pd.read_csv(FILE_R1_PODSUMOWANIE)
        r2_kandydaci_df = pd.read_csv(FILE_R2_KANDYDACI)
        r2_podsumowanie_df = pd.read_csv(FILE_R2_PODSUMOWANIE)
    except FileNotFoundError as e:
        print(f"Error: File not found. Please check file paths. Missing file: {e.filename}")
        return None, None, None, None
    if r1_kandydaci_df.empty or r1_podsumowanie_df.empty or r2_kandydaci_df.empty or r2_podsumowanie_df.empty:
        print("Error: One or more data files are empty.")
        return None, None, None, None
    return r1_kandydaci_df, r1_podsumowanie_df, r2_kandydaci_df, r2_podsumowanie_df


def process_r1_votes_for_expected(r1_kandydaci_df, r1_podsumowanie_df):
    # This function calculates expected votes at URL_ID level first
    # It will be aggregated later in main
    # GEO_COLS includes URL_ID for this stage
    temp_geo_cols = GEO_COLS_FOR_AGG + ['URL_ID']
    r1_votes_agg = r1_kandydaci_df.groupby(['URL_ID', 'Candidate'])['Votes'].sum().unstack(fill_value=0)
    all_r1_candidate_names = r1_kandydaci_df['Candidate'].unique().tolist()
    for cand_name in all_r1_candidate_names:
        if cand_name not in r1_votes_agg.columns:
            r1_votes_agg[cand_name] = 0
    r1_votes_agg = r1_votes_agg.reset_index()

    # Ensure all geo columns are present in r1_podsumowanie_df
    r1_summary_geo_cols = [col for col in temp_geo_cols if col in r1_podsumowanie_df.columns]
    r1_summary_selected = r1_podsumowanie_df[r1_summary_geo_cols + [R1_SUMMARY_VOTES_COL]].copy()
    r1_summary_selected.rename(columns={R1_SUMMARY_VOTES_COL: 'R1_Total_Valid_Votes_Station'}, inplace=True)
    
    r1_processed_df = pd.merge(r1_summary_selected, r1_votes_agg, on='URL_ID', how='left')
    candidate_vote_cols = [col for col in all_r1_candidate_names] # all_r1_candidate_names already from unstack
    r1_processed_df[candidate_vote_cols] = r1_processed_df[candidate_vote_cols].fillna(0)
    r1_processed_df['R1_Total_Valid_Votes_Station'] = r1_processed_df['R1_Total_Valid_Votes_Station'].fillna(0)
    
    # Calculate expected votes at station level
    df_calc = r1_processed_df.copy()
    df_calc['Expected_Nawrocki_R2'] = 0.0
    df_calc['Expected_Trzaskowski_R2'] = 0.0
    if CANDIDATE_NAWROCKI in df_calc.columns:
         df_calc['Expected_Nawrocki_R2'] += df_calc[CANDIDATE_NAWROCKI]
    if CANDIDATE_TRZASKOWSKI in df_calc.columns:
         df_calc['Expected_Trzaskowski_R2'] += df_calc[CANDIDATE_TRZASKOWSKI]
    for r1_cand, transfers in TRANSFER_RULES.items():
        if r1_cand in df_calc.columns:
            df_calc['Expected_Nawrocki_R2'] += df_calc[r1_cand] * transfers.get(CANDIDATE_NAWROCKI, 0.0)
            df_calc['Expected_Trzaskowski_R2'] += df_calc[r1_cand] * transfers.get(CANDIDATE_TRZASKOWSKI, 0.0)
    other_r1_candidates = [
        cand for cand in all_r1_candidate_names
        if cand not in R2_FINALISTS_FROM_R1 and cand not in SPECIFIED_TRANSFER_CANDIDATES_R1
    ]
    df_calc['Other_R1_Votes_Sum'] = 0.0 # ensure float
    for cand in other_r1_candidates:
        if cand in df_calc.columns:
            df_calc['Other_R1_Votes_Sum'] += df_calc[cand]
    df_calc['Expected_Nawrocki_R2'] += df_calc['Other_R1_Votes_Sum'] * 0.50
    df_calc['Expected_Trzaskowski_R2'] += df_calc['Other_R1_Votes_Sum'] * 0.50
    
    return df_calc # Returns station-level expected votes, R1 totals, and geo_cols

def calculate_new_voters_at_gmina(r1_podsumowanie_df, r2_podsumowanie_df):
    """Calculates new voters by comparing R1 and R2 total valid votes at Gmina level."""
    r1_gmina_totals = r1_podsumowanie_df.groupby(GEO_COLS_FOR_AGG)[R1_SUMMARY_VOTES_COL].sum().reset_index()
    r1_gmina_totals.rename(columns={R1_SUMMARY_VOTES_COL: 'R1_Total_Valid_Votes_Gmina'}, inplace=True)

    r2_gmina_totals = r2_podsumowanie_df.groupby(GEO_COLS_FOR_AGG)[R2_SUMMARY_VOTES_COL].sum().reset_index()
    r2_gmina_totals.rename(columns={R2_SUMMARY_VOTES_COL: 'R2_Total_Valid_Votes_Gmina'}, inplace=True)

    gmina_summary = pd.merge(r1_gmina_totals, r2_gmina_totals, on=GEO_COLS_FOR_AGG, how='outer')
    gmina_summary['R1_Total_Valid_Votes_Gmina'] = gmina_summary['R1_Total_Valid_Votes_Gmina'].fillna(0)
    gmina_summary['R2_Total_Valid_Votes_Gmina'] = gmina_summary['R2_Total_Valid_Votes_Gmina'].fillna(0)

    net_new_voters_gmina = gmina_summary['R2_Total_Valid_Votes_Gmina'] - gmina_summary['R1_Total_Valid_Votes_Gmina']
    gmina_summary['Actual_New_Voters_Gmina'] = net_new_voters_gmina.clip(lower=0)
    
    return gmina_summary[GEO_COLS_FOR_AGG + ['R1_Total_Valid_Votes_Gmina', 'R2_Total_Valid_Votes_Gmina', 'Actual_New_Voters_Gmina']]


def aggregate_expected_votes_to_gmina(expected_votes_station_df, new_voters_gmina_df):
    """Aggregates station-level expected votes to Gmina and adds new voter share."""
    gmina_expected = expected_votes_station_df.groupby(GEO_COLS_FOR_AGG)[
        ['Expected_Nawrocki_R2', 'Expected_Trzaskowski_R2']
    ].sum().reset_index()

    # Merge with Gmina-level new voter data
    gmina_expected = pd.merge(gmina_expected, new_voters_gmina_df, on=GEO_COLS_FOR_AGG, how='left')
    gmina_expected['Actual_New_Voters_Gmina'] = gmina_expected['Actual_New_Voters_Gmina'].fillna(0)
    
    # Add new voter contributions
    gmina_expected['Expected_Nawrocki_R2'] += gmina_expected['Actual_New_Voters_Gmina'] * NEW_VOTER_DISTRIBUTION[CANDIDATE_NAWROCKI]
    gmina_expected['Expected_Trzaskowski_R2'] += gmina_expected['Actual_New_Voters_Gmina'] * NEW_VOTER_DISTRIBUTION[CANDIDATE_TRZASKOWSKI]
    
    return gmina_expected


def process_r2_actual_votes_at_gmina(r2_kandydaci_df):
    """Aggregates actual R2 votes at Gmina level."""
    if r2_kandydaci_df is None or r2_kandydaci_df.empty or 'Candidate' not in r2_kandydaci_df.columns:
        return pd.DataFrame(columns=GEO_COLS_FOR_AGG + [f'Actual_{CANDIDATE_NAWROCKI}_R2', f'Actual_{CANDIDATE_TRZASKOWSKI}_R2'])

    r2_finalists_votes = r2_kandydaci_df[r2_kandydaci_df['Candidate'].isin(R2_FINALISTS_FROM_R1)]
    if r2_finalists_votes.empty:
        return pd.DataFrame(columns=GEO_COLS_FOR_AGG + [f'Actual_{CANDIDATE_NAWROCKI}_R2', f'Actual_{CANDIDATE_TRZASKOWSKI}_R2'])

    # Aggregate at Gmina level directly
    # Ensure GEO_COLS_FOR_AGG are present in r2_finalists_votes
    geo_cols_present = [col for col in GEO_COLS_FOR_AGG if col in r2_finalists_votes.columns]
    if not geo_cols_present:
        print("Error in process_r2_actual_votes_at_gmina: Missing geo columns for aggregation.")
        return pd.DataFrame(columns=GEO_COLS_FOR_AGG + [f'Actual_{CANDIDATE_NAWROCKI}_R2', f'Actual_{CANDIDATE_TRZASKOWSKI}_R2'])

    r2_gmina_actual = r2_finalists_votes.groupby(geo_cols_present + ['Candidate'])['Votes'].sum().unstack(fill_value=0)
    
    r2_gmina_actual.rename(columns={
        CANDIDATE_NAWROCKI: f'Actual_{CANDIDATE_NAWROCKI}_R2',
        CANDIDATE_TRZASKOWSKI: f'Actual_{CANDIDATE_TRZASKOWSKI}_R2'
    }, inplace=True)
    
    if f'Actual_{CANDIDATE_NAWROCKI}_R2' not in r2_gmina_actual.columns:
        r2_gmina_actual[f'Actual_{CANDIDATE_NAWROCKI}_R2'] = 0.0
    if f'Actual_{CANDIDATE_TRZASKOWSKI}_R2' not in r2_gmina_actual.columns:
        r2_gmina_actual[f'Actual_{CANDIDATE_TRZASKOWSKI}_R2'] = 0.0
        
    return r2_gmina_actual.reset_index()


def plot_and_save_results(df, filename="election_comparison_R2_GMINA_totals.png"):
    # ... (Plotting function remains largely the same, just sums the Gmina-level data)
    if df.empty:
        print("Cannot plot results, DataFrame is empty.")
        return
    print("\n--- Plotting Results (Gmina Aggregated) ---")
    actual_nawrocki_col = f'Actual_{CANDIDATE_NAWROCKI}_R2'
    actual_trzaskowski_col = f'Actual_{CANDIDATE_TRZASKOWSKI}_R2'
    if actual_nawrocki_col not in df.columns: df[actual_nawrocki_col] = 0.0
    if actual_trzaskowski_col not in df.columns: df[actual_trzaskowski_col] = 0.0

    total_expected_nawrocki = df['Expected_Nawrocki_R2'].sum()
    total_actual_nawrocki = df[actual_nawrocki_col].sum()
    total_expected_trzaskowski = df['Expected_Trzaskowski_R2'].sum()
    total_actual_trzaskowski = df[actual_trzaskowski_col].sum()

    print(f"Total Expected Nawrocki: {total_expected_nawrocki:,.0f}")
    print(f"Total Actual Nawrocki: {total_actual_nawrocki:,.0f}")
    print(f"Total Expected Trzaskowski: {total_expected_trzaskowski:,.0f}")
    print(f"Total Actual Trzaskowski: {total_actual_trzaskowski:,.0f}")

    # ... (rest of plotting is the same)
    candidates_plot = [CANDIDATE_NAWROCKI, CANDIDATE_TRZASKOWSKI]
    expected_votes_plot = [total_expected_nawrocki, total_expected_trzaskowski]
    actual_votes_plot = [total_actual_nawrocki, total_actual_trzaskowski]
    plot_data = pd.DataFrame({
        'Candidate': candidates_plot + candidates_plot,
        'Vote_Type': ['Expected'] * len(candidates_plot) + ['Actual'] * len(candidates_plot),
        'Total_Votes': expected_votes_plot + actual_votes_plot
    })
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    bar_plot = sns.barplot(x='Candidate', y='Total_Votes', hue='Vote_Type', data=plot_data, palette={"Expected": "skyblue", "Actual": "salmon"})
    for p in bar_plot.patches:
        bar_plot.annotate(format(p.get_height(), ',.0f'), 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'center', 
                           xytext = (0, 9), 
                           textcoords = 'offset points')
    plt.title('National Total Votes (from Gmina Aggregation): Expected vs. Actual (Round 2)', fontsize=14)
    plt.ylabel('Total Votes', fontsize=12)
    plt.xlabel('Candidate', fontsize=12)
    plt.xticks(rotation=10, ha='right', fontsize=10); plt.yticks(fontsize=10)
    plt.legend(title='Vote Type'); plt.tight_layout()
    try:
        plt.savefig(filename); print(f"Plot saved to {filename}")
    except Exception as e: print(f"Error saving plot: {e}")


# --- Main Execution ---
def main():
    print("Starting election analysis (Gmina Level Comparison)...")
    loaded_data = load_and_prepare_data()
    if loaded_data is None or any(df is None for df in loaded_data):
        print("Critical Error: Data loading failed. Exiting.")
        return
    r1_kandydaci_df, r1_podsumowanie_df, r2_kandydaci_df, r2_podsumowanie_df = loaded_data
    print("Data loaded successfully.")

    # 1. Calculate expected votes based on R1 voters (at station level initially)
    expected_votes_station_df = process_r1_votes_for_expected(r1_kandydaci_df, r1_podsumowanie_df)
    if expected_votes_station_df.empty:
        print("Error: R1 data processing for expected votes resulted in an empty DataFrame. Exiting.")
        return
    print("Station-level R1 votes processed for expected R2 shares.")

    # 2. Calculate new voters at Gmina level
    new_voters_gmina_df = calculate_new_voters_at_gmina(r1_podsumowanie_df, r2_podsumowanie_df)
    if new_voters_gmina_df.empty:
        print("Warning: Gmina-level new voter calculation resulted in an empty DataFrame.")
    print("Gmina-level new voter contributions calculated.")

    # 3. Aggregate station-level expected votes to Gmina and add new voter share
    gmina_expected_df = aggregate_expected_votes_to_gmina(expected_votes_station_df, new_voters_gmina_df)
    if gmina_expected_df.empty:
        print("Error: Aggregating expected votes to Gmina resulted in an empty DataFrame. Exiting.")
        return
    print("Expected R2 votes aggregated to Gmina level and new voters added.")
    
    # 4. Process actual R2 votes at Gmina level
    gmina_actual_df = process_r2_actual_votes_at_gmina(r2_kandydaci_df)
    if gmina_actual_df.empty:
        print("Warning: Processing actual R2 votes at Gmina level resulted in an empty DataFrame.")
    print("Actual R2 votes aggregated to Gmina level.")

    # 5. Merge Gmina-level expected and actual data
    # GEO_COLS_FOR_AGG = ['Wojewodztwo_Name', 'Powiat_MnpP_Name', 'Gmina_Name']
    print(f"\n--- Debug: Before Gmina Merge ---")
    print(f"gmina_expected_df columns: {gmina_expected_df.columns.tolist()}")
    print(f"gmina_actual_df columns: {gmina_actual_df.columns.tolist()}")
    
    comparison_gmina_df = pd.merge(gmina_expected_df, gmina_actual_df, on=GEO_COLS_FOR_AGG, how='outer') # Outer to keep all Gminy
    
    # Fill NaNs that could result from outer merge or empty DFs
    comparison_gmina_df['Expected_Nawrocki_R2'] = comparison_gmina_df['Expected_Nawrocki_R2'].fillna(0.0)
    comparison_gmina_df['Expected_Trzaskowski_R2'] = comparison_gmina_df['Expected_Trzaskowski_R2'].fillna(0.0)
    comparison_gmina_df[f'Actual_{CANDIDATE_NAWROCKI}_R2'] = comparison_gmina_df[f'Actual_{CANDIDATE_NAWROCKI}_R2'].fillna(0.0)
    comparison_gmina_df[f'Actual_{CANDIDATE_TRZASKOWSKI}_R2'] = comparison_gmina_df[f'Actual_{CANDIDATE_TRZASKOWSKI}_R2'].fillna(0.0)
    comparison_gmina_df['R1_Total_Valid_Votes_Gmina'] = comparison_gmina_df.get('R1_Total_Valid_Votes_Gmina', 0.0).fillna(0.0)
    comparison_gmina_df['R2_Total_Valid_Votes_Gmina'] = comparison_gmina_df.get('R2_Total_Valid_Votes_Gmina', 0.0).fillna(0.0)

    print("Gmina-level expected and actual R2 votes merged.")

    # 6. Calculate Differences
    comparison_gmina_df['Diff_Nawrocki'] = comparison_gmina_df[f'Actual_{CANDIDATE_NAWROCKI}_R2'] - comparison_gmina_df['Expected_Nawrocki_R2']
    comparison_gmina_df['Diff_Trzaskowski'] = comparison_gmina_df[f'Actual_{CANDIDATE_TRZASKOWSKI}_R2'] - comparison_gmina_df['Expected_Trzaskowski_R2']
    print("Differences calculated at Gmina level.")

    final_columns_gmina = GEO_COLS_FOR_AGG + \
                    ['R1_Total_Valid_Votes_Gmina', 'R2_Total_Valid_Votes_Gmina', 'Actual_New_Voters_Gmina',
                     'Expected_Nawrocki_R2', f'Actual_{CANDIDATE_NAWROCKI}_R2', 'Diff_Nawrocki',
                     'Expected_Trzaskowski_R2', f'Actual_{CANDIDATE_TRZASKOWSKI}_R2', 'Diff_Trzaskowski']
    
    for col in final_columns_gmina: # Ensure columns exist
        if col not in comparison_gmina_df.columns:
            comparison_gmina_df[col] = 0.0 if "Votes" in col or "Actual_" in col or "Expected_" in col or "Diff_" in col else "N/A"

    final_report_gmina_df = comparison_gmina_df[final_columns_gmina].copy()
    
    float_cols_to_round = ['Expected_Nawrocki_R2', 'Diff_Nawrocki', 
                           'Expected_Trzaskowski_R2', 'Diff_Trzaskowski',
                           f'Actual_{CANDIDATE_NAWROCKI}_R2', f'Actual_{CANDIDATE_TRZASKOWSKI}_R2',
                           'R1_Total_Valid_Votes_Gmina', 'R2_Total_Valid_Votes_Gmina', 'Actual_New_Voters_Gmina']
    for col in float_cols_to_round:
        if col in final_report_gmina_df.columns:
            final_report_gmina_df[col] = final_report_gmina_df[col].round(2)

    print("\n--- Final Comparison Report (Gmina Level - First 5 rows) ---")
    if not final_report_gmina_df.empty:
        print(final_report_gmina_df.head())
    else:
        print("Final Gmina report DataFrame is empty.")

    print("\n--- Summary Statistics for Gmina Differences ---")
    diff_stats_cols = ['Diff_Nawrocki', 'Diff_Trzaskowski']
    if not final_report_gmina_df.empty and all(c in final_report_gmina_df.columns for c in diff_stats_cols):
        print(final_report_gmina_df[diff_stats_cols].describe())
    else:
        print("Difference columns not available for summary or Gmina DataFrame is empty.")
        
    print("\nAnalysis complete (Gmina Level).")
    
    final_report_gmina_df.to_csv("election_comparison_GMINA_level.csv", index=False)
    print("Final report saved to election_comparison_GMINA_level.csv")

    plot_and_save_results(final_report_gmina_df) # Plot will sum Gmina totals

if __name__ == "__main__":
    main()