import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

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
ADDRESS_KEY_COL = 'Address_Key'
GEO_COLS_BASE = ['Wojewodztwo_Name', 'Powiat_MnpP_Name', 'Gmina_Name']
GEO_COLS_FOR_AGG = GEO_COLS_BASE + [ADDRESS_KEY_COL]
R1_SUMMARY_VOTES_COL = 'liczba_glosow_waznych_na_kandydatow'
R2_SUMMARY_VOTES_COL = 'liczba_glosow_waznych_na_kandydatow'

# --- Helper Functions ---

def load_and_prepare_data():
    try:
        r1_kandaci_df = pd.read_csv(FILE_R1_KANDYDACI)
        r1_podsumowanie_df = pd.read_csv(FILE_R1_PODSUMOWANIE)
        r2_kandaci_df = pd.read_csv(FILE_R2_KANDYDACI)
        r2_podsumowanie_df = pd.read_csv(FILE_R2_PODSUMOWANIE)
    except FileNotFoundError as e:
        print(f"Error: File not found. Please check file paths. Missing file: {e.filename}")
        return None, None
        
    # --- FIX 1: Clean the R2 Summary DataFrame ---
    # Filter out malformed header/summary rows that are not actual polling stations.
    # The 'na=False' ensures that any NaN values in the address column are not considered matches.
    r2_podsumowanie_df = r2_podsumowanie_df[
        ~r2_podsumowanie_df['Commission_Address_Raw'].str.contains("Wybory Prezydenta", na=False)
    ].copy()

    # --- FIX 2: Enforce numeric types on crucial columns to prevent silent errors ---
    # Coerce errors will turn any non-numeric values (from bad rows) into NaN.
    # This prevents the whole column from being treated as text ('object').
    r1_kandaci_df['Votes'] = pd.to_numeric(r1_kandaci_df['Votes'], errors='coerce')
    r2_kandaci_df['Votes'] = pd.to_numeric(r2_kandaci_df['Votes'], errors='coerce')
    r1_podsumowanie_df[R1_SUMMARY_VOTES_COL] = pd.to_numeric(r1_podsumowanie_df[R1_SUMMARY_VOTES_COL], errors='coerce')
    r2_podsumowanie_df[R2_SUMMARY_VOTES_COL] = pd.to_numeric(r2_podsumowanie_df[R2_SUMMARY_VOTES_COL], errors='coerce')

    # Drop any rows where the conversion to numeric failed
    r1_kandaci_df.dropna(subset=['Votes'], inplace=True)
    r2_kandaci_df.dropna(subset=['Votes'], inplace=True)
    r1_podsumowanie_df.dropna(subset=[R1_SUMMARY_VOTES_COL], inplace=True)
    r2_podsumowanie_df.dropna(subset=[R2_SUMMARY_VOTES_COL], inplace=True)

    if r1_kandaci_df.empty or r1_podsumowanie_df.empty or r2_kandaci_df.empty or r2_podsumowanie_df.empty:
        print("Error: One or more data files are empty after cleaning.")
        return None, None

    # Merge candidate votes with address and summary data
    cols_to_add_from_summary = ['URL_ID', 'Commission_Address_Raw']
    r1_podsumowanie_to_merge = r1_podsumowanie_df[cols_to_add_from_summary + [R1_SUMMARY_VOTES_COL]].copy()
    r2_podsumowanie_to_merge = r2_podsumowanie_df[cols_to_add_from_summary + [R2_SUMMARY_VOTES_COL]].copy()
    
    # Merge R1 data
    r1_full_df = pd.merge(r1_kandaci_df, r1_podsumowanie_to_merge, on='URL_ID', how='left')
    
    # Merge R2 data
    r2_full_df = pd.merge(r2_kandaci_df, r2_podsumowanie_to_merge, on='URL_ID', how='left')
    
    # Create the stable Address Key on the now-complete dataframes
    add_and_clean_address_key(r1_full_df)
    add_and_clean_address_key(r2_full_df)
    print("Data loaded, cleaned, and enriched with geographic information.")
    
    return r1_full_df, r2_full_df

def add_and_clean_address_key(df):
    """Creates a cleaned, lowercase, stripped address key for stable merging."""
    df[ADDRESS_KEY_COL] = df['Commission_Address_Raw'].astype(str).str.lower().str.strip()
    return df

def process_r1_and_calculate_expected_address_votes(r1_full_df):
    """Aggregates R1 votes at the Address level and calculates expected R2 votes."""
    r1_address_votes = r1_full_df.groupby(GEO_COLS_FOR_AGG + ['Candidate'])['Votes'].sum().unstack(fill_value=0)
    
    all_r1_candidates = list(r1_full_df['Candidate'].unique())
    for cand in all_r1_candidates:
        if cand not in r1_address_votes.columns:
            r1_address_votes[cand] = 0

    df_calc = r1_address_votes.copy()
    df_calc['Expected_Nawrocki_R2'] = 0.0
    df_calc['Expected_Trzaskowski_R2'] = 0.0

    if CANDIDATE_NAWROCKI in df_calc.columns: df_calc['Expected_Nawrocki_R2'] += df_calc[CANDIDATE_NAWROCKI]
    if CANDIDATE_TRZASKOWSKI in df_calc.columns: df_calc['Expected_Trzaskowski_R2'] += df_calc[CANDIDATE_TRZASKOWSKI]

    for r1_cand, transfers in TRANSFER_RULES.items():
        if r1_cand in df_calc.columns:
            df_calc['Expected_Nawrocki_R2'] += df_calc[r1_cand] * transfers.get(CANDIDATE_NAWROCKI, 0.0)
            df_calc['Expected_Trzaskowski_R2'] += df_calc[r1_cand] * transfers.get(CANDIDATE_TRZASKOWSKI, 0.0)
            
    other_r1_candidates = [cand for cand in all_r1_candidates if cand not in R2_FINALISTS_FROM_R1 and cand not in SPECIFIED_TRANSFER_CANDIDATES_R1]
    existing_other_cands = [c for c in other_r1_candidates if c in df_calc.columns]
    if existing_other_cands:
        df_calc['Other_R1_Votes_Sum'] = df_calc[existing_other_cands].sum(axis=1)
        df_calc['Expected_Nawrocki_R2'] += df_calc['Other_R1_Votes_Sum'] * 0.50
        df_calc['Expected_Trzaskowski_R2'] += df_calc['Other_R1_Votes_Sum'] * 0.50
    
    return df_calc[['Expected_Nawrocki_R2', 'Expected_Trzaskowski_R2']].reset_index()

def calculate_new_voters_at_address(r1_full_df, r2_full_df):
    """Calculates new voters by comparing R1 and R2 total valid votes at Address level."""
    r1_station_totals = r1_full_df.drop_duplicates(subset=['URL_ID'])
    r1_address_totals = r1_station_totals.groupby(GEO_COLS_FOR_AGG)[R1_SUMMARY_VOTES_COL].sum().reset_index()
    r1_address_totals.rename(columns={R1_SUMMARY_VOTES_COL: 'R1_Total_Valid_Votes_Addr'}, inplace=True)

    r2_station_totals = r2_full_df.drop_duplicates(subset=['URL_ID'])
    r2_address_totals = r2_station_totals.groupby(GEO_COLS_FOR_AGG)[R2_SUMMARY_VOTES_COL].sum().reset_index()
    r2_address_totals.rename(columns={R2_SUMMARY_VOTES_COL: 'R2_Total_Valid_Votes_Addr'}, inplace=True)

    address_summary = pd.merge(r1_address_totals, r2_address_totals, on=GEO_COLS_FOR_AGG, how='outer')
    address_summary.fillna(0, inplace=True)

    net_new_voters = address_summary['R2_Total_Valid_Votes_Addr'] - address_summary['R1_Total_Valid_Votes_Addr']
    address_summary['Actual_New_Voters_Addr'] = net_new_voters.clip(lower=0)
    
    return address_summary

def process_r2_actual_votes_at_address(r2_full_df):
    """Aggregates actual R2 votes at the Address level."""
    r2_finalists_votes = r2_full_df[r2_full_df['Candidate'].isin(R2_FINALISTS_FROM_R1)]
    r2_address_actual = r2_finalists_votes.groupby(GEO_COLS_FOR_AGG + ['Candidate'])['Votes'].sum().unstack(fill_value=0)
    
    r2_address_actual.rename(columns={
        CANDIDATE_NAWROCKI: f'Actual_{CANDIDATE_NAWROCKI}_R2',
        CANDIDATE_TRZASKOWSKI: f'Actual_{CANDIDATE_TRZASKOWSKI}_R2'
    }, inplace=True)
    
    if f'Actual_{CANDIDATE_NAWROCKI}_R2' not in r2_address_actual.columns: r2_address_actual[f'Actual_{CANDIDATE_NAWROCKI}_R2'] = 0.0
    if f'Actual_{CANDIDATE_TRZASKOWSKI}_R2' not in r2_address_actual.columns: r2_address_actual[f'Actual_{CANDIDATE_TRZASKOWSKI}_R2'] = 0.0
        
    return r2_address_actual.reset_index()

def plot_and_save_results(df, filename="election_comparison_R2_ADDRESS_totals.png"):
    if df.empty:
        print("Cannot plot results, DataFrame is empty.")
        return
    print("\n--- Plotting Results (National Totals from Address Aggregation) ---")
    total_expected_nawrocki = df['Expected_Nawrocki_R2'].sum()
    total_actual_nawrocki = df[f'Actual_{CANDIDATE_NAWROCKI}_R2'].sum()
    total_expected_trzaskowski = df['Expected_Trzaskowski_R2'].sum()
    total_actual_trzaskowski = df[f'Actual_{CANDIDATE_TRZASKOWSKI}_R2'].sum()
    print(f"Total Expected Nawrocki: {total_expected_nawrocki:,.0f}")
    print(f"Total Actual Nawrocki: {total_actual_nawrocki:,.0f}")
    print(f"Total Expected Trzaskowski: {total_expected_trzaskowski:,.0f}")
    print(f"Total Actual Trzaskowski: {total_actual_trzaskowski:,.0f}")
    candidates_plot = [CANDIDATE_NAWROCKI, CANDIDATE_TRZASKOWSKI]
    expected_votes_plot = [total_expected_nawrocki, total_expected_trzaskowski]
    actual_votes_plot = [total_actual_nawrocki, total_actual_trzaskowski]
    plot_data = pd.DataFrame({'Candidate': candidates_plot * 2, 'Vote_Type': ['Expected'] * 2 + ['Actual'] * 2, 'Total_Votes': expected_votes_plot + actual_votes_plot})
    plt.figure(figsize=(12, 7)); sns.set_style("whitegrid")
    bar_plot = sns.barplot(x='Candidate', y='Total_Votes', hue='Vote_Type', data=plot_data, palette={"Expected": "skyblue", "Actual": "salmon"})
    for p in bar_plot.patches:
        assert isinstance(p, Rectangle)
        bar_plot.annotate(format(p.get_height(), ',.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')
    plt.title('National Total Votes (from Address-Level Aggregation): Expected vs. Actual (Round 2)', fontsize=14)
    plt.ylabel('Total Votes', fontsize=12); plt.xlabel('Candidate', fontsize=12)
    plt.xticks(rotation=10, ha='right', fontsize=10); plt.yticks(fontsize=10)
    plt.legend(title='Vote Type'); plt.tight_layout()
    try:
        plt.savefig(filename); print(f"Plot saved to {filename}")
    except Exception as e: print(f"Error saving plot: {e}")

def main():
    print("Starting election analysis (Address Level Comparison)...")
    
    # FIX: Unpack directly and then check for failure (None).
    r1_full_df, r2_full_df = load_and_prepare_data()

    # The load function returns (None, None) on failure. Check for this.
    if r1_full_df is None or r2_full_df is None:
        print("Critical Error: Data loading and preparation failed. Exiting.")
        return

    # 1. Calculate expected votes from R1 transfers at the Address level.
    address_expected_df = process_r1_and_calculate_expected_address_votes(r1_full_df)
    print("Expected R2 votes from R1 transfers calculated at Address level.")

    # 2. Calculate new voters at the Address level.
    new_voters_address_df = calculate_new_voters_at_address(r1_full_df, r2_full_df)
    print("Address-level new voter contributions calculated.")
    
    # 3. Process actual R2 votes at the Address level.
    address_actual_df = process_r2_actual_votes_at_address(r2_full_df)
    print("Actual R2 votes aggregated to Address level.")

    # 4. Merge dataframes on the stable GEO_COLS_FOR_AGG key.
    comparison_df = pd.merge(address_expected_df, new_voters_address_df, on=GEO_COLS_FOR_AGG, how='left')
    
    comparison_df['Actual_New_Voters_Addr'] = comparison_df['Actual_New_Voters_Addr'].fillna(0)
    comparison_df['Expected_Nawrocki_R2'] += comparison_df['Actual_New_Voters_Addr'] * NEW_VOTER_DISTRIBUTION[CANDIDATE_NAWROCKI]
    comparison_df['Expected_Trzaskowski_R2'] += comparison_df['Actual_New_Voters_Addr'] * NEW_VOTER_DISTRIBUTION[CANDIDATE_TRZASKOWSKI]
    print("New voter shares added to expected Address totals.")
    
    comparison_df = pd.merge(comparison_df, address_actual_df, on=GEO_COLS_FOR_AGG, how='outer')
    print("Address-level expected and actual R2 votes merged.")

    # 5. Clean up NaNs from the outer merge and calculate differences.
    float_cols_to_fill = ['Expected_Nawrocki_R2', 'Expected_Trzaskowski_R2', f'Actual_{CANDIDATE_NAWROCKI}_R2', f'Actual_{CANDIDATE_TRZASKOWSKI}_R2', 'R1_Total_Valid_Votes_Addr', 'R2_Total_Valid_Votes_Addr', 'Actual_New_Voters_Addr']
    for col in float_cols_to_fill:
        if col in comparison_df.columns:
            comparison_df[col] = comparison_df[col].fillna(0.0)

    comparison_df['Diff_Nawrocki'] = comparison_df[f'Actual_{CANDIDATE_NAWROCKI}_R2'] - comparison_df['Expected_Nawrocki_R2']
    comparison_df['Diff_Trzaskowski'] = comparison_df[f'Actual_{CANDIDATE_TRZASKOWSKI}_R2'] - comparison_df['Expected_Trzaskowski_R2']
    print("Differences calculated at Address level.")

    # 6. Prepare final report
    final_columns = GEO_COLS_FOR_AGG + ['R1_Total_Valid_Votes_Addr', 'R2_Total_Valid_Votes_Addr', 'Actual_New_Voters_Addr', 'Expected_Nawrocki_R2', f'Actual_{CANDIDATE_NAWROCKI}_R2', 'Diff_Nawrocki', 'Expected_Trzaskowski_R2', f'Actual_{CANDIDATE_TRZASKOWSKI}_R2', 'Diff_Trzaskowski']
    final_report_df = comparison_df.reindex(columns=final_columns).fillna(0.0)
    float_cols_to_round = [c for c in final_report_df.columns if c not in GEO_COLS_FOR_AGG]
    final_report_df[float_cols_to_round] = final_report_df[float_cols_to_round].round(2)

    print("\n--- Final Comparison Report (Address Level - First 5 rows) ---")
    print(final_report_df.head())

    print("\n--- Summary Statistics for Address-Level Differences ---")
    print(final_report_df[['Diff_Nawrocki', 'Diff_Trzaskowski']].describe())
        
    print("\nAnalysis complete (Address Level).")
    
    final_report_df.to_csv("election_comparison_ADDRESS_level.csv", index=False)
    print("Final report saved to election_comparison_ADDRESS_level.csv")

    plot_and_save_results(final_report_df)

if __name__ == "__main__":
    main()