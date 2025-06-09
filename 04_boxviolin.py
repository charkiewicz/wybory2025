import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re # For cleaning filenames

# --- Configuration ---
CSV_FILE_PATH = 'data/polska_prezydent2025_tura2_obkw_kandydaci_NATIONAL_FINAL.csv'
OUTPUT_DIR = 'data/plots/'

# --- Helper function to sanitize filenames ---
def sanitize_filename(name):
    """
    Sanitizes a string to be used as a filename.
    Replaces spaces with underscores and removes characters not typically allowed.
    """
    name = name.replace(' ', '_')
    name = re.sub(r'[^\w\._-]', '', name) # Keep word chars, dots, underscores, hyphens
    return name

# --- Main script ---
if __name__ == "__main__":
    # 1. Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory '{OUTPUT_DIR}' ensured.")

    # 2. Load the CSV data
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Successfully loaded data from '{CSV_FILE_PATH}'.")
    except FileNotFoundError:
        print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
        print("Please make sure the CSV file is in the correct location and the path is correct.")
        # Create a dummy CSV for testing if it's missing
        print("\nCreating a dummy 'election_data.csv' for demonstration purposes...")
        dummy_data = """Wojewodztwo_Name,Powiat_MnpP_Name,Gmina_Name,URL_ID,Candidate,Votes
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404408,BARTOSZEWICZ Artur,2
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404408,BARTOSZEWICZ Artur,3
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404408,BIEJAT Magdalena Agnieszka,32
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404408,BIEJAT Magdalena Agnieszka,35
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404408,BIEJAT Magdalena Agnieszka,150
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404408,TRZASKOWSKI Rafa Kazimierz,361
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404408,TRZASKOWSKI Rafa Kazimierz,370
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404409,BARTOSZEWICZ Artur,6
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404409,BIEJAT Magdalena Agnieszka,29
Województwo dolnośląskie,Powiat bolesawiecki,m. Bolesawiec,1404409,TRZASKOWSKI Rafa Kazimierz,381
Województwo lubelskie,Powiat bialski,m. Biaa Podlaska,1404500,BARTOSZEWICZ Artur,10
Województwo lubelskie,Powiat bialski,m. Biaa Podlaska,1404500,BIEJAT Magdalena Agnieszka,50
Województwo lubelskie,Powiat bialski,m. Biaa Podlaska,1404500,TRZASKOWSKI Rafa Kazimierz,400
Województwo lubelskie,Powiat bialski,m. Biaa Podlaska,1404501,BARTOSZEWICZ Artur,1
Województwo lubelskie,Powiat bialski,m. Biaa Podlaska,1404501,BARTOSZEWICZ Artur,70
Województwo lubelskie,Powiat bialski,m. Biaa Podlaska,1404501,BIEJAT Magdalena Agnieszka,45
Województwo lubelskie,Powiat bialski,m. Biaa Podlaska,1404501,TRZASKOWSKI Rafa Kazimierz,20
"""
        with open(CSV_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(dummy_data)
        df = pd.read_csv(CSV_FILE_PATH)
        print(f"Dummy '{CSV_FILE_PATH}' created and loaded.")
        # exit() # Potentially exit if you don't want to plot dummy data

    # Ensure 'Votes' column is numeric
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
    df.dropna(subset=['Votes'], inplace=True) # Remove rows where 'Votes' could not be converted

    # Get unique candidate names
    candidates = df['Candidate'].unique()
    print(f"\nFound {len(candidates)} unique candidates.")

    # --- 3. Individual Plots for Each Candidate ---
    print("\nGenerating individual plots for each candidate...")
    for candidate_name in candidates:
        print(f"  Processing plots for: {candidate_name}")
        candidate_data = df[df['Candidate'] == candidate_name]
        votes_data = candidate_data['Votes']

        if votes_data.empty or len(votes_data) < 2: # Need at least 2 data points for meaningful box/violin
            print(f"    Not enough data points for {candidate_name} (found {len(votes_data)}). Skipping individual plots.")
            continue

        s_candidate_name = sanitize_filename(candidate_name)

        # Box Plot
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=votes_data, color="skyblue")
        plt.title(f'Vote Distribution (Box Plot) for {candidate_name}')
        plt.ylabel('Number of Votes per Polling Station')
        plt.xlabel(candidate_name) # Or remove if redundant with title
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        box_plot_path = os.path.join(OUTPUT_DIR, f"{s_candidate_name}_boxplot.png")
        plt.savefig(box_plot_path)
        plt.close()
        print(f"    Saved: {box_plot_path}")

        # Violin Plot
        plt.figure(figsize=(8, 6))
        sns.violinplot(y=votes_data, color="lightgreen", inner="quartile") # inner="quartile" shows quartiles
        plt.title(f'Vote Distribution (Violin Plot) for {candidate_name}')
        plt.ylabel('Number of Votes per Polling Station')
        plt.xlabel(candidate_name) # Or remove
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        violin_plot_path = os.path.join(OUTPUT_DIR, f"{s_candidate_name}_violinplot.png")
        plt.savefig(violin_plot_path)
        plt.close()
        print(f"    Saved: {violin_plot_path}")

    # --- 4. Combined Plots for All Candidates ---
    print("\nGenerating combined plots for all candidates...")

    if df.empty or len(df['Candidate'].unique()) == 0:
        print("  No data to generate combined plots. Exiting.")
    else:
        # Determine appropriate figure width for combined plots
        num_candidates = len(df['Candidate'].unique())
        fig_width = max(12, num_candidates * 1.5) # Adjust multiplier as needed

        # Combined Box Plot
        plt.figure(figsize=(fig_width, 8))
        sns.boxplot(x='Candidate', y='Votes', data=df, palette="pastel")
        plt.title('Vote Distribution (Box Plot) - All Candidates')
        plt.xlabel('Candidate')
        plt.ylabel('Number of Votes per Polling Station')
        plt.xticks(rotation=45, ha="right") # Rotate labels for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        combined_box_path = os.path.join(OUTPUT_DIR, "all_candidates_boxplot.png")
        plt.savefig(combined_box_path)
        plt.close()
        print(f"  Saved: {combined_box_path}")

        # Combined Violin Plot
        plt.figure(figsize=(fig_width, 8))
        sns.violinplot(x='Candidate', y='Votes', data=df, palette="pastel", inner="quartile")
        plt.title('Vote Distribution (Violin Plot) - All Candidates')
        plt.xlabel('Candidate')
        plt.ylabel('Number of Votes per Polling Station')
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        combined_violin_path = os.path.join(OUTPUT_DIR, "all_candidates_violinplot.png")
        plt.savefig(combined_violin_path)
        plt.close()
        print(f"  Saved: {combined_violin_path}")

    print("\nAll plots generated and saved.")