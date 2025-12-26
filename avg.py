import pandas as pd
import io

def calculate_fleet_averages(file_path):
    # This handles files that have extra info lines at the top
    try:
        # First attempt: standard CSV
        df = pd.read_csv(file_path, comment='#')
        # If 'machine_id' isn't the first column, it might be a commented header
        if 'machine_id' not in df.columns:
            df = pd.read_csv(file_path, skiprows=1)
    except Exception:
        print(f"Skipping {file_path} - check format.")
        return None

    # Calculate means for the core metrics
    averages = df[['auc', 'fc1', 'pa_k_auc']].mean()
    return averages

# --- YOUR FILES ---
files = {
    "Random Masking, 1:1 with acae paper": "results/acae_randm_smd_results.csv",
    "Univariate jerk + 50% veto multivariate + weighted loss": "results/acae_jerk_smd_weightedLoss.csv",
    "Univariate jerk + 50% veto multivariate ": "results/acae_jerk_smd_results.csv",
    "Univariate jerk + consensus mask + weighted loss": "results/acae_jerk_smd_consensusMask.csv",
    "Univariate jerk + consensus mask + weighted loss+ High-Res HNN": "results/acae_hnn_sobolev_consensus.csv",
    "Univariate jerk + consensus mask + weighted loss+ High-Res HNN + dual latent": "results/acae_hnn_sobolev_consensus_duaLatent.csv"
}

print("--- GLOBAL FLEET AVERAGES (28 MACHINES) ---")
for name, path in files.items():
    avg = calculate_fleet_averages(path)
    if avg is not None:
        print(f"\n🚀 {name}:")
        print(f"  AUC:   {avg['auc']:.4f}")
        print(f"  Fc1:   {avg['fc1']:.4f}")
        print(f"  PA%K:  {avg['pa_k_auc']:.4f}")