import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.router import route_features # Live router from implementation

# --- CONFIGURATION ---
DATA_PATHS = {
    "SMD": "/home/utsab/Downloads/smd/ServerMachineDataset",
    "MSL": "/home/utsab/Downloads/MSL",
    "SMAP": "/home/utsab/Downloads/SMAP",
    "PSM": "/home/utsab/Downloads/PSM"
}

# Source directory for your 4 separate metric files
BENCHMARK_DIR = "dataset_benchmarks"
METRIC_FILES = {
    "SMD": "SMD.csv",
    "MSL": "MSL.csv", 
    "SMAP": "SMAP.csv",
    "PSM": "PSM.csv"
}

def get_routing_split(m_id, dataset, data_root):
    """Loads raw data and executes the live router for any dataset."""
    try:
        if dataset == "SMD":
            path = os.path.join(data_root, "test", f"{m_id}.txt")
            data = np.loadtxt(path, delimiter=',').astype(np.float32)
        elif dataset in ["MSL", "SMAP"]:
            path = os.path.join(data_root, "test", f"{m_id}.npy")
            data = np.load(path).astype(np.float32)
        elif dataset == "PSM":
            path = os.path.join(data_root, "test.csv")
            data = pd.read_csv(path).values[:, 1:].astype(np.float32)
        else: return None, None

        # Execute actual dual-latent routing used in training
        (phy_data, res_data, _, _), _, _ = route_features(data, data)
        return phy_data.shape[1], res_data.shape[1]
    except Exception as e:
        print(f"❌ Error routing {m_id} ({dataset}): {e}")
        return None, None

def plot_aggregate_scalability(ax, df, x_col, title, metrics, colors):
    """Plots all individual points and overlays fleet average."""
    # Step 1: Group by feature count (x_col) for the mean lines
    fleet_avg = df.groupby(x_col)[metrics].mean().reset_index()
    x_val = fleet_avg[x_col].values
    
    for metric, color in zip(metrics, colors):
        # --- NEW: Plot every individual machine point first ---
        # alpha=0.15 ensures that overlapping points show density
        ax.scatter(df[x_col], df[metric], color=color, alpha=0.15, s=30, edgecolors='none')
        
        y_val = fleet_avg[metric].values
        # Plot fleet-wide average points (mean of all datasets at this X)
        sns.lineplot(data=fleet_avg, x=x_col, y=metric, ax=ax, color=color, 
                     marker='o', markersize=12, linewidth=1.5, alpha=0.6)
        
        # Fit 3rd Order Fleet Trendline
        if len(x_val) > 3:
            z = np.polyfit(x_val, y_val, 3)
            p_fit = np.poly1d(z)
            x_range = np.linspace(x_val.min(), x_val.max(), 100)
            ax.plot(x_range, p_fit(x_range), color=color, linewidth=4, 
                    linestyle='--', label=f'Fleet Trend: {metric}')

    ax.set_title(title, fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel(f"Features in {x_col} Branch", fontsize=16)
    ax.set_ylabel("Fleet Avg Metric Score", fontsize=16)
    
    # Scale X-axis to 60 for MSL compatibility (up to 55 features)
    ax.set_xticks(range(0, 61, 5))
    ax.set_xlim(-1, 60)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', frameon=True)

def main():
    print("🚀 Loading separate benchmarks and initiating Global Fleet Analysis...")
    all_fleet_data = []

    for dataset, csv_name in METRIC_FILES.items():
        csv_path = os.path.join(BENCHMARK_DIR, csv_name)
        if not os.path.exists(csv_path):
            print(f"⚠️ Warning: {csv_path} not found. Skipping {dataset}.")
            continue
        
        df_scores = pd.read_csv(csv_path)
        print(f"📂 Processing {dataset} ({len(df_scores)} machines)...")
        # Handle 'id' vs 'machine_id' column naming
        id_col = 'id' if 'id' in df_scores.columns else 'machine_id'

        for _, row in df_scores.iterrows():
            m_id = str(row[id_col]).strip()
            lookup_id = "PSM_Pooled" if dataset == "PSM" else m_id
            
            p, r = get_routing_split(lookup_id, dataset, DATA_PATHS.get(dataset, ""))
            
            if p is not None:
                all_fleet_data.append({
                    'P': p, 'R': r, 
                    'AUC': row['auc'], 'FC1': row['fc1'], 'PAK': row['pa_k_auc'],
                    'Dataset': dataset
                })
        print(f"✅ {dataset} Routing Complete.")

    df_master = pd.DataFrame(all_fleet_data)
    if df_master.empty:
        print("🛑 Error: No data collected from any CSV.")
        return

    # Create 2-panel plot (Consensus vs Residual) - EXACT ORIGINAL SIZE
    fig, axes = plt.subplots(2, 1, figsize=(22, 20))
    metrics, colors = ['AUC', 'FC1', 'PAK'], ['#1f77b4', '#ff7f0e', '#2ca02c']

    plot_aggregate_scalability(axes[0], df_master, 'P', "HNN Consensus Branch: Global Fleet Scalability", metrics, colors)
    plot_aggregate_scalability(axes[1], df_master, 'R', "TCN Residual Branch: Global Fleet Scalability", metrics, colors)

    # EXACT ORIGINAL PLOT SIZE LOGIC
    plt.tight_layout(pad=10.0)
    save_path = "results/analysis/total_fleet_multi_csv_all_points.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\n🏁 Fleet analysis complete. Aggregate plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()