import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering

# --- CONFIGURATION ---
DATA_ROOT = "/home/utsab/Downloads/smd/ServerMachineDataset"
RESULTS_CSV = "results/acae_hnn_sobolev_TCN_consensus_duaLatent.csv"
DIST_THRESHOLD = 0.5 

def calculate_topology_split(machine_id, data_root, dist_threshold=0.5):
    """Replicates the routing logic to find P and R."""
    train_path = os.path.join(data_root, "train", f"{machine_id}.txt")
    if not os.path.exists(train_path):
        return None, None
    
    train_data = np.loadtxt(train_path, delimiter=',')
    stds = np.std(train_data, axis=0)
    active_indices = np.where(stds > 1e-6)[0]
    train_active = train_data[:, active_indices]
    
    if train_active.shape[1] < 2:
        return train_active.shape[1], 0
        
    corr = np.nan_to_num(np.corrcoef(train_active, rowvar=False))
    dist = 1 - np.abs(corr)
    
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=dist_threshold, linkage='complete')
    labels = clustering.fit_predict(dist)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    p_count = np.max(counts)
    r_count = train_active.shape[1] - p_count
    return int(p_count), int(r_count)

def plot_branch_line(ax, df, x_col, title, metrics, colors):
    """Handles the plotting logic for each subplot."""
    # Group by integer count and take mean
    grouped = df.groupby(x_col)[metrics].mean().reset_index()
    
    for metric, color in zip(metrics, colors):
        sns.lineplot(data=grouped, x=x_col, y=metric, ax=ax, label=metric, 
                     color=color, marker='o', linewidth=3, markersize=10)
    
    ax.set_title(title, fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel(f"Number of Features in {x_col} Branch", fontsize=16)
    ax.set_ylabel("Mean Metric Score", fontsize=16)
    
    # FORCE INTEGER X-AXIS
    x_min, x_max = int(df[x_col].min()), int(df[x_col].max())
    ax.set_xticks(range(x_min, x_max + 1))
    
    ax.set_ylim(df[metrics].min().min() - 0.05, 1.0) 
    ax.tick_params(labelsize=12)
    ax.legend(prop={'size': 14}, loc='lower right')

def main():
    print("🚀 Initializing ACAE Bottleneck Analysis...")
    
    # --- DATA EXTRACTION ---
    if not os.path.exists(RESULTS_CSV):
        print(f"❌ ERROR: Results file {RESULTS_CSV} not found!")
        return

    print("🧬 Probing topologies and loading CSV scores...")
    df_metrics = pd.read_csv(RESULTS_CSV)
    analysis_data = []

    for idx, row in df_metrics.iterrows():
        m_id = str(row['machine_id']).strip()
        if 'machine' not in m_id.lower():
            continue
        
        p, r = calculate_topology_split(m_id, DATA_ROOT, DIST_THRESHOLD)
        
        if p is not None:
            analysis_data.append({
                'P': p, 
                'R': r, 
                'AUC': row['auc'], 
                'FC1': row['fc1'], 
                'PAK': row['pa_k_auc']
            })

    if not analysis_data:
        print(f"❌ ERROR: No machine data found in {DATA_ROOT}. Check your paths!")
        return

    df = pd.DataFrame(analysis_data)
    print(f"✅ Loaded {len(df)} machines. Creating wide line plot...")

    # --- 🚀 THE WIDE LINE PLOT ---
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, axes = plt.subplots(2, 1, figsize=(22, 14)) 
    metrics = ['AUC', 'FC1', 'PAK']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

    plot_branch_line(axes[0], df, 'P', "Performance vs. Consensus (HNN) Branch Size", metrics, colors)
    plot_branch_line(axes[1], df, 'R', "Performance vs. Residual (TCN) Branch Size", metrics, colors)    

    plt.tight_layout(pad=6.0)
    
    # Ensure result directory exists
    output_dir = "results/analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, "topology_bottleneck_line_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Analysis complete!")
    print(f"📍 Plot saved at: {save_path}")
    
    # Summary for the supervisor
    print("\n📈 Statistical Summary by Branch Size:")
    print("-" * 40)
    best_p = df.groupby('P')['FC1'].mean().idxmax()
    best_r = df.groupby('R')['FC1'].mean().idxmax()
    print(f"Optimal Consensus size (P) for FC1: {best_p}")
    print(f"Optimal Residual size (R) for FC1: {best_r}")
    print("-" * 40)
    
    plt.show()

if __name__ == "__main__":
    main()
