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
    """
    STRICT ALIGNMENT VERSION: Replicates the routing logic.
    Ensures P and R represent the actual dimensions seen by the HNN and TCN.
    """
    train_path = os.path.join(data_root, "train", f"{machine_id}.txt")
    if not os.path.exists(train_path):
        return None, None
    
    train_data = np.loadtxt(train_path, delimiter=',')
    stds = np.std(train_data, axis=0)
    active_indices = np.where(stds > 1e-6)[0]
    train_active = train_data[:, active_indices]
    
    total_active = len(active_indices)
    if total_active < 2:
        return total_active, 0
        
    corr = np.nan_to_num(np.corrcoef(train_active, rowvar=False))
    dist = 1 - np.abs(corr)
    
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=dist_threshold, 
        linkage='complete'
    )
    labels = clustering.fit_predict(dist)
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    p_count = np.max(counts)
    r_count = total_active - p_count
    
    return int(p_count), int(r_count)

def plot_branch_line(ax, df, x_col, title, metrics, colors):
    """Handles the plotting logic for each subplot with added polynomial trend lines."""
    # Group by integer count and take mean
    grouped = df.groupby(x_col)[metrics].mean().reset_index()
    x_data = grouped[x_col].values
    
    for metric, color in zip(metrics, colors):
        y_data = grouped[metric].values
        
        # 1. Plot the jagged raw data line (low alpha to highlight trend)
        sns.lineplot(data=grouped, x=x_col, y=metric, ax=ax, label=f'Mean {metric}', 
                     color=color, marker='o', linewidth=2, markersize=8, alpha=0.4)
        
        # 2. Fit and plot the Trend Line (3rd Order Polynomial)
        # 3rd order is perfect for capturing one major peak or valley (bottleneck)
        if len(x_data) > 3: # Need at least 4 points to fit degree 3
            z = np.polyfit(x_data, y_data, 3)
            p_fit = np.poly1d(z)
            x_smooth = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x_smooth, p_fit(x_smooth), color=color, linewidth=4, 
                    linestyle='--', label=f'Trend {metric}')
    
    ax.set_title(title, fontsize=22, fontweight='bold', pad=20)
    ax.set_xlabel(f"Number of Features in {x_col} Branch", fontsize=16)
    ax.set_ylabel("Mean Metric Score", fontsize=16)
    
    # FORCE INTEGER X-AXIS
    x_min, x_max = int(df[x_col].min()), int(df[x_col].max())
    ax.set_xticks(range(x_min, x_max + 1))
    
    ax.set_ylim(df[metrics].min().min() - 0.05, 1.05) 
    ax.tick_params(labelsize=12)
    # Put legend outside or in columns to accommodate trend labels
    ax.legend(prop={'size': 11}, loc='lower right', ncol=2)
    ax.grid(True, alpha=0.3)

def main():
    print("🚀 Initializing ACAE Bottleneck Analysis with Trends...")
    
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
        print(f"❌ ERROR: No machine data found. Check paths!")
        return

    df = pd.DataFrame(analysis_data)
    print(f"✅ Loaded {len(df)} machines. Generating plots...")

    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, axes = plt.subplots(2, 1, figsize=(24, 16)) 
    metrics = ['AUC', 'FC1', 'PAK']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

    plot_branch_line(axes[0], df, 'P', "Performance vs. Consensus (HNN) Branch: Physics Stability Trend", metrics, colors)
    plot_branch_line(axes[1], df, 'R', "Performance vs. Residual (TCN) Branch: Temporal Noise Trend", metrics, colors)    

    plt.tight_layout(pad=8.0)
    
    output_dir = "results/analysis"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "topology_bottleneck_with_trends.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Analysis complete! Plot saved at: {save_path}")
    
    # Statistical Summary
    print("\n📈 Trend Summary:")
    print("-" * 40)
    best_p = df.groupby('P')['FC1'].mean().idxmax()
    best_r = df.groupby('R')['FC1'].mean().idxmax()
    print(f"FC1 Peaks at Consensus (P) = {best_p}")
    print(f"FC1 Peaks at Residual (R) = {best_r}")
    print("-" * 40)
    
    plt.show()

if __name__ == "__main__":
    main()