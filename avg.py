import pandas as pd
import os

def calculate_fleet_averages(name, file_path):
    if not os.path.exists(file_path):
        print(f"⚠️ File not found for {name}: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path)
        # Clean whitespace from headers just in case
        df.columns = df.columns.str.strip()
        
        # Core metrics to average
        metrics = ['auc', 'fc1', 'pa_k_auc']
        
        # Check for ID columns based on your specific dataset mappings
        if name == "SMD":
            id_col = 'machine_id'
        else:
            id_col = 'id'
            
        if id_col not in df.columns:
            # Fallback: if 'id' isn't there, just look for whatever the first column is
            id_col = df.columns[0]
            print(f"🔍 {name}: Using '{id_col}' as the identifier.")

        # Ensure metrics are numeric (handles empty rows or strings)
        for col in metrics:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows where metrics might be NaN (failed runs)
        df = df.dropna(subset=metrics)
        
        return {
            "count": len(df),
            "means": df[metrics].mean()
        }
    except Exception as e:
        print(f"💥 Error processing {name}: {e}")
        return None

# Mapping paths based on your previous directory structure
files = {
    "PSM": "dataset_benchmarks/PSM.csv",
    "SMAP": "dataset_benchmarks/SMAP.csv",
    "SMD": "dataset_benchmarks/SMD.csv",
    "MSL": "dataset_benchmarks/MSL.csv"
}

print("📊 Dual-Anchor Engine: Final Benchmark Summary")
print("="*50)

for name, path in files.items():
    result = calculate_fleet_averages(name, path)
    if result is not None:
        avg = result["means"]
        print(f"\n🚀 {name} ({result['count']} entities):")
        print(f"  AUC:   {avg['auc']:.4f}")
        print(f"  Fc1:   {avg['fc1']:.4f}")
        print(f"  PA%K:  {avg['pa_k_auc']:.4f}")

print("\n" + "="*50)