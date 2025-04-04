import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# Helper: Load and process one file
def process_file(filename):
    df = pd.read_csv(filename, index_col=False, comment='#')
    df["neg_sample"] = df["neg_sample"].astype(str).str.strip()
    df["auc_roc"] = pd.to_numeric(df["au_roc_score"], errors="coerce")

    neg_sample_map = {
        "rnd": "Standard",
        "hist_nre": "Historical",
        "induc_nre": "Inductive",
        "rp_ns": "RP-NS"
    }
    df["Setting"] = df["neg_sample"].map(neg_sample_map)

    # Keep best per (dataset, Setting)
    best_df = df.loc[df.groupby(["dataset", "Setting"])["auc_roc"].idxmax()].copy()
    return best_df[["dataset", "Setting", "auc_roc"]]

# Patterns for each method
method_patterns = {
    "InterBase": "base3_*.csv",
    "EdgeBank": "edgebank_*.csv",
    "PopTrack": "poptrack_*.csv"
}

all_best_dfs = []
version_labels = {}

# Select best version per method
for method, pattern in method_patterns.items():
    candidates = glob.glob(pattern)
    best_score = -1
    best_df = None
    best_file = None

    for fname in candidates:
        try:
            df = process_file(fname)
            avg_auc = df["auc_roc"].mean()
            if avg_auc > best_score:
                best_score = avg_auc
                best_df = df
                best_file = fname
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

    if best_df is not None:
        version = os.path.splitext(os.path.basename(best_file))[0]
        label = method  # Simpler label, without filename
        print(f"Selected best version for {method}: {best_file}")
        best_df["Method"] = label
        version_labels[method] = label
        all_best_dfs.append(best_df)

# Combine all selected best versions
full_df = pd.concat(all_best_dfs, ignore_index=True)

# --- Plot 1: Per-dataset performance for InterBase ---
base_label = version_labels.get("InterBase", "InterBase")
base_df = full_df[full_df["Method"] == base_label]
setting_avg = base_df.groupby("Setting", as_index=False)["auc_roc"].mean()

sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=base_df,
    x="Setting",
    y="auc_roc",
    hue="dataset",
    palette="Set2"
)

#for container in ax.containers:
   # ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=8)

for i, row in setting_avg.iterrows():
    setting = row["Setting"]
    y = row["auc_roc"]
    group_center = list(setting_avg["Setting"]).index(setting)
    x_start = group_center - 0.4
    x_end = group_center + 0.4
    ax.plot([x_start, x_end], [y, y], color="black", linewidth=1.5)
    ax.text(group_center, y + 0.015, f"{y:.2f}", ha="center", va="bottom", fontsize=9)

plt.ylabel("Best AU-ROC")
plt.xlabel("")
plt.ylim(0, 1)
plt.legend(title="", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=4)
plt.tight_layout()
plt.title("Per-dataset performance for InterBase")
plt.show()

# --- Plot 2: Compare all best versions ---
comparison_avg = full_df.groupby(["Setting", "Method"], as_index=False)["auc_roc"].mean()

plt.figure(figsize=(8, 5))
ax2 = sns.barplot(
    data=comparison_avg,
    x="Setting",
    y="auc_roc",
    hue="Method",
    palette="Set1"
)

for container in ax2.containers:
    ax2.bar_label(container, fmt="%.2f", label_type="edge", fontsize=8)

plt.ylabel("Avg. AU-ROC")
plt.xlabel("")
plt.ylim(0, 1)
plt.legend(title="", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=4)
plt.tight_layout()
plt.title("Comparison of Best Versions per Method")
plt.show()