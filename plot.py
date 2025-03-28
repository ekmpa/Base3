import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("stats_01.csv", index_col=False)
df["neg_sample"] = df["neg_sample"].astype(str).str.strip()

# Map to friendly labels
neg_sample_map = {
    "rnd": "Standard",
    "hist_nre": "Historical",
    "induc_nre": "Inductive"
}
df["Setting"] = df["neg_sample"].map(neg_sample_map)
df["auc_roc"] = pd.to_numeric(df["au_roc_score"], errors="coerce")

# Keep the best AUC per (dataset, setting)
best_df = df.loc[df.groupby(["dataset", "Setting"])["auc_roc"].idxmax()].copy()

# Prepare for first figure: per-dataset performance per setting
avg_df = best_df[["dataset", "Setting", "auc_roc"]]

# First barplot: average per-dataset AU-ROC per setting
setting_avg = avg_df.groupby("Setting", as_index=False)["auc_roc"].mean()

# --- First Plot ---
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=avg_df,
    x="Setting",
    y="auc_roc",
    hue="dataset",
    palette="Set2"
)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=8)

# Horizontal line & label for average per setting
for i, row in setting_avg.iterrows():
    setting = row["Setting"]
    y = row["auc_roc"]
    group_center = list(setting_avg["Setting"]).index(setting)
    n_hues = avg_df["dataset"].nunique()
    bar_width = 0.8
    group_width = bar_width
    x_start = group_center - group_width / 2
    x_end = group_center + group_width / 2
    ax.plot([x_start, x_end], [y, y], color="black", linewidth=1.5)
    ax.text(group_center, y + 0.015, f"{y:.2f}", ha="center", va="bottom", color="black", fontsize=9)

plt.ylabel("Best AU-ROC")
plt.xlabel("")
plt.ylim(0, 1)
plt.legend(title="", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=4)
plt.tight_layout()
plt.show()

# --- Second Plot: Comparison per method ---
# Example placeholders – you can update these manually
edgebank = {
    "Standard": 0.76,
    "Historical": 0.75,
    "Inductive": 0.47
}
#poptrack = {"Standard": 0.72, "Historical": 0.69, "Inductive": 0.67}

# Create comparison dataframe
comparison_df = pd.DataFrame({
    "Setting": setting_avg["Setting"],
    "Base3": setting_avg["auc_roc"],
    "EdgeBank": setting_avg["Setting"].map(edgebank),
    #"PopTrack": setting_avg["Setting"].map(poptrack)
}).melt(id_vars="Setting", var_name="Method", value_name="AU-ROC")

# Second barplot
plt.figure(figsize=(8, 5))
ax2 = sns.barplot(
    data=comparison_df,
    x="Setting",
    y="AU-ROC",
    hue="Method",
    palette="Set1"
)

# Add bar labels
for container in ax2.containers:
    ax2.bar_label(container, fmt="%.2f", label_type="edge", fontsize=8)

plt.ylabel("Avg. AU-ROC")
plt.xlabel("")
plt.ylim(0, 1)
plt.legend(title="", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=3)
plt.tight_layout()
plt.show()