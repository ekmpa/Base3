import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
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

# Group by dataset + setting, average AUROC
avg_df = df.groupby(["dataset", "Setting"], as_index=False)["auc_roc"].mean()

# Plotting
sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))

sns.barplot(
    data=avg_df,
    x="Setting",
    y="auc_roc",
    hue="dataset",
    palette="Set2"
)

plt.ylabel("Avg. AU-ROC")
plt.xlabel("")
plt.ylim(0, 1)
plt.legend(title="", loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=4)
plt.tight_layout()
plt.show()