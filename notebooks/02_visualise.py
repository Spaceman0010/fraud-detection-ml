import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/creditcard.csv")

# --- Plot 1: Class imbalance ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(data=df, x="Class", ax=axes[0])
axes[0].set_title("Class Distribution (raw counts)")
axes[0].set_xticklabels(["Legitimate", "Fraud"])

df["Class"].value_counts(normalize=True).plot(kind="bar", ax=axes[1], color=["steelblue", "crimson"])
axes[1].set_title("Class Distribution (%)")
axes[1].set_xticklabels(["Legitimate", "Fraud"], rotation=0)

plt.tight_layout()
plt.savefig("notebooks/class_distribution.png")
print("Saved notebooks/class_distribution.png")

# --- Plot 2: Amount by class ---
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Class", y="Amount", ax=ax)
ax.set_yscale("log")
ax.set_xticklabels(["Legitimate", "Fraud"])
ax.set_title("Transaction Amount by Class (log scale)")
plt.tight_layout()
plt.savefig("notebooks/amount_by_class.png")
print("Saved notebooks/amount_by_class.png")