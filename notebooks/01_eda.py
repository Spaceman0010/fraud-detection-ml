import pandas as pd
import numpy as np

df = pd.read_csv("data/creditcard.csv")

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nNull values per column:\n", df.isnull().sum().sum(), "total nulls")

print("\nClass distribution:")
print(df["Class"].value_counts())
print("\nFraud percentage:", round(df["Class"].value_counts(normalize=True)[1] * 100, 4), "%")

print("\nAmount stats by class:")
print(df.groupby("Class")["Amount"].describe())

print("\nTime range (seconds):", df["Time"].min(), "to", df["Time"].max())
print("Time range (hours):", round(df["Time"].max() / 3600, 1))