import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(path="data/creditcard.csv"):
    return pd.read_csv(path)


def split_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state=42):
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    print("Before SMOTE — train class distribution:")
    print(y_train.value_counts())

    X_train_res, y_train_res = apply_smote(X_train, y_train)

    print("\nAfter SMOTE — train class distribution:")
    print(y_train_res.value_counts())

    print("\nTest set (untouched, real distribution):")
    print(y_test.value_counts())