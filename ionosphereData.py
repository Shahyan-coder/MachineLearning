from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_ionosphere_data():
    ionosphere = fetch_ucirepo(id=52)
    X_df = ionosphere.data.features
    y_df = ionosphere.data.targets

    if X_df.isnull().values.any():
        X_df = X_df.dropna()

    if y_df.isnull().values.any():
        y_df = y_df.dropna()

    common_idx = X_df.index.intersection(y_df.index)
    X_df = X_df.loc[common_idx]
    y_df = y_df.loc[common_idx]

    if 'Class' in y_df.columns:
        y = y_df['Class'].map({'g': 1, 'b': 0})
    else:
        y = y_df.iloc[:, 0].map({'g': 1, 'b': 0})

    X = X_df.to_numpy()
    y = y.to_numpy().ravel()

    return X, y

def analyze_data(X, y):
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns=['Class'])

    combined_df = pd.concat([X_df, y_df], axis=1)

    print("\nClass Distribution:")
    print(combined_df['Class'].value_counts())

    print("\nDescriptive Statistics for Features:")
    print(X_df.describe())

    print("\nFeature Correlations:")
    correlation_matrix = X_df.corr()
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.show()

    sns.pairplot(data=combined_df, vars=X_df.columns[0:5], hue='Class')
    plt.show()

def main():
    X, y = process_ionosphere_data()
    analyze_data(X, y)

if __name__ == "__main__":
    main()
