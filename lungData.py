from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def process_lung_cancer_data():
    lung_cancer = fetch_ucirepo(id=62)
    X_df = lung_cancer.data.features
    y_df = lung_cancer.data.targets

    X_df = X_df.replace('?', np.NaN)
    y_df = y_df.replace('?', np.NaN)

    if X_df.isnull().values.any():
        X_df = X_df.fillna(X_df.mode().iloc[0])

    if y_df.isnull().values.any():
        y_df = y_df.fillna(y_df.mode().iloc[0])

    common_idx = X_df.index.intersection(y_df.index)
    X_df = X_df.loc[common_idx]
    y_df = y_df.loc[common_idx]

    if 'class' in y_df.columns:
        y = y_df['class']
    else:
        y = y_df.iloc[:, 0]

    X = X_df.to_numpy()
    y = y.to_numpy().ravel()

    return X, y

def analyze_data(X, y):
    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y, columns=['class'])

    combined_df = pd.concat([X_df, y_df], axis=1)

    print("\nClass Distribution:")
    print(combined_df['class'].value_counts())

    print("\nDescriptive Statistics for Features:")
    print(X_df.describe())

    print("\nFeature Correlations:")
    correlation_matrix = X_df.corr()
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.show()

    sns.pairplot(data=combined_df, vars=X_df.columns[0:5], hue='class')
    plt.show()


def main():
    X, y = process_lung_cancer_data()
    analyze_data(X, y)

if __name__ == "__main__":
    main()
