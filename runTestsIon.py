import numpy as np
import matplotlib.pyplot as plt
from knn import KNearestNeighbors, cross_validate_for_k as knn_cv
from logReg import LogisticRegression, k_fold_cross_validation as lr_cv
from ionosphereData import process_ionosphere_data

import matplotlib.pyplot as plt

def plot_accuracy(title, xlabel, ylabel, x_values, y_values):
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', label='Accuracy per K')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(loc='best')
    plt.show()

def compare_accuracy(X, y):
    k_values = range(1, 20)
    best_k = None
    best_accuracy = 0

    print("Starting KNN cross-validation...")
    for k in k_values:
        print(f"Evaluating K={k}...")
        current_accuracy = knn_cv(X, y, k)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_k = k

    print("Starting Logistic Regression cross-validation...")
    lr_accuracy = lr_cv(LogisticRegression(), X, y)
    
    print(f"Best K for KNN: {best_k} with accuracy of {best_accuracy}")
    print(f"Accuracy for Logistic Regression: {lr_accuracy}")


def test_k_values(X, y):
    k_values = list(range(1, 20))
    accuracies = []

    print("Testing different K values...")
    for k in k_values:
        print(f"Testing K={k}...")
        accuracy = knn_cv(X, y, k)
        if accuracy is not None:
            accuracies.append(accuracy)
        else:
            print(f"Error: k={k} produced invalid accuracy.")
            accuracies.append(0)

    if not accuracies or all(accuracy == 0 for accuracy in accuracies):
        print("Error: No accuracy data available.")
        return None, None

    best_k_index = np.argmax(accuracies)
    if best_k_index >= len(k_values):
        print("Error: Best index is out of range.")
        return None, None

    best_k = k_values[best_k_index]
    best_accuracy = accuracies[best_k_index]
    plot_accuracy("KNN Accuracy with Different K", "K value", "Accuracy", k_values, accuracies)
    print(f"Best K: {best_k} with accuracy of {best_accuracy}")
    return best_k, best_accuracy


def test_learning_rates(X, y):
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    accuracies = []
    iteration_counts = []

    print("Testing different learning rates for Logistic Regression...")
    for lr in learning_rates:
        print(f"Testing learning rate={lr}...")
        model = LogisticRegression(learning_rate=lr)
        accuracy, iters = lr_cv(model, X, y, return_iterations=True)
        accuracies.append(accuracy)
        iteration_counts.append(iters)
    plot_accuracy("LR Accuracy with Different Learning Rates", "Learning Rates", "Accuracy", learning_rates, accuracies)
    plot_accuracy("LR Convergence Speed with Different Learning Rates", "Learning Rates", "Iterations to Converge", learning_rates, iteration_counts)

def compare_accuracy_by_dataset_size(X, y, best_k):
    train_sizes = np.linspace(0.1, 1.0, 10)
    knn_accuracies = []
    lr_accuracies = []

    print("Comparing accuracies by dataset size...")
    for size in train_sizes:
        print(f"Testing dataset size={size}...")
        train_size = int(size * X.shape[0])
        X_train = X[:train_size]
        y_train = y[:train_size]

        knn_accuracy = knn_cv(X_train, y_train, best_k)
        knn_accuracies.append(knn_accuracy)
        
        lr_accuracy = lr_cv(LogisticRegression(), X_train, y_train)
        lr_accuracies.append(lr_accuracy)

    plot_accuracy("Model Accuracy vs Dataset Size - KNN", "Dataset Size", "Accuracy", train_sizes, knn_accuracies)
    plt.plot(train_sizes, lr_accuracies, label='Logistic Regression')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    X, y = process_ionosphere_data()
    
    compare_accuracy(X, y)
    
    test_learning_rates(X, y)

    best_k, best_accuracy = test_k_values(X, y)
    print(f"Best K value found: {best_k} with accuracy of {best_accuracy}")
    
    compare_accuracy_by_dataset_size(X, y, best_k)
