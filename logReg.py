import numpy as np
from ionosphereData import process_ionosphere_data

class LogisticRegression:
    def __init__(self, learning_rate=0.01, number_of_iterations=100000):
        self.learning_rate = learning_rate
        self.number_of_iterations = number_of_iterations
        self.weights = None
        self.iterations_done = 0

    def fit(self, X, y):
        num_features = X.shape[1]
        self.weights = np.zeros(num_features)

        for i in range(self.number_of_iterations):
            linear_model = np.dot(X, self.weights)
            predictions = 1 / (1 + np.exp(-linear_model))
            errors = y - predictions
            gradient = np.dot(X.T, errors) / y.size
            self.weights += self.learning_rate * gradient

        self.iterations_done = self.number_of_iterations

    def predict(self, X):
        linear_model = np.dot(X, self.weights)
        predictions = 1 / (1 + np.exp(-linear_model))
        class_predictions = [1 if i > 0.5 else 0 for i in predictions]
        return class_predictions

def evaluate_accuracy(y_true, y_pred):
    correct_counter = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct_counter += 1
    accuracy = correct_counter / len(y_true)
    return accuracy

def k_fold_cross_validation(model, X, y, k=5, return_iterations=False):
    fold_size = len(X) // k
    accuracies = []
    iteration_counts = []

    for i in range(k):
        start_index = i * fold_size
        end_index = (i + 1) * fold_size if i != k - 1 else len(X)
        X_train = np.concatenate((X[:start_index], X[end_index:]), axis=0)
        y_train = np.concatenate((y[:start_index], y[end_index:]), axis=0)
        X_test = X[start_index:end_index]
        y_test = y[start_index:end_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = evaluate_accuracy(y_test, y_pred)
        accuracies.append(accuracy)

        if return_iterations:
            iteration_counts.append(model.iterations_done)

    average_accuracy = sum(accuracies) / len(accuracies)

    if return_iterations:
        average_iterations = sum(iteration_counts) / len(iteration_counts)
        return average_accuracy, average_iterations
    else:
        return average_accuracy

if __name__ == "__main__":
    X, y = process_ionosphere_data()
    model = LogisticRegression(learning_rate=0.1, number_of_iterations=3000)
    average_accuracy = k_fold_cross_validation(model, X, y, k=5)
    print("5-Fold Cross-Validation Average Accuracy:", "{:.2%}".format(average_accuracy))
