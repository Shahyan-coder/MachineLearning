import numpy as np
from ionosphereData import process_ionosphere_data
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            single_pred = self._predict(x)
            y_pred.append(single_pred)
        return np.array(y_pred)

    def _predict(self, x):
        distances = []
        for x_train in self.X_train:
            distance = np.sqrt(np.sum((x - x_train)**2))
            distances.append(distance)

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = []
        for i in k_indices:
            label = self.y_train[i]
            k_nearest_labels.append(label)

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

def evaluate_acc(y, y_hat):
    correct = 0
    for i in range(len(y)):
        if y[i] == y_hat[i]:
            correct += 1
    accuracy = correct / len(y)
    return accuracy

# def cross_validate_for_k(X, y, k, k_folds=5):
#     fold_size = len(X) // k_folds
#     accuracies = []
#     model = KNearestNeighbors(k=k)

#     for i in range(k_folds):
#         start_index = i * fold_size
#         end_index = (i + 1) * fold_size if i != k_folds - 1 else len(X)
#         X_train = np.concatenate([X[:start_index], X[end_index:]])
#         y_train = np.concatenate([y[:start_index], y[end_index:]])
#         X_test = X[start_index:end_index]
#         y_test = y[start_index:end_index]

#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         accuracy = evaluate_acc(y_test, y_pred)
#         accuracies.append(accuracy)

#     avg_accuracy = 0
#     for accuracy in accuracies:
#         avg_accuracy += accuracy
#     avg_accuracy /= len(accuracies)

#     return avg_accuracy

def cross_validate_for_k(X, y, k, k_folds=5):
    if len(X) < k_folds:
        raise ValueError("Not enough samples to perform cross-validation")

    fold_size = max(1, len(X) // k_folds)
    accuracies = []
    model = KNearestNeighbors(k=k)

    for i in range(k_folds):
        start_index = i * fold_size
        end_index = min(len(X), (i + 1) * fold_size)
        X_train = np.concatenate([X[:start_index], X[end_index:]])
        y_train = np.concatenate([y[:start_index], y[end_index:]])
        X_test = X[start_index:end_index]
        y_test = y[start_index:end_index]
        # print(f"Fold {i+1}/{k_folds}")
        # print(f"Start index: {start_index}")
        # print(f"End index: {end_index}")
        # print(f"X_train length: {len(X_train)}")
        # print(f"y_train length: {len(y_train)}")
        # print(f"X_test length: {len(X_test)}")
        # print(f"y_test length: {len(y_test)}")
        if len(y_test) == 0:
            print(f"Warning: y_test is empty for fold {i+1}! Skipping this fold.")
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = evaluate_acc(y_test, y_pred)
        accuracies.append(accuracy)
    if not accuracies:
        raise ValueError("No folds were executed. Dataset might be too small.")

    avg_accuracy = sum(accuracies) / len(accuracies)

    return avg_accuracy

def k_fold_cross_validation(X, y, k_values=[3, 5, 7], k_folds=5):
    fold_size = len(X) // k_folds
    k_best = k_values[0]
    best_accuracy = 0

    for k in k_values:
        accuracies = []
        model = KNearestNeighbors(k=k)
        #print(f"Evaluating for k={k}...")

        for i in range(k_folds):
            start_index = i * fold_size
            end_index = (i + 1) * fold_size if i != k_folds - 1 else len(X)
            X_train = np.concatenate([X[:start_index], X[end_index:]])
            y_train = np.concatenate([y[:start_index], y[end_index:]])
            X_test = X[start_index:end_index]
            y_test = y[start_index:end_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = evaluate_acc(y_test, y_pred)
            accuracies.append(accuracy)
            #print(f"  Fold {i+1}/{k_folds}, Accuracy: {accuracy:.2%}")

        avg_accuracy = sum(accuracies) / len(accuracies)
        #print(f"  Average Accuracy for k={k}: {avg_accuracy:.2%}\n")

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            k_best = k

    return k_best, best_accuracy


if __name__ == "__main__":
    X, y = process_ionosphere_data()
    k_best, best_accuracy = k_fold_cross_validation(X, y, k_values=[3, 5, 7, 9, 11])
    print(f"Best K: {k_best} with Average Accuracy: {best_accuracy:.2%}")
