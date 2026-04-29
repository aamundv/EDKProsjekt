import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Compares Iris feature subsets by training the classifier on two train/test split variants.

header = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"]

iris_df = pd.read_csv("iris_files/iris.data", names=header, header=None)

training_df_1 = iris_df.groupby("class", group_keys=False).head(30)
testing_df_1 = iris_df.groupby("class", group_keys=False).tail(20)

training_df_2 = iris_df.groupby("class", group_keys=False).tail(30)
testing_df_2 = iris_df.groupby("class", group_keys=False).head(20)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def gradient_MSE(samples, weights):
    gradient = 0
    error_sum = 0
    for _, row in samples.iterrows():
        feature_vector = [row[col] for col in samples.columns[:-1]] + [1]

        g_k = sigmoid(weights @ feature_vector)

        true_label_k = [1 if row["class"] == "Iris-setosa" else 0,
                    1 if row["class"] == "Iris-versicolor" else 0,
                    1 if row["class"] == "Iris-virginica" else 0]
        
        error = np.array(g_k - true_label_k) * np.array(g_k) * np.array(1 - g_k)
        error_sum += (g_k - true_label_k) @ (g_k - true_label_k)
        gradient += np.outer(error, feature_vector)
    
    return gradient, error_sum / len(samples)


def train(samples, learning_rate, iterations):
    weights = np.zeros((3, len(samples.columns)))
    for _ in range(iterations):
        gradient, MSE = gradient_MSE(samples, weights)
        weights -= learning_rate * gradient
        if MSE < 1e-3:
            break
    return weights


def classify(sample, weights):
    feature_vector = [sample[element] for element in sample.index[:-1]] + [1]

    g_k = sigmoid(weights @ feature_vector)
    predicted_class_index = np.argmax(g_k)

    if predicted_class_index == 0:
        return "Iris-setosa"
    elif predicted_class_index == 1:
        return "Iris-versicolor"
    else:
        return "Iris-virginica"
    

def evaluate_accuracy(samples, weights):
    correct_predictions = 0
    for _, row in samples.iterrows():
        predicted_class = classify(row, weights)
        if predicted_class == row["class"]:
            correct_predictions += 1

    confusion_mat = confusion_matrix(samples["class"], [classify(row, weights) for _, row in samples.iterrows()])

    return correct_predictions / len(samples), confusion_mat


ranked_features = ["sepal width in cm", "sepal length in cm", "petal length in cm", "petal width in cm"]

results_df = pd.DataFrame(columns=["Features", "Training Set", "Training Accuracy", "Confusion Matrix"])

learning_rate = 0.008
features_to_remove = []

for element in ranked_features:
    training_df = training_df_1.drop(columns=features_to_remove)
    testing_df = testing_df_1.drop(columns=features_to_remove)

    weights = train(training_df, learning_rate, iterations=10000)
    test_accuracy, confusion_mat = evaluate_accuracy(testing_df, weights)

    new_results_df = pd.DataFrame({
        "Features": ", ".join(training_df.columns[:-1]),
        "Training Set": "Set 1",
        "Test Accuracy": test_accuracy,
        "Confusion Matrix": [confusion_mat]
    })
    results_df = pd.concat([results_df, new_results_df], ignore_index=True)
    print(f"Completed training and evaluation 1 with features: {', '.join(training_df.columns[:-1])}")
    
    training_df = training_df_2.drop(columns=features_to_remove)
    testing_df = testing_df_2.drop(columns=features_to_remove)

    weights = train(training_df, learning_rate, iterations=10000)
    test_accuracy, confusion_mat = evaluate_accuracy(testing_df, weights)
    
    new_results_df = pd.DataFrame({
        "Features": ", ".join(training_df.columns[:-1]),
        "Training Set": "Set 2",
        "Test Accuracy": test_accuracy,
        "Confusion Matrix": [confusion_mat]
    })
    results_df = pd.concat([results_df, new_results_df], ignore_index=True)
    print(f"Completed training and evaluation 2 with features: {', '.join(training_df.columns[:-1])}")

    features_to_remove.append(element)

results_output_file = "iris_2_results_df.pkl"
results_df.to_pickle(results_output_file)
print(f"Saved results to {results_output_file}")