import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# Trains and evaluate an Iris linear classifier with two different train/test splits
# Also has code to perform a grid search using test accuracy to select the best learning rate


header = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"]

iris_df = pd.read_csv("iris_files/iris.data", names=header, header=None)

training_df_1 = iris_df.groupby("class", group_keys=False).head(30)
testing_df_1 = iris_df.groupby("class", group_keys=False).tail(20)
data_split_1 = {"training": training_df_1, "testing": testing_df_1}

training_df_2 = iris_df.groupby("class", group_keys=False).tail(30)
testing_df_2 = iris_df.groupby("class", group_keys=False).head(20)
data_split_2 = {"training": training_df_2, "testing": testing_df_2}

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def gradient_MSE(samples, weights):
    gradient = 0
    error_sum = 0
    for _, row in samples.iterrows():
        sepal_length = row["sepal length in cm"]
        sepal_width = row["sepal width in cm"]
        petal_length = row["petal length in cm"]
        petal_width = row["petal width in cm"]
        feature_vector = [sepal_length, sepal_width, petal_length, petal_width, 1]

        g_k = sigmoid(weights @ feature_vector)

        true_label_k = [1 if row["class"] == "Iris-setosa" else 0,
                    1 if row["class"] == "Iris-versicolor" else 0,
                    1 if row["class"] == "Iris-virginica" else 0]
        
        error = np.array(g_k - true_label_k) * np.array(g_k) * np.array(1 - g_k)
        error_sum += (g_k - true_label_k) @ (g_k - true_label_k)
        gradient += np.outer(error, feature_vector)
    
    return gradient, error_sum / len(samples)

def train(samples, learning_rate, iterations):
    weights = np.zeros((3, 5))
    for _ in range(iterations):
        gradient, MSE = gradient_MSE(samples, weights)
        weights -= learning_rate * gradient
        if MSE < 1e-3:
            break
    return weights

def classify(sample, weights):
    sepal_length = sample["sepal length in cm"]
    sepal_width = sample["sepal width in cm"]
    petal_length = sample["petal length in cm"]
    petal_width = sample["petal width in cm"]
    feature_vector = [sepal_length, sepal_width, petal_length, petal_width, 1]

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
    return correct_predictions / len(samples)

def learning_rate_grid_search(train_samples, test_samples, train_iterations=200, num_rates=12):
    learning_rates = np.logspace(-4, 0, num=num_rates)
    accuracies = []

    for learning_rate in learning_rates:
        weights = train(train_samples, learning_rate, train_iterations)
        accuracy = evaluate_accuracy(test_samples, weights)
        accuracies.append(accuracy)

    return learning_rates, np.array(accuracies)

def perform_grid_search(data_split=data_split_1):
    learning_rates, accuracies = learning_rate_grid_search(data_split["training"], data_split["testing"], train_iterations=1000, num_rates=50)

    fig = go.Figure(
        data=[
            go.Scatter(
                x=learning_rates,
                y=accuracies,
                mode="lines+markers",
                name="Accuracy"
            )
        ]
    )

    fig.update_layout(
        title="Learning Rate vs Accuracy",
        xaxis_title="Learning Rate",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1]),
        template="plotly_white"
    )
    fig.update_xaxes(type="log")
    fig.write_html("learning_rate_vs_accuracy.html")

    best_idx = int(np.argmax(accuracies))
    print(f"Best learning rate: {learning_rates[best_idx]:.6f}, Accuracy: {accuracies[best_idx]:.4f}")
    return learning_rates[best_idx]

def test_classification(learning_rate, data_split):
    weights = train(data_split["training"], learning_rate, 1000)
    for _, row in data_split["testing"].iterrows():
        predicted_class = classify(row, weights)
        print(f"Predicted: {predicted_class}, Actual: {row['class']}")
    accuracy = evaluate_accuracy(data_split["testing"], weights)
    print(f"Accuracy with learning rate {learning_rate:.6f}: {accuracy:.4f}")
    confusion = confusion_matrix(data_split["testing"]["class"], [classify(row, weights) for _, row in data_split["testing"].iterrows()])
    print("Confusion Matrix:")
    print(confusion)
    

#best_learning_rate = perform_grid_search()
best_learning_rate = 0.008

test_classification(best_learning_rate, data_split_1)

test_classification(best_learning_rate, data_split_2)