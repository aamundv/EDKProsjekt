import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Takes in the classification results from one of the MNIST classifiers
# Produces and plots a confusion matrix
# Calculates the overall error rate and prints it 
# Plots some misclassifies and some correctly classified examples with their true and predicted labels.

# classified_results = pd.read_pickle("mnist_nn_classified_results.pkl")
# classified_results = pd.read_pickle("mnist_kmeans_classified_results.pkl")
classified_results = pd.read_pickle("mnist_knn_classified_results.pkl")

test_data = pd.read_pickle("MNIST_files/mnist_test.pkl")

print(classified_results.head())

confusion_mat = ConfusionMatrixDisplay.from_predictions(
    classified_results["true_label"].astype(int),
    classified_results["predicted_label"].astype(int),
)
confusion_mat.figure_.suptitle("Confusion Matrix for MNIST Classification using a KNN Classifier")
plt.show()

error_rate = 1 - classified_results["correctly_classified"].mean()
print(f"Overall error rate: {error_rate:.4f}")

correctly_plotted_counter = 0
misclassified_plotted_counter = 0
for element in classified_results.itertuples():
    if element.correctly_classified and correctly_plotted_counter < 5:
        plt.imshow(test_data.iloc[element.Index].drop("label").to_numpy().reshape(28, 28), cmap="gray")
        plt.title(f"True label: {element.true_label}, Predicted label: {element.predicted_label}")
        plt.show()
        correctly_plotted_counter += 1
    elif not element.correctly_classified and misclassified_plotted_counter < 5:
        plt.imshow(test_data.iloc[element.Index].drop("label").to_numpy().reshape(28, 28), cmap="gray")
        plt.title(f"True label: {element.true_label}, Predicted label: {element.predicted_label}")
        plt.show()
        misclassified_plotted_counter += 1
    if correctly_plotted_counter >= 5 and misclassified_plotted_counter >= 5:
        break
    