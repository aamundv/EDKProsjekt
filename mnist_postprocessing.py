import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

classified_results = pd.read_pickle("mnist_clustered_classified_results.pkl")
test_data = pd.read_pickle("MNIST_files/mnist_test.pkl")

print(classified_results.head())

confusion_mat = ConfusionMatrixDisplay.from_predictions(
    classified_results["true_label"].astype(int),
    classified_results["predicted_label"].astype(int),
)
confusion_mat.figure_.suptitle("Confusion Matrix for MNIST Classification")
confusion_mat.plot()
plt.show()

for element in classified_results.itertuples():
    if not element.correctly_classified:
        plt.imshow(test_data.iloc[element.Index].drop("label").to_numpy().reshape(28, 28), cmap="gray")
        plt.title(f"True label: {element.true_label}, Predicted label: {element.predicted_label}")
        plt.show()