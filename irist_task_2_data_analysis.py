import pandas as pd

# Loads and prints Iris classification results from task 2.

results = pd.read_pickle("iris_2_results_df.pkl")

for index, row in results.iterrows():
    print(f"Features: {row['Features']}")
    print(f"Training Set: {row['Training Set']}")
    print(f"Test Accuracy: {row['Test Accuracy']:.4f}")
    print("Confusion Matrix:")
    print(row["Confusion Matrix"])
    print("-" * 40)