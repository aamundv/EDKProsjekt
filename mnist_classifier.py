import pandas as pd
import numpy as np
from scipy.spatial import distance
import time

CHUNK_SIZE = 1000

test_data = pd.read_pickle("MNIST_files/mnist_test.pkl")
train_data = pd.read_pickle("MNIST_files/mnist_train.pkl")

print(test_data.head())

def nearest_neighbor_chunked(train_features, train_labels, test_chunk, chunk_size=CHUNK_SIZE):
    # test_chunk shape: (num_test_samples, num_features)
    best_distances = np.full(test_chunk.shape[0], np.inf)
    best_labels = np.empty(test_chunk.shape[0], dtype=train_labels.dtype)

    for start in range(0, len(train_features), chunk_size):
        end = start + chunk_size
        train_chunk = train_features[start:end]
        label_chunk = train_labels[start:end]

        # cdist input shapes are (num_samples, num_features) for both arrays.
        distances = distance.cdist(train_chunk, test_chunk, metric="euclidean")
        best_local_idx = np.argmin(distances, axis=0)
        best_local_distance = distances[best_local_idx, np.arange(test_chunk.shape[0])]

        improved = best_local_distance < best_distances
        best_distances[improved] = best_local_distance[improved]
        best_labels[improved] = label_chunk[best_local_idx[improved]]

    return best_labels


train_labels = train_data["label"].to_numpy()
train_features = train_data.drop(columns=["label"]).to_numpy(dtype=np.float32)
test_labels = test_data["label"].to_numpy()
test_features = test_data.drop(columns=["label"]).to_numpy(dtype=np.float32)

time_start = time.time()

classified_results = pd.DataFrame(columns=["true_label", "predicted_label", "correctly_classified"])
for test_start in range(0, len(test_features), CHUNK_SIZE):
    test_end = test_start + CHUNK_SIZE
    test_chunk = test_features[test_start:test_end]
    label_chunk = test_labels[test_start:test_end]

    predicted_chunk = nearest_neighbor_chunked(train_features, train_labels, test_chunk)
    current_results = pd.DataFrame(
        {"true_label": label_chunk, "predicted_label": predicted_chunk, "correctly_classified": label_chunk == predicted_chunk}
    )
    classified_results = pd.concat([classified_results, current_results], ignore_index=True)

    print(f"Processed {min(test_end, len(test_features))}/{len(test_features)} test images")
    print(f"Current time taken: {time.time() - time_start} seconds")

print(f"Total time taken: {time.time() - time_start} seconds")

classified_results.to_pickle("mnist_classified_results.pkl")