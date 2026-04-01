import pandas as pd
import numpy as np
from scipy.spatial import distance
import time

CHUNK_SIZE = 1000

test_data = pd.read_pickle("MNIST_files/mnist_test.pkl")
train_data = pd.read_pickle("MNIST_files/mnist_train.pkl")

print(test_data.head())

def k_nearest_neighbor_chunked(train_features, train_labels, test_chunk, k=1,  chunk_size=CHUNK_SIZE):
    # test_chunk shape: (num_test_samples, num_features)
    num_test_samples = test_chunk.shape[0]
    best_distances = np.full((num_test_samples, k), np.inf)
    best_labels = np.zeros((num_test_samples, k), dtype=train_labels.dtype)

    for start in range(0, len(train_features), chunk_size):
        end = start + chunk_size
        train_chunk = train_features[start:end]
        label_chunk = train_labels[start:end]

        # cdist input shapes are (num_samples, num_features) for both arrays.
        distances = distance.cdist(train_chunk, test_chunk, metric="euclidean")
        local_k = min(k, train_chunk.shape[0])

        best_local_idx = np.argpartition(distances, local_k - 1, axis=0)[:local_k, :]
        best_local_distance = np.take_along_axis(distances, best_local_idx, axis=0)
        best_local_labels = label_chunk[best_local_idx]

        # Merge existing best candidates with this chunk's candidates and keep global top-k.
        merged_distances = np.concatenate([best_distances, best_local_distance.T], axis=1)
        merged_labels = np.concatenate([best_labels, best_local_labels.T], axis=1)

        best_global_idx = np.argpartition(merged_distances, k - 1, axis=1)[:, :k]
        best_distances = np.take_along_axis(merged_distances, best_global_idx, axis=1)
        best_labels = np.take_along_axis(merged_labels, best_global_idx, axis=1)

    predicted_labels = np.empty(num_test_samples, dtype=train_labels.dtype)
    for i in range(num_test_samples):
        valid_neighbors = np.isfinite(best_distances[i])
        neighbor_labels = best_labels[i, valid_neighbors].astype(np.int64)
        predicted_labels[i] = np.bincount(neighbor_labels).argmax()

    return predicted_labels


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

    predicted_chunk = k_nearest_neighbor_chunked(train_features, train_labels, test_chunk, k=7)
    current_results = pd.DataFrame(
        {"true_label": label_chunk, "predicted_label": predicted_chunk, "correctly_classified": label_chunk == predicted_chunk}
    )
    classified_results = pd.concat([classified_results, current_results], ignore_index=True)

    print(f"Processed {min(test_end, len(test_features))}/{len(test_features)} test images")
    print(f"Current time taken: {time.time() - time_start} seconds")

print(f"Total time taken: {time.time() - time_start} seconds")

classified_results.to_pickle("mnist_knn_classified_results.pkl")