from sklearn.cluster import KMeans
from scipy.spatial import distance
import pandas as pd
import numpy as np
import time

# K-means classifier for MNIST, saves the classification results.

CHUNK_SIZE = 1000
M = 64

test_data = pd.read_pickle("MNIST_files/mnist_test.pkl")
train_data = pd.read_pickle("MNIST_files/mnist_train.pkl")


def nearest_template_chunked(template_features, template_labels, test_chunk):
	distances = distance.cdist(template_features, test_chunk, metric="euclidean")
	best_template_idx = np.argmin(distances, axis=0)
	return template_labels[best_template_idx]


train_labels = train_data["label"].to_numpy(dtype=np.int64)
train_features = train_data.drop(columns=["label"]).to_numpy(dtype=np.float32)
test_labels = test_data["label"].to_numpy(dtype=np.int64)
test_features = test_data.drop(columns=["label"]).to_numpy(dtype=np.float32)

time_start = time.time()

kmeans = KMeans(n_clusters=M, random_state=42)
cluster_ids = kmeans.fit_predict(train_features)
templates = kmeans.cluster_centers_.astype(np.float32)

# Assign one class label to each template using majority vote in its cluster.
template_labels = np.zeros(M, dtype=np.int64)
for cluster_idx in range(M):
	cluster_member_labels = train_labels[cluster_ids == cluster_idx]
	if cluster_member_labels.size == 0:
		# Fallback for empty clusters (rare): use most common global class.
		template_labels[cluster_idx] = np.bincount(train_labels).argmax()
	else:
		template_labels[cluster_idx] = np.bincount(cluster_member_labels).argmax()

print(f"Finished clustering and template creation after {time.time() - time_start} seconds. Starting classification of test set...")

time_start = time.time()

result_chunks = []
for test_start in range(0, len(test_features), CHUNK_SIZE):
	test_end = test_start + CHUNK_SIZE
	test_chunk = test_features[test_start:test_end]
	label_chunk = test_labels[test_start:test_end]

	predicted_chunk = nearest_template_chunked(templates, template_labels, test_chunk)
	current_results = pd.DataFrame(
		{
			"true_label": label_chunk,
			"predicted_label": predicted_chunk,
			"correctly_classified": label_chunk == predicted_chunk,
		}
	)
	result_chunks.append(current_results)

	print(f"Processed {min(test_end, len(test_features))}/{len(test_features)} test images")
	print(f"Current time taken: {time.time() - time_start} seconds")

classified_results = pd.concat(result_chunks, ignore_index=True)

print(f"Total time taken: {time.time() - time_start} seconds")
print(f"Accuracy: {classified_results['correctly_classified'].mean():.4f}")

classified_results.to_pickle("mnist_kmeans_classified_results.pkl")