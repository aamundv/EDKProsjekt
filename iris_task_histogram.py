import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Plots Iris class histograms per feature and rank features by class-overlap.


header = ["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm", "class"]
iris_df = pd.read_csv("iris_files/iris.data", names=header, header=None)

# Remove any empty/invalid rows and ensure the feature columns are numeric.
iris_df = iris_df.dropna(subset=["class"]).copy()
for feature_name in header[:-1]:
	iris_df[feature_name] = pd.to_numeric(iris_df[feature_name], errors="coerce")
iris_df = iris_df.dropna(subset=header[:-1])

features = header[:-1]
classes = sorted(iris_df["class"].unique())


def distribution_overlap(values_a, values_b, bins=20):
	"""Return normalized histogram overlap between two 1D distributions."""
	combined_min = min(values_a.min(), values_b.min())
	combined_max = max(values_a.max(), values_b.max())

	if combined_min == combined_max:
		return 1.0

	bin_edges = np.linspace(combined_min, combined_max, bins + 1)
	hist_a, _ = np.histogram(values_a, bins=bin_edges)
	hist_b, _ = np.histogram(values_b, bins=bin_edges)

	hist_a = hist_a / hist_a.sum() if hist_a.sum() else hist_a
	hist_b = hist_b / hist_b.sum() if hist_b.sum() else hist_b

	return float(np.minimum(hist_a, hist_b).sum())


feature_overlap_scores = {}
for feature_name in features:
	pair_overlaps = []
	for idx_a in range(len(classes)):
		for idx_b in range(idx_a + 1, len(classes)):
			class_a = classes[idx_a]
			class_b = classes[idx_b]
			values_a = iris_df.loc[iris_df["class"] == class_a, feature_name].to_numpy()
			values_b = iris_df.loc[iris_df["class"] == class_b, feature_name].to_numpy()
			pair_overlaps.append(distribution_overlap(values_a, values_b, bins=20))

	feature_overlap_scores[feature_name] = float(np.mean(pair_overlaps)) if pair_overlaps else 0.0

fig = make_subplots(rows=2, cols=2, subplot_titles=features)

for i, feature_name in enumerate(features):
	row = i // 2 + 1
	col = i % 2 + 1

	for class_name in classes:
		class_values = iris_df.loc[iris_df["class"] == class_name, feature_name]
		fig.add_trace(
			go.Histogram(
				x=class_values,
				name=class_name,
				opacity=0.65,
				showlegend=(i == 0),
			),
			row=row,
			col=col,
		)

	fig.update_xaxes(title_text=feature_name, row=row, col=col)
	fig.update_yaxes(title_text="Count", row=row, col=col)

fig.update_layout(
	title_text="Iris Feature Histograms by Class",
	barmode="overlay",
	bargap=0.1,
	legend_title_text="Class",
	template="plotly_white",
	height=800,
	width=1100,
)

output_file = "iris_feature_class_histograms.html"
fig.write_html(output_file)
print(f"Saved histogram plot to {output_file}")

sorted_features = sorted(feature_overlap_scores.items(), key=lambda item: item[1], reverse=True)
print("\nFeatures ranked by overlap (highest to lowest):")
for rank, (feature_name, overlap_score) in enumerate(sorted_features, start=1):
	print(f"{rank}. {feature_name}: {overlap_score:.3f}")
