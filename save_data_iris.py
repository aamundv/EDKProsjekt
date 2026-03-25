#!/usr/bin/env python3
"""Translate iris_files/train_test_sg5.m to Python and persist pandas pickle files.

Outputs:
- iris_full.pkl: Full labeled dataset using selected feature columns.
- iris_folds.pkl: Per-fold split data (design/test) for all classes.
- iris_results.pkl: Per-fold confusion counts and accuracies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_feature_cols(raw: str) -> list[int]:
	cols = [int(x.strip()) for x in raw.split(",") if x.strip()]
	if not cols:
		raise ValueError("At least one feature column must be provided")
	if any(c < 1 for c in cols):
		raise ValueError("Feature columns are 1-indexed and must be >= 1")
	return cols


def _load_classes(data_dir: Path) -> dict[int, np.ndarray]:
	return {
		1: np.loadtxt(data_dir / "class_1", delimiter=","),
		2: np.loadtxt(data_dir / "class_2", delimiter=","),
		3: np.loadtxt(data_dir / "class_3", delimiter=","),
	}


def _cyclic_inclusive_indices(start_1idx: int, end_1idx: int, n_total: int) -> np.ndarray:
	if start_1idx < 1 or start_1idx > n_total:
		raise ValueError("start index out of range")
	if end_1idx < 1 or end_1idx > n_total:
		raise ValueError("end index out of range")

	if end_1idx < start_1idx:
		left = np.arange(start_1idx - 1, n_total)
		right = np.arange(0, end_1idx)
		return np.concatenate([left, right])

	return np.arange(start_1idx - 1, end_1idx)


def _normal_pdf_1d(x: np.ndarray, mean: float, std: float) -> np.ndarray:
	std_safe = max(float(std), 1e-12)
	z = (x - mean) / std_safe
	return np.exp(-0.5 * z * z) / (std_safe * np.sqrt(2.0 * np.pi))


def _predict_classes(
	x_samples: np.ndarray,
	means: np.ndarray,
	stds: np.ndarray,
) -> np.ndarray:
	# Product of per-feature independent normal likelihoods (Naive Bayes style).
	probs = np.ones((x_samples.shape[0], means.shape[0]), dtype=np.float64)
	for class_idx in range(means.shape[0]):
		for feat_idx in range(x_samples.shape[1]):
			probs[:, class_idx] *= _normal_pdf_1d(
				x_samples[:, feat_idx],
				mean=float(means[class_idx, feat_idx]),
				std=float(stds[class_idx, feat_idx]),
			)
	return np.argmax(probs, axis=1) + 1


def _confusion(true_labels: np.ndarray, pred_labels: np.ndarray, n_classes: int = 3) -> np.ndarray:
	conf = np.zeros((n_classes, n_classes), dtype=np.int64)
	for i in range(1, n_classes + 1):
		mask = true_labels == i
		for j in range(1, n_classes + 1):
			conf[i - 1, j - 1] = int(np.sum(pred_labels[mask] == j))
	return conf


def convert_iris(data_dir: Path, output_dir: Path, feature_cols_1idx: list[int]) -> None:
	classes = _load_classes(data_dir)
	n_total = classes[1].shape[0]
	if any(arr.shape[0] != n_total for arr in classes.values()):
		raise ValueError("All classes must have the same number of samples")

	feature_idx0 = [c - 1 for c in feature_cols_1idx]
	feature_names = [f"feature_{c}" for c in feature_cols_1idx]

	selected = {
		cls: np.atleast_2d(arr[:, feature_idx0]) if len(feature_idx0) > 1 else arr[:, feature_idx0].reshape(-1, 1)
		for cls, arr in classes.items()
	}

	full_frames: list[pd.DataFrame] = []
	for cls in [1, 2, 3]:
		df_cls = pd.DataFrame(selected[cls], columns=feature_names)
		df_cls.insert(0, "label", cls)
		df_cls.insert(1, "sample_idx", np.arange(1, n_total + 1))
		full_frames.append(df_cls)

	full_df = pd.concat(full_frames, ignore_index=True)

	fold_frames: list[pd.DataFrame] = []
	result_rows: list[dict[str, int | float]] = []

	for k in range(1, 6):
		n1d = (k - 1) * 10 + 1
		n2d = ((k + 2) * 10 - 1) % 50 + 1
		n1t = (n2d + 1) % 50
		if n1t == 0:
			n1t = 50
		n2t = (n2d + 19) % 50 + 1

		idx_d = _cyclic_inclusive_indices(n1d, n2d, n_total)
		idx_t = _cyclic_inclusive_indices(n1t, n2t, n_total)

		x_design = {cls: selected[cls][idx_d] for cls in [1, 2, 3]}
		x_test = {cls: selected[cls][idx_t] for cls in [1, 2, 3]}

		means = np.vstack([x_design[cls].mean(axis=0) for cls in [1, 2, 3]])
		stds = np.vstack([x_design[cls].std(axis=0, ddof=1) for cls in [1, 2, 3]])

		for split_name, split_data, split_idx in [
			("design", x_design, idx_d),
			("test", x_test, idx_t),
		]:
			true_labels = []
			pred_labels = []

			for cls in [1, 2, 3]:
				x_cls = split_data[cls]
				pred_cls = _predict_classes(x_cls, means, stds)

				true_labels.append(np.full(x_cls.shape[0], cls, dtype=np.int64))
				pred_labels.append(pred_cls)

				df_cls = pd.DataFrame(x_cls, columns=feature_names)
				df_cls.insert(0, "label", cls)
				df_cls.insert(1, "pred", pred_cls)
				df_cls.insert(2, "split", split_name)
				df_cls.insert(3, "fold", k)
				df_cls.insert(4, "sample_idx", split_idx + 1)
				fold_frames.append(df_cls)

			true_arr = np.concatenate(true_labels)
			pred_arr = np.concatenate(pred_labels)
			conf = _confusion(true_arr, pred_arr, n_classes=3)
			acc = float((true_arr == pred_arr).mean())

			for true_cls in range(1, 4):
				for pred_cls in range(1, 4):
					result_rows.append(
						{
							"fold": k,
							"split": split_name,
							"true_class": true_cls,
							"pred_class": pred_cls,
							"count": int(conf[true_cls - 1, pred_cls - 1]),
							"accuracy": acc,
						}
					)

	folds_df = pd.concat(fold_frames, ignore_index=True)
	results_df = pd.DataFrame(result_rows)

	output_dir.mkdir(parents=True, exist_ok=True)
	full_path = output_dir / "iris_full.pkl"
	folds_path = output_dir / "iris_folds.pkl"
	results_path = output_dir / "iris_results.pkl"

	full_df.to_pickle(full_path)
	folds_df.to_pickle(folds_path)
	results_df.to_pickle(results_path)

	print(f"Saved full data:    {full_path}")
	print(f"Saved fold data:    {folds_path}")
	print(f"Saved fold results: {results_path}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Translate train_test_sg5.m and save iris data/results as pandas pickle files."
	)
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=Path("iris_files"),
		help="Directory containing class_1, class_2 and class_3.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("iris_files"),
		help="Directory where pickle files are written.",
	)
	parser.add_argument(
		"--feature-cols",
		type=str,
		default="4",
		help="Comma-separated 1-indexed feature columns to use (default matches MATLAB script: 4).",
	)
	args = parser.parse_args()

	feature_cols = _parse_feature_cols(args.feature_cols)
	convert_iris(args.data_dir, args.output_dir, feature_cols)


if __name__ == "__main__":
	main()
