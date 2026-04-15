#!/usr/bin/env python3
# Purpose: Convert MNIST binary image/label files into pandas pickle datasets and metadata.

# Disclaimer: This script was created by asking github copilot to translate the original MATLAB read09.m script into Python
# It was manually validated afterwards to ensure the output was correct.

"""Convert MNIST binary files into pandas-friendly serialized files.

This script mirrors the behavior of MNIST_files/read09.m, but uses
vectorized NumPy reads and pandas output files for fast reload.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import pandas as pd


def _read_idx_images(path: Path) -> tuple[np.ndarray, int, int]:
	with path.open("rb") as f:
		magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
		if magic != 2051:
			raise ValueError(f"Unexpected image magic number in {path}: {magic}")

		data = np.fromfile(f, dtype=np.uint8)

	expected = num_images * rows * cols
	if data.size != expected:
		raise ValueError(
			f"Image file size mismatch in {path}: expected {expected} values, got {data.size}"
		)

	return data.reshape(num_images, rows * cols), rows, cols


def _read_idx_labels(path: Path) -> np.ndarray:
	with path.open("rb") as f:
		magic, num_labels = struct.unpack(">II", f.read(8))
		if magic != 2049:
			raise ValueError(f"Unexpected label magic number in {path}: {magic}")

		labels = np.fromfile(f, dtype=np.uint8)

	if labels.size != num_labels:
		raise ValueError(
			f"Label file size mismatch in {path}: expected {num_labels} values, got {labels.size}"
		)

	return labels


def convert_mnist(input_dir: Path, output_dir: Path) -> None:
	train_images, rows, cols = _read_idx_images(input_dir / "train_images.bin")
	test_images, test_rows, test_cols = _read_idx_images(input_dir / "test_images.bin")
	train_labels = _read_idx_labels(input_dir / "train_labels.bin")
	test_labels = _read_idx_labels(input_dir / "test_labels.bin")

	if rows != test_rows or cols != test_cols:
		raise ValueError(
			"Train and test image dimensions differ: "
			f"train=({rows}, {cols}), test=({test_rows}, {test_cols})"
		)

	if train_images.shape[0] != train_labels.shape[0]:
		raise ValueError("Number of train images and train labels does not match")
	if test_images.shape[0] != test_labels.shape[0]:
		raise ValueError("Number of test images and test labels does not match")

	pixel_columns = [f"px_{i}" for i in range(rows * cols)]
	train_df = pd.DataFrame(train_images, columns=pixel_columns)
	test_df = pd.DataFrame(test_images, columns=pixel_columns)

	train_df.insert(0, "label", train_labels.astype(np.uint8))
	test_df.insert(0, "label", test_labels.astype(np.uint8))

	output_dir.mkdir(parents=True, exist_ok=True)
	train_path = output_dir / "mnist_train.pkl"
	test_path = output_dir / "mnist_test.pkl"
	meta_path = output_dir / "mnist_meta.pkl"

	train_df.to_pickle(train_path)
	test_df.to_pickle(test_path)

	meta = pd.Series(
		{
			"num_train": int(train_df.shape[0]),
			"num_test": int(test_df.shape[0]),
			"row_size": int(rows),
			"col_size": int(cols),
			"vec_size": int(rows * cols),
		}
	)
	meta.to_pickle(meta_path)

	print(f"Saved train data: {train_path}")
	print(f"Saved test data:  {test_path}")
	print(f"Saved metadata:   {meta_path}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Convert MNIST binary files to pandas pickle files."
	)
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=Path("MNIST_files"),
		help="Directory containing train/test *_images.bin and *_labels.bin files.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("MNIST_files"),
		help="Directory where pandas .pkl files are written.",
	)
	args = parser.parse_args()

	convert_mnist(args.input_dir, args.output_dir)


if __name__ == "__main__":
	main()
