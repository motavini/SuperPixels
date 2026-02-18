# SLIC Superpixels Implementation

This repository contains a custom implementation of the SLIC (Simple Linear Iterative Clustering) algorithm for superpixel generation, along with comparisons to other methods (OpenCV SLIC, SEEDS, Graph Segmentation).

This project was developed by Felipe Brito and Vinicius da Mata e Mota for the Computer Vision course (Reconnaissance Visuelle).

## Features

- **Custom SLIC Implementation**: A Python implementation of SLIC from scratch.
- **Comparison Benchmarks**: Compare the custom implementation against:
  - OpenCV's SLIC
  - SEEDS
  - Graph Based Segmentation
- **Evaluation Metrics**:
  - Boundary Recall
  - Undersegmentation Error
  - Achievable Segmentation Accuracy (ASA)
  - Runtime performance
- **Experiment Framework**: Scripts to run experiments on BSDS500 or local images and generate plots.

## Project Structure

- `custom_SLIC.py`: Core implementation of the SLIC algorithm.
- `experiments.py`: Main script to run comparisons and generate performance plots.
- `metrics.py`: Implementation of evaluation metrics (BR, UE, ASA).
- `open_cv_SLIC.py`: Wrappers for OpenCV's superpixel algorithms.
- `dataloader.py`: Utilities for loading images and ground truth (BSDS500).

## Requirements

- Python 3.x
- OpenCV (`opencv-contrib-python` for `ximgproc` module)
- NumPy
- Matplotlib

## Usage

### Run SLIC Demo
To run a simple demonstration of the custom SLIC algorithm on a single image:
```bash
python custom_SLIC.py
```
This will process a local image and display the result.

### Run Experiments
To run the full suite of experiments and comparisons:
```bash
python experiments.py
```
This script will:
1. Look for BSDS500 dataset or use local images.
2. Apply multiple superpixel algorithms.
3. Calculate metrics.
4. Generate plot images in the current directory (e.g., `plot_br_*.png`, `plot_runtime_*.png`).

## References

- Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P., & Süsstrunk, S. (2012). SLIC superpixels compared to state-of-the-art superpixel methods. IEEE transactions on pattern analysis and machine intelligence.
