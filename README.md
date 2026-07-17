# Brain Tumor Classification Using Deep Learning (BraTS2021)

This repository contains the implementation of a deep learning-based binary brain tumor classification framework developed using the BraTS2021 dataset.

## Overview

The framework classifies brain MRI scans into two categories:

- High-Grade Glioma (HGG)
- Low-Grade Glioma (LGG)

The model achieved the following performance:

- Accuracy: 89.05%
- Precision: 96.10%
- Recall: 81.40%
- F1-score: 88.14%
- ROC-AUC: 0.9640

## Repository Structure

```text
brats2021-classification/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── src/
    ├── config.py
    ├── dataset.py
    ├── evaluate.py
    ├── losses.py
    ├── metrics.py
    ├── model.py
    ├── preprocessing.py
    ├── train.py
    ├── utils.py
    └── visualize.py
```

## Dataset

This project uses the BraTS2021 dataset. The dataset is not included in this repository because of its size and distribution conditions.

## Installation

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Results

The final confusion matrix was:

```text
[[967, 33],
 [186, 814]]
```

## Citation

Please cite the associated journal article when using this repository.

## License

This project is distributed under the MIT License.