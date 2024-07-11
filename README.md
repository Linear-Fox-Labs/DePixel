# DePixel
Python-based tool designed to distinguish between real and AI-generated images.

The method uses Learned Noise Patterns (LNP) and one-class classification to map real images to a dense subspace, allowing detection of generated images as outliers. This approach achieves good detection accuracy while using much less training data compared to previous methods.

## Features

- Denoising network for LNP extraction
- Feature extraction from LNP amplitude spectra
- One-class SVM classifier for image authenticity prediction
- Robustness to various image post-processing operations

## Installation

1 + 2. Clone the repository and create a virtual environment (optional but recommended):
```
git clone https://github.com/Linear-Fox-Labs/DePixel
cd DePixel
pip install -r requirements.txt
```

## Acknowledgments

- DePixel based on: Xiuli Bi, Bo Liu, et al. for their research paper "Detecting Generated Images by Real Images Only" https://arxiv.org/abs/2311.00962