
<div style="text-align: center;">
  <img src="https://cdn.linearfox.com/assets/img/logo/512/whiteblack.png" alt="DePixel Logo" width="100"/>
</div>

# DePixel - Image Authenticity Detection
Distinguish between real and AI-generated images.

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

## Usage
1. Prepare your dataset:
- Place real images in the `data/real_images/` directory
- Place test images (real or AI-generated) in the `data/test_images/` directory
2. Run the main script.

## Customization
- Adjust `IMAGE_SIZE` in `main.py` to change the input image size
- Modify `PLOT_FEATURES` in `main.py` to change the number of features displayed in distribution plots
- Fine-tune classifier parameters in `src/classifier.py` for better performance
- Modify the config.yaml file to adjust settings.

## Training
To train the denoising network:
```python scripts/train.py```

This will train the model using the real images and save the trained model as denoising_network.pth.

## Output

- Console output with processing details and classification results
- Feature distribution plots saved as PNG files in the project directory

<table>
  <tr>
    <td><img src="data/results/dog1test.png" alt="Dog1 Test Image" width="600"/></td>
    <td><img src="data/results/dog2test.png" alt="Dog2 Test Image" width="600"/></td>
  </tr>
  <tr>
    <td>Real Image Dog1.jpg</td>
    <td>Ai-Generated Image Dog2.jpg</td>    
  </tr>
</table>


## Acknowledgments
- DePixel based on: Xiuli Bi, Bo Liu, et al. for their research paper "Detecting Generated Images by Real Images Only" https://arxiv.org/abs/2311.00962

- DePixel Authors: Nathan Fargo at Linear Fox Labs.