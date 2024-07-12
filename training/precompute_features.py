import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import pickle
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(project_root, 'denoising_network.pth')

sys.path.insert(0, project_root)

from src.denoising_network import DenoisingNetwork
from src.lnp_extractor import LNPExtractor
from src.feature_extractor import FeatureExtractor

IMAGE_SIZE = (256, 256)
REAL_IMAGES_PATH = 'data/real_images/'
FEATURES_SAVE_PATH = 'data/models/precomputed_features.pkl'

REAL_IMAGES_PATH = os.path.join(project_root, 'data', 'real_images')
FEATURES_SAVE_PATH = os.path.join(project_root, 'data', 'precomputed_features.pkl')
  
def load_image(path):
    try:
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        with Image.open(path) as img:
            return transform(img.convert('RGB'))
    except Exception as e:
        print(f"Error processing {path}: {str(e)}")
        return None

def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            img = load_image(img_path)
            if img is not None:
                images.append(img)
    return images

def process_images(images, lnp_extractor, feature_extractor, device):
    features = []
    for img in tqdm(images, desc="Processing images"):
        img_tensor = img.unsqueeze(0).to(device)
        lnp = lnp_extractor.extract_lnp(img_tensor)
        feature = feature_extractor.extract_features(lnp.squeeze(0))
        features.append(feature.cpu().numpy())
    return features

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    denoising_network = DenoisingNetwork().to(device)
    denoising_network.load_state_dict(torch.load(model_path, map_location=device))
    denoising_network.eval()
    
    lnp_extractor = LNPExtractor(denoising_network)
    feature_extractor = FeatureExtractor()

    real_images = load_images_from_directory(REAL_IMAGES_PATH)
    print(f"Number of real images loaded: {len(real_images)}")

    if not real_images:
        print("No valid images found for processing.")
        return

    real_features = process_images(real_images, lnp_extractor, feature_extractor, device)

    # Save the precomputed features
    with open(FEATURES_SAVE_PATH, 'wb') as f:
        pickle.dump(real_features, f)

    print(f"Precomputed features saved to {FEATURES_SAVE_PATH}")

if __name__ == "__main__":
    main()