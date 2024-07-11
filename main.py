import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from src.denoising_network import DenoisingNetwork
from src.lnp_extractor import LNPExtractor
from src.feature_extractor import FeatureExtractor
from src.classifier import Classifier
 
IMAGE_SIZE = (256, 256)
PLOT_FEATURES = 5
REAL_IMAGES_PATH = 'data/real_images/'
TEST_IMAGES_PATH = 'data/test_images/'

def load_image(path):
    """Load and preprocess a single image."""
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
    """Load all images from a directory."""
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            img = load_image(img_path)
            if img is not None:
                images.append(img)
    return images

def plot_feature_distributions(real_features, test_features, filename):
    """Plot feature distributions for real and test images."""
    real_features = np.array(real_features)
    test_features = np.array(test_features)
    
    if test_features.ndim == 1:
        test_features = test_features.reshape(1, -1)
    
    plt.figure(figsize=(15, 5))
    for i in range(min(PLOT_FEATURES, real_features.shape[1])):
        plt.subplot(1, PLOT_FEATURES, i+1)
        plt.hist(real_features[:, i], bins=20, alpha=0.5, label='Real')
        plt.axvline(test_features[0, i], color='r', linestyle='dashed', linewidth=2, label='Test')
        plt.title(f'Feature {i+1}')
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def process_images(images, lnp_extractor, feature_extractor, device, image_type="real"):
    """Process a batch of images and extract features."""
    features = []
    for i, img in enumerate(images):
        img_tensor = img.unsqueeze(0).to(device)
        lnp = lnp_extractor.extract_lnp(img_tensor)
        feature = feature_extractor.extract_features(lnp.squeeze(0))
        features.append(feature.cpu().numpy())
        
        print(f"Processed {image_type} image {i+1}/{len(images)}")
        print(f"  LNP shape: {lnp.shape}")
        print(f"  Features shape: {feature.shape}")
        print(f"  Features mean: {feature.mean().item():.4f}")
        print(f"  Features std: {feature.std().item():.4f}")
    
    return features

def compare_features(real_features, test_features):
    """Compare test features with real features."""
    real_features_array = np.array(real_features)
    real_mean = real_features_array.mean(axis=0)
    real_std = real_features_array.std(axis=0)
    test_features_np = test_features.cpu().numpy()
    
    print("\nFeature comparison:")
    print(f"  Mean difference: {np.mean(np.abs(real_mean - test_features_np)):.4f}")
    print(f"  Std difference: {np.mean(np.abs(real_std - np.std(test_features_np))):.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    denoising_network = DenoisingNetwork().to(device)
    lnp_extractor = LNPExtractor(denoising_network)
    feature_extractor = FeatureExtractor()
    classifier = Classifier()

    real_images = load_images_from_directory(REAL_IMAGES_PATH)
    print(f"Number of real images loaded: {len(real_images)}")
    if not real_images:
        print("No valid images found for training.")
        return

    real_features = process_images(real_images, lnp_extractor, feature_extractor, device)

    classifier.train(real_features)
    print("Classifier trained")

    test_images = load_images_from_directory(TEST_IMAGES_PATH)
    
    for i, test_img in enumerate(test_images):
        test_img_tensor = test_img.unsqueeze(0).to(device)
        test_lnp = lnp_extractor.extract_lnp(test_img_tensor)
        test_features = feature_extractor.extract_features(test_lnp.squeeze(0))
        
        print(f"\nTest image {i+1}:")
        print(f"  LNP shape: {test_lnp.shape}")
        print(f"  Features shape: {test_features.shape}")
        print(f"  Features mean: {test_features.mean().item():.4f}")
        print(f"  Features std: {test_features.std().item():.4f}")
        
        result = classifier.predict([test_features.cpu().numpy()])
        if result[0]:
            print("Image is likely real")
        else:
            print("Image is likely generated")
        
        plot_feature_distributions(real_features, test_features.cpu().numpy(), f'feature_distributions_test_{i+1}.png')
        compare_features(real_features, test_features)
 
        real_features_array = np.array(real_features)
        test_features_np = test_features.cpu().numpy()
        feature_diff = np.abs(np.mean(real_features_array, axis=0) - test_features_np[0])
        top_diff_indices = np.argsort(feature_diff)[-5:][::-1]
        print("\nTop 5 most different features:")
        for idx in top_diff_indices:
            print(f"Feature {idx}: diff = {feature_diff[idx]:.4f}")

if __name__ == "__main__":
    main()