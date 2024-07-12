import torch
from torchvision import transforms
from PIL import Image
import os
import logging
from tqdm import tqdm
from src.denoising_network import DenoisingNetwork
# import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

IMAGE_SIZE = (256, 256) # Size of the images to be loaded
TEST_IMAGES_PATH = 'data/test_images/'
MODEL_PATH = 'denoising_network.pth'

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
        logging.error(f"Error processing {path}: {str(e)}")
        return None

def load_images_from_directory(directory):
    """Load all images from a directory."""
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            img = load_image(img_path)
            if img is not None:
                label = 1 if filename.startswith('real') else 0
                images.append((img, filename, label))
    return images

def process_output(output):
    """Process the model output to determine if the image is real or generated."""
    mean_output = output.mean().item()
    return 1 if mean_output > 0.5 else 0, mean_output


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = DenoisingNetwork().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    logging.info("Model loaded successfully")

    test_images = load_images_from_directory(TEST_IMAGES_PATH)
    logging.info(f"Number of test images loaded: {len(test_images)}")

    with torch.no_grad():
        for img, filename, true_label in tqdm(test_images, desc="Processing images"):
            img_tensor = img.unsqueeze(0).to(device)
            
            output = model(img_tensor)
            prediction, score = process_output(output)
            
            logging.info(f"Image: {filename}, Prediction: {'real' if prediction == 1 else 'generated'}, Score: {score:.4f}, True Label: {'real' if true_label == 1 else 'generated'}")

if __name__ == "__main__":
    main()