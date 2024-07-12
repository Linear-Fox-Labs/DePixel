import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score
from torchvision import transforms
from PIL import Image
from src.denoising_network import DenoisingNetwork
from src.lnp_extractor import LNPExtractor
from src.feature_extractor import FeatureExtractor
from src.classifier import Classifier
import yaml
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PLOT_FEATURES = 2

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_images(image_paths, transform):
    images = []
    for path in tqdm(image_paths, desc="Loading images"):
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = transform(img)
            images.append(img_tensor)
        except Exception as e:
            logging.error(f"Error loading image {path}: {str(e)}")
    return torch.stack(images)

def extract_features(images, lnp_extractor, feature_extractor, device):
    features = []
    for img in tqdm(images, desc="Extracting features"):
        img_tensor = img.unsqueeze(0).to(device)
        lnp = lnp_extractor.extract_lnp(img_tensor)
        feature = feature_extractor.extract_features(lnp.squeeze(0))
        features.append(feature.cpu().numpy())
    return np.array(features)

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10,7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

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

def compare_features(real_features, fake_features):
    real_mean = np.mean(real_features, axis=0)
    real_std = np.std(real_features, axis=0)
    fake_mean = np.mean(fake_features, axis=0)
    fake_std = np.std(fake_features, axis=0)
    
    logging.info("\nFeature comparison:")
    logging.info(f"  Mean difference: {np.mean(np.abs(real_mean - fake_mean)):.4f}")
    logging.info(f"  Std difference: {np.mean(np.abs(real_std - fake_std)):.4f}")
     
    num_features = min(5, real_features.shape[1])
    plt.figure(figsize=(15, 5))
    for i in range(num_features):
        plt.subplot(1, num_features, i+1)
        plt.hist(real_features[:, i], bins=20, alpha=0.5, label='Real')
        plt.hist(fake_features[:, i], bins=20, alpha=0.5, label='Fake')
        plt.title(f'Feature {i+1}')
        if i == 0:
            plt.legend()
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()

def evaluate(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
 
    denoising_network = DenoisingNetwork().to(device)
    denoising_network.load(config['model']['denoising_network_path'])
    lnp_extractor = LNPExtractor(denoising_network)
    feature_extractor = FeatureExtractor()
    classifier = Classifier()
 
    transform = transforms.Compose([
        transforms.Resize(tuple(config['inference']['image_size'])),
        transforms.ToTensor(),
    ])

    real_images = load_images(config['evaluation']['real_images'], transform)
    fake_images = load_images(config['evaluation']['fake_images'], transform)

    real_features = extract_features(real_images, lnp_extractor, feature_extractor, device)
    fake_features = extract_features(fake_images, lnp_extractor, feature_extractor, device)
 
    compare_features(real_features, fake_features)
 
    y_true = np.concatenate([np.ones(len(real_features)), np.zeros(len(fake_features))])
    X = np.concatenate([real_features, fake_features])
 
    classifier.train(real_features)
 
    y_pred = classifier.predict(X)
    y_pred_proba = classifier.predict_proba(X)[:, 1]
 
    accuracy = accuracy_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_true, y_pred)

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}")

    plot_roc_curve(fpr, tpr, roc_auc)
    plot_confusion_matrix(cm, ['Fake', 'Real'])

    logging.info("Evaluation complete. ROC curve, confusion matrix, and feature distributions saved.")

if __name__ == "__main__":
    config = load_config('configs/config.yaml')
    evaluate(config)