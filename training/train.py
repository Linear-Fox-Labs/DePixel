import os
import sys  
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
from src.denoising_network import DenoisingNetwork
import yaml
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

class SingleFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, 0  # 0 is a dummy label

def print_directory_contents(directory):
    print(f"Contents of {directory}:")
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize(tuple(config['training']['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
    ])
    
    data_dir = os.path.join(project_root, config['training']['data_dir'])
    print(f"Attempting to access data directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist.")
        print("Current working directory:", os.getcwd())
        print_directory_contents(os.getcwd())
        return

    print_directory_contents(data_dir)

    # Try to use ImageFolder, if it fails, use SingleFolderDataset
    try:
        dataset = datasets.ImageFolder(data_dir, transform=transform)
    except FileNotFoundError:
        print("No class folders found. Using all images in the directory.")
        dataset = SingleFolderDataset(data_dir, transform=transform)

    if len(dataset) == 0:
        print("Error: No images found in the specified directory.")
        return

    print(f"Number of images found: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    denoising_network = DenoisingNetwork().to(device)
    optimizer = torch.optim.Adam(denoising_network.parameters(), lr=config['training']['learning_rate'])
    criterion = torch.nn.MSELoss()
    
    for epoch in range(config['training']['num_epochs']):
        denoising_network.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        for batch in progress_bar:
            images, _ = batch
            images = images.to(device)
            
            optimizer.zero_grad()
            denoised = denoising_network(images)
            loss = criterion(denoised, images)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}, Average Loss: {avg_loss:.4f}")
    
    # Save the trained model
    model_path = os.path.join(project_root, config['model']['denoising_network_path'])
    torch.save(denoising_network.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    config_path = os.path.join(project_root, 'config.yaml')
    config = load_config(config_path)
    train(config)