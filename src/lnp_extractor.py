import torch

class LNPExtractor:
    def __init__(self, denoising_network):
        self.denoising_network = denoising_network

    def extract_lnp(self, image):
        with torch.no_grad():
            denoised = self.denoising_network(image)
        return image - denoised