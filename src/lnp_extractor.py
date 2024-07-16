import torch

class LNPExtractor:
    def __init__(self, denoising_network):
        self.denoising_network = denoising_network

    def extract_lnp(self, image):
        with torch.no_grad():
            denoised = self.denoising_network(image)
        lnp = image - denoised
        lnp = (lnp - lnp.mean()) / (lnp.std() + 1e-8)
        lnp = torch.clamp(lnp, -1, 1)
        
        return lnp