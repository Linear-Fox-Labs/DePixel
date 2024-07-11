import torch

class FeatureExtractor:
    def __init__(self, k=32):
        self.k = k

    def extract_features(self, lnp):
        # Ensure lnp is 3D: (channels, height, width)
        if lnp.dim() == 4:  # (batch, channels, height, width)
            lnp = lnp.squeeze(0)
        elif lnp.dim() == 2:  # (height, width)
            lnp = lnp.unsqueeze(0)
        
        fft_lnp = torch.fft.fft2(lnp)
        amplitude_spectrum = torch.abs(fft_lnp)
        enhanced_spectrum = self._enhance_spectrum(amplitude_spectrum)
        features = self._sample_features(enhanced_spectrum)
        features = (features - features.mean()) / (features.std() + 1e-8)

        return features

    def _enhance_spectrum(self, spectrum):
        A_u = torch.mean(spectrum, dim=1, keepdim=True)
        enhanced = torch.where(spectrum < A_u, torch.zeros_like(spectrum), spectrum**2)
        return enhanced

    def _sample_features(self, enhanced_spectrum):
        C, H, W = enhanced_spectrum.shape
        m_indices = torch.arange(0, H, self.k)
        n_indices = torch.arange(0, W, self.k)
        features = enhanced_spectrum[:, m_indices][:, :, n_indices].flatten()
        return features