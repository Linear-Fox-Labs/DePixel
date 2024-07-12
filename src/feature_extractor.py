import torch
import torch.nn.functional as F

class FeatureExtractor:
    def __init__(self, k=32):
        self.k = k

    def extract_features(self, lnp):
        amplitude_spectrum = torch.abs(torch.fft.fft2(lnp))

        enhanced_spectrum = self._enhance_spectrum(amplitude_spectrum)
        sampled_features = self._sample_features(enhanced_spectrum) 
        gradient_features = self._gradient_features(lnp)
        noise_features = self._noise_features(lnp)
        
        all_features = torch.cat([sampled_features, gradient_features, noise_features])
        
        return (all_features - all_features.mean()) / (all_features.std() + 1e-8)

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

    def _gradient_features(self, lnp):
        dx = lnp[:, :, 1:] - lnp[:, :, :-1]
        dy = lnp[:, 1:, :] - lnp[:, :-1, :]
        gradient_magnitude = torch.sqrt(dx[:, :-1, :]**2 + dy[:, :, :-1]**2)
        return torch.flatten(F.avg_pool2d(gradient_magnitude, kernel_size=16))

    def _noise_features(self, lnp):
        high_freq = lnp - F.avg_pool2d(lnp, kernel_size=3, stride=1, padding=1)
        return torch.flatten(F.avg_pool2d(high_freq**2, kernel_size=16))