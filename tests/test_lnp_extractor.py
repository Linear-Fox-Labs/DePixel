import unittest
import torch
from src.denoising_network import DenoisingNetwork
from src.lnp_extractor import LNPExtractor

class TestLNPExtractor(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        denoising_network = DenoisingNetwork().to(self.device)
        self.lnp_extractor = LNPExtractor(denoising_network)

    def test_lnp_shape(self):
        input_tensor = torch.randn(1, 3, 256, 256).to(self.device)
        lnp = self.lnp_extractor.extract_lnp(input_tensor)
        self.assertEqual(lnp.shape, input_tensor.shape)

    def test_lnp_range(self):
        input_tensor = torch.rand(1, 3, 256, 256).to(self.device)
        lnp = self.lnp_extractor.extract_lnp(input_tensor)
        self.assertTrue(torch.all(lnp >= -1) and torch.all(lnp <= 1))

    def test_lnp_mean(self):
        input_tensor = torch.rand(1, 3, 256, 256).to(self.device)
        lnp = self.lnp_extractor.extract_lnp(input_tensor)
        self.assertAlmostEqual(lnp.mean().item(), 0, delta=0.1)

    def test_lnp_std(self):
        input_tensor = torch.rand(1, 3, 256, 256).to(self.device)
        lnp = self.lnp_extractor.extract_lnp(input_tensor)
        self.assertLess(lnp.std().item(), 1)

if __name__ == '__main__':
    unittest.main()