import unittest
from modules.transformer import *
import torch

class TestTransformer(unittest.TestCase):
    def test_encoder_decoder_unit(self):
        in_seq = torch.rand([2, 4, 3])
        encoder_unit = EncoderUnit(8, 3, 2, 2, 4)
        decoder_unit = DecoderUnit(8, 3, 2, 2, 4)
        encoded = encoder_unit(in_seq)
        self.assertEqual(encoded.shape, torch.Size([2, 4, 3]))
        out_seq = torch.rand([2, 5, 3])
        decoded = decoder_unit(encoded, out_seq)
        print(decoded)
