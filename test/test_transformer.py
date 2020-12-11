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
        self.assertEqual(decoded.shape, torch.Size([2, 5, 3]))


    def test_positional_encoding(self):
        seq = torch.zeros([2, 4, 3])
        pe = PositionalEncoding(3)
        seq = pe(seq)
        self.assertTrue(abs(seq[0][1][0] - 0.8415) < 1e-4)


    def test_encoder_decoder(self):
        in_seq = torch.rand([2, 4, 3])
        out_seq = torch.rand([2, 5, 3])
        encoder = Encoder(6, 8, 3, 2, 2, 4)
        decoder = Decoder(6, 8, 3, 2, 2, 4)
        memory = encoder(in_seq)
        decoded = decoder(memory, out_seq)
        self.assertEqual(decoded.shape, torch.Size([2, 5, 3]))
        
