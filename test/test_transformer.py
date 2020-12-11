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
        seq = pe.eval()(seq)
        self.assertTrue(abs(seq[0][1][0] - 0.8415) < 1e-4)


    def test_encoder_decoder(self):
        in_seq = torch.rand([2, 4, 3])
        out_seq = torch.rand([2, 5, 3])
        encoder = Encoder(6, 8, 3, 2, 2, 4)
        decoder = Decoder(6, 8, 3, 2, 2, 4)
        memory = encoder(in_seq)
        decoded = decoder(memory, out_seq)
        self.assertEqual(decoded.shape, torch.Size([2, 5, 3]))
        
    def test_transformer_core(self):
        in_seq = torch.rand([2, 4, 3])
        out_seq = torch.rand([2, 5, 3])
        tc = TransformerCore(6, 6, 8, 3, 2, 2, 4)
        decoded = tc(in_seq, out_seq)
        self.assertEqual(decoded.shape, torch.Size([2, 5, 3]))

    def test_transformer(self):
        in_seq = torch.LongTensor([[1, 2, 3, 4], [8, 7, 6, 5]])
        out_seq = torch.LongTensor([[0, 1, 2], [3, 9, 3]])
        transformer = Transformer(10, 6, 6, 8, 3, 2, 2, 4)
        probs = transformer(in_seq, out_seq)
        print(probs.shape)
        print(probs.size(2))
        print(probs.shape[2])
        print(probs.view(-1, 10).shape)
