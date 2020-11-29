import unittest
from modules.attention import *
import torch

class TestAttention(unittest.TestCase):

    def test_scaled_dot_attention(self):
        Q = torch.tensor([ [[1.0, 2.0, 3.0],
                            [1.0, 2.0, 3.0],
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]],
                           [[3.0, 2.0, 1.1],
                            [1.2, 2.4, 3.3],
                            [1.3, 2.5, 3.7],
                            [9.0, 2.2, 3.5]] ])

        K = torch.tensor([ [[3.0, 2.0, 3.0],
                            [1.0, 5.0, 3.0],
                            [1.0, 7.0, 3.0],
                            [4.0, 9.0, 6.0]],
                           [[3.0, 2.0, 1.1],
                            [9.2, 1.4, 3.3],
                            [2.3, 3.5, 3.7],
                            [9.0, 1.2, 3.5]] ])

        V = torch.tensor([ [[3.0, 2.0, 3.0],
                            [3.0, 5.0, 2.0],
                            [3.0, 4.0, 1.0],
                            [4.0, 5.0, 6.0]],
                           [[3.0, 2.0, 6.1],
                            [1.2, 3.4, 2.3],
                            [1.3, 4.5, 0.7],
                            [2.0, 5.2, 3.5]] ])
        

        attn = ScaledDotProductAttention()
        pred = attn(Q, K, V)
        self.assertEqual(pred.shape, torch.Size([2, 4, 3]))
        
if __name__ == '__main__':
    unittest.main()
