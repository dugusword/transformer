import unittest
from modules.feed_forward import *

class TestFeedForward(unittest.TestCase):
    def test_feed_forward(self):
        seq = torch.tensor([ [[1.0, 2.0, 3.0],
                              [1.0, 2.0, 3.0],
                              [1.0, 2.0, 3.0],
                              [4.0, 5.0, 6.0]],
                             [[3.0, 2.0, 1.1],
                              [1.2, 2.4, 3.3],
                              [1.3, 2.5, 3.7],
                              [9.0, 2.2, 3.5]] ])
        fnn = FeedForwardNetwork(3, 5)
        pred = fnn(seq)
        self.assertEqual(pred.shape, torch.Size([2, 4, 3]))
