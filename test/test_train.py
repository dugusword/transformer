import unittest
from train import *
from modules.transformer import Transformer
import torch

class TestTrain(unittest.TestCase):
    def test_optimizer(self):
        model = Transformer(6)
        optimizer = scheduled_adam_optimizer(model)
    
    def test_label_smoothing(self):
        pred = torch.rand([7, 6])
        gold = torch.LongTensor([1, 2, 3, 4, 5, 0, 0])
        loss = LabelSmoothing(pred.shape[1], 0.1, 0)
        self.assertEqual(loss(pred, gold).shape, torch.Size([]))
        
    def test_compute_loss(self):
        pred = torch.rand([2, 7, 6])
        gold = torch.LongTensor([[1, 2, 3, 4, 5, 0, 0],
                                 [2, 1, 3, 3, 1, 0, 0]])
        criterion = LabelSmoothing(pred.shape[1], 0.1, 0)
        loss = compute_loss(pred, gold, 5, criterion)
        self.assertEqual(loss.shape, torch.Size([]))
        
if __name__ == '__main__':
    unittest.main()
