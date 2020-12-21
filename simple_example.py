from train import *
from modules.transformer import Transformer
import torch
from torch.autograd import Variable

def data_gen(n_vocab, batch_size, n_batch, device):
    for i in range(n_batch):
        data = torch.randint(1, n_vocab, [batch_size, 10])
        data[:, 0] = 1
        data = data.to(device)
        yield Batch(data, data)


if __name__ == '__main__':
    n_vocab = 15
    model = Transformer(n_vocab)
    criterion = LabelSmoothing(n_vocab, 0.1)
    optimizer = scheduled_adam_optimizer(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data_iter = data_gen(n_vocab, 30, 200000, device)
    for epoch in range(1000):
        run_epoch(data_iter, model, criterion, optimizer)
