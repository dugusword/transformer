import time
import torch
from torch import nn, optim
import torch.nn.functional as F


class Scheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.last_step = 0
        self.factor = d_model ** (-0.5)
        self.warmup_factor = warmup_steps ** (-1.5)

    def step(self):
        self.last_step += 1
        f, step, wf = self.factor, self.last_step, self.warmup_factor
        lr = f * min(step ** (-0.5), step * wf)
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.optimizer.step()

def scheduled_adam_optimizer(model):
    adam = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    return Scheduler(adam, model.embedding.embedding_dim, 4000)

        
class Batch:
    def __init__(self, in_seq, out_seq, padding_idx=0):
        self.in_seq = in_seq
        self.out_seq = out_seq[:, :-1]
        self.y = out_seq[:, 1:]
        self.ntokens = (self.y != padding_idx).sum()


class LabelSmoothing(nn.Module):
    def __init__(self, n_vocab, eps, padding_idx=0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.eps = eps
        self.n_vocab = n_vocab
        
    def forward(self, pred, gold):
        """
        Parameters
        ----------
        pred : 2d tensor (batch_size * seq_len, n_vocab)
        gold : 1d tensor of int (batch_size * seq_len)
        """
        n_vocab, eps, padding_idx = self.n_vocab, self.eps, self.padding_idx
        dist = pred.clone()
        dist.fill_(eps / (n_vocab - 2))
        dist.scatter_(1, gold.view(gold.shape[0], -1), 1 - eps)
        dist[:, padding_idx] = 0
        mask = (gold != padding_idx)
        dist.mul_(mask.view(dist.shape[0], 1))
        return self.criterion(pred, dist)
        

def compute_loss(pred, gold, ntokens, criterion):
    """
    Parameters
    ----------
    pred : 3d tensor (batch_size, seq_len, n_vocab)
        this is the output of the transformer model
    gold : 2d tenasor of int (batch_size, seq_len)
        target sequence
    """
    # flatten pred and gold
    pred = pred.reshape(-1, pred.shape[2])
    gold = gold.reshape(-1)
    loss = criterion(pred, gold) / ntokens
    return loss

    
        
def run_epoch(data_iter, model, criterion, optimizer):
    start = time.time()
    model.train()
    total_loss = 0
    total_tokens = 0
    tokens = 0
    
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.in_seq, batch.out_seq)
        loss = compute_loss(out, batch.y, batch.ntokens, criterion)
        loss.backward()
        optimizer.step()
        total_loss += loss
        tokens += batch.ntokens
        total_tokens += batch.ntokens
        if i % 100 == 0:
            elapsed = time.time() - start
            fmt = "Step: %d Elapsed: %f Loss: %f Tokens/sec: %f"
            lpt = loss / batch.ntokens
            tps = tokens / elapsed
            print(fmt % (i, elapsed, lpt, tps))
            tokens = 0
            start = time.time()

    return total_loss, total_tokens

