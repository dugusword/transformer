import time
import torch
from torch import nn, optim
import torch.nn.functional as F


class Scheduler:
    """
    Optimizer wrapper with a customized schedule
    
    Attributes
    ----------
    optimizer     : torch.optim.Optimizer
    last_step     : int
    factor        : float
    warmup_factor : float
    """
    
    def __init__(self, optimizer, d_model, warmup_steps):
        """
        Parameters
        ----------
        optimizer    : torch.optim.Optimizer
            wrapped optimizer
        d_model      : int
            size of embedding
        warmup_steps : int
            number of warmup steps before slowing down
        """
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

    def zero_grad(self):
        self.optimizer.zero_grad()
        
def scheduled_adam_optimizer(model):
    """
    Parameters
    ----------
    model : module.Transformer
    """
    adam = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    return Scheduler(adam, model.embedding.embedding_dim, 4000)

        
class Batch:
    """
    Container of a Batch

    Attributes
    ----------
    in_seq  : 2d tensor of int (batch_size, seq_len)
        input sequence
    out_seq : 2d tensor of int (batch_size, seq_len - 1)
        output sequence
    y       : 2d tensor of int (batch_size, seq_len - 1)
        gold prediction given in_seq and out_seq
    ntokens : int
        number of valid tokens
    """
    
    def __init__(self, in_seq, out_seq, padding_idx=0):
        """
        Parameters
        ----------
        in_seq      : 2d tensor of int (batch_size, seq_len)
            input sequence
        out_seq     : 2d tensor of int (batch_size, seq_len)
            output/target sequence
        padding_idx : int
            indicator of which value should be considered as padding
        """
        self.in_seq = in_seq
        self.out_seq = out_seq[:, :-1]
        self.y = out_seq[:, 1:]
        self.ntokens = (self.y != padding_idx).sum()


class LabelSmoothing(nn.Module):
    """
    Label Smoothing

    Attributes
    ----------
    criterion   : torch.nn.KLDivLoss
    padding_idx : int
    eps         : float
    n_vocab     : int
    """
    
    def __init__(self, n_vocab, eps, padding_idx=0):
        """
        Parameters
        ----------
        n_vocab     : int
            size of vocab
        eps         : float
            portion of the one hot that will be taken away
        padding_idx : int
            indicator of which value should be considered as padding
        """
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
    pred      : 3d tensor (batch_size, seq_len, n_vocab)
        this is the output of the transformer model
    gold      : 2d tenasor of int (batch_size, seq_len)
        target sequence
    ntokens   : int
        number of valid tokens
    criterion : function(2d tensor, 1d long tensor)
        a loss function. 1st arg is log probability distribution.
        2nd arg is raw gold data. the index of first argument corresponds
        to the value of 2nd arg.
    """
    # flatten pred and gold
    pred = pred.reshape(-1, pred.shape[2])
    gold = gold.reshape(-1)
    loss = criterion(pred, gold) / ntokens
    return loss

    
        
def run_epoch(data_iter, model, criterion, optimizer):
    """
    Parameters
    ----------
    data_iter : iterator of Batch
    model     : module.Transformer
    criterion : function(2d tensor, 1d long tensor) 
        use LabelSmoothing
    optimizer : torch.optim.Optimizer
        use scheduled_adam_optimizer(model) to generate one
    """
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
        optimizer.zero_grad()
        total_loss += loss
        tokens += batch.ntokens
        total_tokens += batch.ntokens
        if i % 100 == 1:
            elapsed = time.time() - start
            fmt = "Step: %d Elapsed: %f Loss: %f Tokens/sec: %f"
            lpt = loss / batch.ntokens
            tps = tokens / elapsed
            print(fmt % (i, elapsed, lpt, tps))
            tokens = 0
            start = time.time()

    return total_loss, total_tokens

