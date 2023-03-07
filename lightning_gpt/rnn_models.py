import torch
import torch.nn as nn
from lightning import LightningModule
import torch.optim.lr_scheduler as lr_scheduler

class LSTM(LightningModule):
    """Stacked LSTM"""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, batch_size, lr, wdecay, optimizer, dropout=0.5, tie_weights=False, device = 'cuda'):
        super(LSTM, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.lr = lr
        self.wdecay = wdecay
        self.optimizer = optimizer
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntokens = ntoken
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.hidden = self.init_hidden(batch_size,device)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz,device):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid).to(device),
                    weight.new_zeros(self.nlayers, bsz, self.nhid).to(device))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid).to(device)
        
    def repackage_hidden(self,h):
        """Wraps hidden states in new Tensors, to detach them from their history."""

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, input, targets):
        input = input.T.contiguous()
        hidden = self.repackage_hidden(self.hidden)
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        output = decoded.view(output.size(0), output.size(1), -1)
        loss = self.criterion(output.view(-1, self.ntokens), targets.view(-1))
        return output, loss
    
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets = batch
        _, loss = self(idx, targets)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        idx, targets = batch
        _, loss = self(idx, targets)
        self.log("validation_loss", loss ,prog_bar=True)
        return loss

    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None
    ) -> torch.Tensor:
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:


        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wdecay)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), betas=(0, 0.999), eps=1e-9, weight_decay=self.wdecay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=2, threshold=0)
        else:
            raise ValueError('Invalid optimizer: ' + self.optimizer)

        return optimizer