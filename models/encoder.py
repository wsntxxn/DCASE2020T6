# -*- coding: utf-8 -*-

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEncoder(nn.Module):
    
    """
    Encodes the given input into a fixed sized dimension
    Base encoder class, cannot be called directly
    All encoders should inherit from this class
    """

    def __init__(self, inputdim, embed_size):
        super(BaseEncoder, self).__init__()
        self.inputdim = inputdim
        self.embed_size = embed_size

    def init(self):
        for m in self.modules():
            m.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        raise NotImplementedError


class Block2D(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)


class LinearSoftPool(nn.Module):
    """LinearSoftPool
    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:
        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050
    """
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / time_decision.sum(
            self.pooldim)


class MeanPool(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)


class AttentionPool(nn.Module):  
    """docstring for AttentionPool"""  
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):  
        super().__init__()  
        self.inputdim = inputdim  
        self.outputdim = outputdim  
        self.pooldim = pooldim  
        self.transform = nn.Linear(inputdim, outputdim)  
        self.activ = nn.Softmax(dim=self.pooldim)  
        self.eps = 1e-7  


    def forward(self, logits, decision):  
        # Input is (B, T, D)  
        # B, T , D  
        w = self.activ(torch.clamp(self.transform(logits), -15, 15))  
        detect = (decision * w).sum(  
            self.pooldim) / (w.sum(self.pooldim) + self.eps)  
        # B, T, D  
        return detect


def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1
    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)
    elif poolingfunction_name == 'attention':  
        return AttentionPool(inputdim=kwargs['inputdim'],  
                             outputdim=kwargs['outputdim'])


class CRNNEncoder(BaseEncoder):

    def __init__(self, inputdim, embed_size, **kwargs):
        super(CRNNEncoder, self).__init__(inputdim, embed_size)
        features = nn.ModuleList()
        self.use_hidden = False
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,
                                                      inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(rnn_input_dim,
                          embed_size // 2,
                          bidirectional=True,
                          batch_first=True)
        self.features.apply(self.init_weights)

    def forward(self, *input):
        x, lens = input
        lens = copy.deepcopy(lens)
        lens = torch.as_tensor(lens)
        N, T, _ = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        
        lens /= 4
        # x: [N, T, E]
        idxs = torch.arange(x.size(1), device="cpu").repeat(N).view(N, x.size(1))
        mask = (idxs < lens.view(-1, 1)).to(x.device)
        # mask: [N, T]

        x_mean_time = x * mask.unsqueeze(-1)
        x_mean = x_mean_time.sum(1) / lens.unsqueeze(1).to(x.device)

        # x_max = x
        # x_max[~mask] = float("-inf")
        # x_max, _ = x_max.max(1)
        # out = x_mean + x_max

        out = x_mean

        return {
            "audio_embeds": out,
            "audio_embeds_time": x_mean_time,
            "state": None,
            "audio_embeds_lens": lens
        }


class RNNEncoder(BaseEncoder):

    def __init__(self, inputdim, embed_size, **kwargs):
        super(RNNEncoder, self).__init__(inputdim, embed_size)
        hidden_size = kwargs.get('hidden_size', 256)
        bidirectional = kwargs.get('bidirectional', False)
        num_layers = kwargs.get('num_layers', 1)
        dropout = kwargs.get('dropout', 0.3)
        rnn_type = kwargs.get('rnn_type', "GRU")
        self.representation = kwargs.get('representation', 'time')
        assert self.representation in ('time', 'mean')
        self.use_hidden = kwargs.get('use_hidden', False)
        self.network = getattr(nn, rnn_type)(
            inputdim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True)
        self.outputlayer = nn.Linear(
            hidden_size * (bidirectional + 1), embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init()

    def forward(self, *input):
        x, lens = input
        lens = torch.as_tensor(lens)
        # x: [N, T, D]
        packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        packed_out, hid = self.network(packed)
        # hid: [num_layers, N, hidden]
        out_time, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out: [N, T, hidden]
        if not self.use_hidden:
            hid = None
        if self.representation == 'mean':
            N = x.size(0)
            idxs = torch.arange(x.size(1), device="cpu").repeat(N).view(N, x.size(1))
            mask = (idxs < lens.view(-1, 1)).to(x.device)
            # mask: [N, T]
            out = out_time * mask.unsqueeze(-1)
            out = out.sum(1) / lens.unsqueeze(1).to(x.device)
        elif self.representation == 'time':
            indices = (lens - 1).reshape(-1, 1, 1).expand(-1, 1, out.size(-1))
            # indices: [N, 1, hidden]
            out = torch.gather(out_time, 1, indices).squeeze(1)

        out = self.bn(self.outputlayer(out))
        return {
            "audio_embeds": out,
            "audio_embeds_time": out_time,
            "state": hid,
            "audio_embeds_lens": lens
        }


if __name__ == "__main__":
    encoder = CRNNEncoder(64, 256)
    x = torch.randn(4, 1500, 64)
    out = encoder(x, torch.tensor([1071, 985, 1500, 666]))
    print(out[0].shape)
