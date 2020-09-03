# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class BaseDecoder(nn.Module):
    """
    Take word/audio embeddings and output the next word probs
    Base decoder, cannot be called directly
    All decoders should inherit from this class
    """

    def __init__(self, input_size, vocab_size):
        super(BaseDecoder, self).__init__()
        self.input_size = input_size
        self.vocab_size = vocab_size

    def forward(self, x):
        raise NotImplementedError


class RNNDecoder(BaseDecoder):

    def __init__(self, input_size, vocab_size, **kwargs):
        super(RNNDecoder, self).__init__(input_size, vocab_size)
        hidden_size = kwargs.get('hidden_size', 256)
        num_layers = kwargs.get('num_layers', 1)
        bidirectional = kwargs.get('bidirectional', False)
        rnn_type = kwargs.get('rnn_type', "GRU")
        self.model = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)
        self.classifier = nn.Linear(
            hidden_size * (bidirectional + 1), vocab_size)
        nn.init.kaiming_uniform_(self.classifier.weight)
    
    def forward(self, *input, **kwargs):
        """
        RNN-style decoder must implement `forward` like this:
            accept a word embedding input and last time hidden state, return the word
            logits output and hidden state of this timestep
        the return dict must contain at least `logits` and `states`
        """
        if len(input) == 1:
            x = input # x: input word embedding/feature at timestep t
            states = None
        elif len(input) == 2:
            x, states = input
        else:
            raise Exception("unknown input type for rnn decoder")

        out, states = self.model(x, states)
        # out: [N, 1, hs], states: [num_layers * num_directionals, N, hs]

        output = {
            "states": states,
            "logits": self.classifier(out)
        }

        return output

    def init_hidden(self, bs):
        bidirectional = self.model.bidirectional
        num_layers = self.model.num_layers
        hidden_size = self.model.hidden_size
        return torch.zeros((bidirectional + 1) * num_layers, bs, hidden_size)

