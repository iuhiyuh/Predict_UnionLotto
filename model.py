# -*- coding: utf-8 -*-
import torch
from torch import optim, nn


class Model(nn.Module):
    def __init__(self, conf):
        super().__init__()

        self.num_layers = conf.num_layers
        self.hidden_dim = conf.hidden_dim
        self.class_num = conf.class_num
        self.device = conf.device

        self.lstm = nn.LSTM(input_size=conf.input_size
                            , hidden_size=self.hidden_dim
                            , num_layers=self.num_layers
                            , batch_first=True).to(self.device)
        self.output = nn.Linear(in_features=conf.hidden_dim
                                , out_features=self.class_num).to(self.device)

    def forward(self, input, hidden=None):
        batch_size, seq_len, _ = input.size()

        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        else:
            h0, c0 = hidden.to(self.device)
        output, hidden = self.lstm(input, (h0, c0))

        output = self.output(output[:, -1, :])

        return output
