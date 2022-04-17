"""
    Code to make a sentence representation, using four different methods:
    1. MeanEmbedding: calculate the mean of the token embeddings
    2. Use LSTM to encode the tokens and use the last hidden state as sentence representation
    3. Use BiLSTM to encode the tokens and use the concatenation of the hidden state of last token of the forward pass 
        and the hidden state of first token of the backward pass as sentence representation
    4. Use BiLSTM to encode the tokens; then concatenate the forward and backward hidden states of each token, 
        and use pooling (max of average) over all tokens to create sentence representation

    All modules are initialised using a dict 'opt' with the relevant parameters for the model
"""

import torch
import torch.nn as nn

class MeanEmbedding(nn.Module):

    def __init__(self, opt):
        super().__init__()

    def forward(self, xs, xs_len):

        # calculate the mean of each x, excluding the padded positions
        repr = torch.stack([x[:x_len].mean(dim=-2) for x, x_len in zip(xs, xs_len)])

        return repr

class UniLSTM(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = opt['embedding_size'],
            hidden_size = opt['hidden_size'],
            num_layers = opt['num_layers'],
            batch_first = True,
            bidirectional = False
        )

    def forward(self, xs, xs_len):
        # input shape is B, L, E
        B, L, E = xs.shape

        # pack, run through LSTM
        packed_padded_x = torch.nn.utils.rnn.pack_padded_sequence(xs, xs_len, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm(packed_padded_x)

        # shape of h_n is L, B, H, so reshape and then use last hidden state as representation
        H = h_n.shape[-1]
        h_n = h_n[:L, :, :].reshape(B, L, H)
        # index_last = torch.tensor(xs_len).unsqueeze(dim=1).unsqueeze(dim=1).expand(size=(B, 1, H)) - 1
        # index_last.to(h_n.device)
        # print("lstm forward devices: ", h_n.device, index_last.device)
        # repr = h_n.gather(dim=1, index=index_last).squeeze(dim=1)
        repr = torch.stack([h[h_len-1, :] for h, h_len in zip(h_n, xs_len)])

        return repr

class BiLSTM(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = opt['embedding_size'],
            hidden_size = opt['hidden_size'],
            num_layers = opt['num_layers'],
            batch_first = True,
            bidirectional = True
        )

    def forward(self, xs, xs_len):
        # input shape is B, L, E
        B, L, E = xs.shape

        # pack, run through LSTM
        packed_padded_x = torch.nn.utils.rnn.pack_padded_sequence(xs, xs_len, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm(packed_padded_x)

        # shape of h_n is 2 x L, B, H, first reshape
        H = h_n.shape[-1]
        h_n = h_n[:2*L, :, :].reshape(B, L, 2, -1)

        # use concat of last forward and first backward hidden state as representation
        last_forward = torch.stack([h[h_len-1, 0, :] for h, h_len in zip(h_n, xs_len)])
        # index_last = torch.tensor(xs_len).unsqueeze(dim=1).unsqueeze(dim=1).expand(size=(B, 1, H)) - 1
        # last_forward = h_n[:, :, 0, :].gather(dim=1, index=index_last).squeeze(dim=1)
        first_backward = h_n[:, 0, 1, :]
        repr = torch.concat([last_forward, first_backward], dim=-1)

        return repr


class PoolBiLSTM(nn.Module):

    def __init__(self, opt):
        super().__init__()
        assert opt["aggregate_method"] in ["max", "avg"], "Invalid aggregation method: {}".format(opt["aggregate_method"])
        self.lstm = nn.LSTM(
            input_size = opt['embedding_size'],
            hidden_size = opt['hidden_size'],
            num_layers = opt['num_layers'],
            batch_first = True,
            bidirectional = True
        )
        self.aggregate_method = opt['aggregate_method']

    def forward(self, xs, xs_len):
        # input shape is B, L, E
        B, L, E = xs.shape

        # pack, run through LSTM
        packed_padded_x = torch.nn.utils.rnn.pack_padded_sequence(xs, xs_len, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm(packed_padded_x)

        # shape of h_n is 2 x L, B, H
        # reshape h_n to combine forward and backward hidden states and perform pooling over the layers
        # TODO: check if pooling includes or excludes the padding
        h_n = h_n[:2*L, :, :].reshape(B, L, -1)
        if self.aggregate_method == "max":
            repr, _ = h_n.max(dim=1)
        elif self.aggregate_method == "avg":
            repr = h_n.mean(dim=1)
        else:
            repr = None # should never occur because of check at initialization

        return repr
