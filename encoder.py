"""
    Code to encode two sentences and generate a score that signals if/how the two sentences are related:
    entailment, neutral, contradiction
"""

import torch
import torch.nn as nn

from models import MeanEmbedding, UniLSTM, BiLSTM, PoolBiLSTM

ENCODERS = {
    'mean': MeanEmbedding,
    'lstm': UniLSTM,
    'bilstm': BiLSTM,
    'poolbilstm': PoolBiLSTM
}
ENCODER_TYPES = list(ENCODERS.keys())
CLASSIFIER_TYPES = ["mlp", "linear"]

class Encoder(nn.Module):

    def __init__(self, embedding, opt):
        super().__init__()

        assert opt["encoder_type"] in ENCODER_TYPES, "Invalid encoder type: {}".format(opt['encoder_type'])
        assert opt["classifier"] in CLASSIFIER_TYPES, "Invalid classifier type: {}".format(opt['classifier'])

        self.embedding = embedding
        self.sentence_encoder = ENCODERS[opt['encoder_type']](opt)
        self.repr_size = {
            "mean": opt['embedding_size'],
            "lstm": opt['hidden_size'],
            "bilstm": opt['hidden_size'] * 2,
            "poolbilstm": opt['hidden_size'] * 2            
        }[opt['encoder_type']]

        if opt["classifier"] == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(self.repr_size * 4, 512),
                nn.ReLU(),
                nn.Linear(512, 3)
            )
        else:
            self.classifier = nn.Linear(self.repr_size * 4, 3)

    def forward(self, premises, hypotheses):

        # input premises and hypotheses are batches with various sentence lengths
        p, p_len = premises
        h, h_len = hypotheses

        # encode the two sentences
        u = self.sentence_encoder(self.embedding(p), p_len)
        v = self.sentence_encoder(self.embedding(h), h_len)

        # combine in one big vector
        combined = torch.concat([u, v, abs(u - v), u * v], dim=1)

        # calculate the score
        out = self.classifier(combined)

        return out


