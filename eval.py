# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import logging

from argparse import ArgumentParser
from models import ENCODERS, ENCODER_TYPES
from vocab import Vocab, PAD_TOKEN
import torch
from torch.nn.utils.rnn import pad_sequence


# Set PATHs
PATH_TO_SENTEVAL = '../SentEval'
PATH_TO_DATA = '../SentEval/data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):

    # initialize vocab with tokenizer and encoder
    vocab = Vocab()
    vocab.add_to_vocab(samples, tokenize=False)
    params.encoder = vocab.encode

    # match embeddings with vocab
    try:
        params.embedding = vocab.match_with_embeddings(
            path=opt['data_dir'] + 'glove_' + params.current_task + '.txt', 
            embedding_size=opt["embedding_size"]
        )
    except:
        params.embedding = vocab.match_with_embeddings(
            path = opt["data_dir"] + opt["embeddings_file"], 
            embedding_size = opt["embedding_size"],
            savepath=opt['data_dir'] + 'glove_' + params.current_task + '.txt'
        )
    params.wvec_dim = opt["embedding_size"]
    return

def batcher(params, batch):

    # load sentence encoder model
    sent_encoder = ENCODERS[opt["encoder_type"]](opt)
    sent_encoder.load_state_dict(torch.load(opt["models_dir"] + "encoder_" + opt["encoder_type"]))

    # take care of empty lines
    batch = [sent if sent != [] else ['.'] for sent in batch]

    # transform batch of sentences into batch of sentence representations
    word_ids = [torch.tensor(params.encoder(sent), dtype=torch.int) for sent in batch]
    sent_lens = [len(sent) for sent in batch]
    words_padded = pad_sequence(word_ids, batch_first=True, padding_value=params.encoder(PAD_TOKEN))
    word_embeddings = params.embedding(words_padded).type(torch.float)
    sent_repr = sent_encoder(word_embeddings, sent_lens)

    return sent_repr


# Set params for SentEval
params_senteval = {
    'task_path': PATH_TO_DATA, 
    'usepytorch': True, 
    'kfold': 5
}
params_senteval['classifier'] = {
    'nhid': 0, 
    'optim': 'rmsprop', 
    'batch_size': 128,
    'tenacity': 3, 
    'epoch_size': 2
}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":

    parser = ArgumentParser()
    # tasks
    parser.add_argument(
        "--tasks", 
        nargs="*", 
        type=str, 
        default=['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'SICKEntailment']
    )

    # files
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--embeddings_file", default= "glove.840B.300d.txt")
    parser.add_argument("--models_dir", default="models/")

    # model options
    parser.add_argument("--encoder_type", default="mean", choices=ENCODER_TYPES)
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--aggregate_method", default="max", choices=["max", "avg"])

    args = parser.parse_args()
    opt = vars(args)
    print('Parameters')
    print('\n'.join(["{:20}\t{}".format(k,v) for k,v in opt.items()]))

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    results = se.eval(opt["tasks"])
    print(results)
