from encoder import Encoder, ENCODER_TYPES, ENCODERS
from data import SNLIdataset
from vocab import Vocab

from torch.utils.data import DataLoader
import torch
import spacy
import numpy as np
from tqdm import tqdm


nlp = spacy.load(r'/Users/FrankVerhoef/opt/anaconda3/envs/pai_parlai/lib/python3.9/site-packages/en_core_web_sm/en_core_web_sm-3.2.0')

B=3
PAD_VALUE=0

opt = {
    "embedding_size": 30,
    "hidden_size": 256,
    "num_layers": 15,
    "aggregate_method": "max",
    "encoder_type": ENCODER_TYPES[0],
    "batch_size": 64,
    "lr": 0.1,
    "lr_limit": 1E-5,
    "weight_decay": 0.99
}

def test_models():
    t  = torch.randint(low=0, high=10, size=(B, opt["num_layers"], opt["embedding_size"]), dtype=torch.float)
    lens = torch.ones((B), dtype=torch.int64) * opt["num_layers"]
    print(t)

    for enc_type in ENCODER_TYPES:
        m = ENCODERS[enc_type](opt)
        print(enc_type, m(t, lens))


def test_dataset():
    dataset = SNLIdataset("data/snli_1_0/snli_small_train.json", tokenizer=None, encoder=None, max_seq_len=opt["num_layers"])
    for i in range(3):
        print(i,dataset[i])


def embedding(x):
    B, L = x.shape
    # TODO in the meantime, return random embedding
    return torch.rand((B, L, opt["embedding_size"]))


def test_vocab():

    vocab = Vocab()
    vocab.load("vocab_test.json")
    dataset = SNLIdataset(
        "data/snli_1_0/snli_small_train.json",
        tokenizer=vocab.tokenize,
        encoder=vocab.encode,
        max_seq_len=opt["num_layers"]
    )

    num_samples = 100
    corpus = [ex["premise"] for ex in dataset[2000:2000+num_samples]]
    corpus += [ex["hypothesis"] for ex in dataset[2000:2000+num_samples]]

    #vocab.add_to_vocab(corpus)
    #print(vocab.t2id.items())  
    #vocab.save("vocab_test3.json")


def test_encoder():

    num_samples = 3
    vocab = Vocab()
    vocab.load("vocab_test.json")
    dataset = SNLIdataset(
        "data/snli_1_0/snli_small_train.json",
        tokenizer=vocab.tokenize,
        encoder=vocab.encode,
        max_seq_len=opt["num_layers"]
    ) 

    b = dataset.batchify(dataset[:num_samples])
    (p, h), t = b

    for enc_type in ENCODER_TYPES:

        opt["encoder_type"] = enc_type
        enc = Encoder(embedding, opt)
        r = enc(p, h)

        print(enc_type, r)


def test_dataloader():

    vocab = Vocab()
    vocab.load("vocab_test.json")
    dataset = SNLIdataset(
        "data/snli_1_0/snli_small_train.json",
        tokenizer=vocab.tokenize,
        encoder=vocab.encode,
        max_seq_len=opt["num_layers"]
    ) 

    l = DataLoader(
        dataset, 
        batch_size=64, 
        collate_fn=dataset.batchify
    )

    for i, batch in enumerate(l):
        print(batch)
        if i>5: break



test_models()
test_dataset()
test_vocab()
test_encoder()
test_dataloader()