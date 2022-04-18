from encoder import Encoder, ENCODER_TYPES, ENCODERS
from data import SNLIdataset
from vocab import Vocab

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np


opt = {
    "vocab_file": "snli_vocab.json",
    "data_dir": "data/",
    "dataset_dir": "snli_1_0/",
    "dataset_file": "snli_small",
    "embeddings_file": "glove.840B.300d.txt",
    "snli_embeddings": "glove.snli.300d.txt",
    "embedding_size": 300,
    "hidden_size": 8,
    "num_layers": 5,
    "aggregate_method": "max",
    "encoder_type": ENCODER_TYPES[0],
    "batch_size": 3,
    "lr": 0.1,
    "lr_limit": 1E-5,
    "weight_decay": 0.99
}

def test_models():
    print("===== TEST MODELS =====")
    t  = torch.randint(low=0, high=10, size=(opt["batch_size"], opt["num_layers"], opt["embedding_size"]), dtype=torch.float)
    lens = np.random.randint(low=1, high=opt["num_layers"], size=(opt["batch_size"])).tolist()
    print(t, lens)

    for enc_type in ENCODER_TYPES:
        m = ENCODERS[enc_type](opt)
        print(enc_type)
        print(m(t, lens))


def test_dataset():
    print("===== TEST DATASET =====")
    dataset_dir = opt["data_dir"] + opt["dataset_dir"]
    dataset = SNLIdataset(dataset_dir + opt["dataset_file"] + "_train.jsonl", tokenizer=None, encoder=None, max_seq_len=opt["num_layers"])
    for i in range(3):
        print(i,dataset[i])


def test_vocab():

    print("===== TEST VOCAB =====")
    dataset_dir = opt["data_dir"] + opt["dataset_dir"]
    vocab = Vocab()
    dataset = SNLIdataset(
        dataset_dir + opt["dataset_file"] + "_train.jsonl",
        tokenizer=vocab.tokenize,
        encoder=vocab.encode,
        max_seq_len=opt["num_layers"]
    )
    if opt["vocab_file"] != None:
        vocab.load(dataset_dir + opt["vocab_file"])
    else:
        try:
            vocab.load(dataset_dir + "snli_vocab.json")
        except:
            corpus = [ex["premise"] for ex in dataset[:1000]]
            corpus += [ex["hypothesis"] for ex in dataset[:1000]]        
            vocab.add_to_vocab(corpus)
            vocab.save(dataset_dir + "snli_vocab.json")

    # match dataset vocabulary with embeddings
    if opt["snli_embeddings"] != None:
        embedding = vocab.match_with_embeddings(path=dataset_dir + opt["snli_embeddings"], embedding_size=opt["embedding_size"])
    else:
        try:
            embedding = vocab.match_with_embeddings(path=dataset_dir + "glove.snli.300d.txt", embedding_size=opt["embedding_size"])
        except:       
            embedding = vocab.match_with_embeddings(
                path=opt["data_dir"] + opt["embeddings_file"], 
                embedding_size=opt["embedding_size"], 
                savepath=dataset_dir + "glove.snli.300d.txt"
            )

    test_sentence = "Frank is really an NLP hero!"
    test_tokens = vocab.tokenize(test_sentence)
    test_indices = vocab.encode(test_tokens)
    test_embeddings = embedding(torch.tensor(test_indices).unsqueeze(dim=0))
    print(test_sentence)
    print(test_tokens)
    print(test_indices)
    print(test_embeddings)

def test_encoder():

    def embedding(x):
        B, L = x.shape
        # return random embedding, is just for testing
        return torch.rand((B, L, opt["embedding_size"]))

    print("===== TEST ENCODER =====")
    dataset_dir = opt["data_dir"] + opt["dataset_dir"]
    num_samples = 3
    vocab = Vocab()
    vocab.load(dataset_dir + opt["vocab_file"])
    dataset = SNLIdataset(
        dataset_dir + opt["dataset_file"] + "_train.jsonl",
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

        print(enc_type)
        print(r)


def test_dataloader():

    print("===== TEST DATALOADER =====")
    dataset_dir = opt["data_dir"] + opt["dataset_dir"]
    vocab = Vocab()
    vocab.load(dataset_dir + opt["vocab_file"])
    dataset = SNLIdataset(
        dataset_dir + opt["dataset_file"] + "_dev.jsonl",
        tokenizer=vocab.tokenize,
        encoder=vocab.encode,
        max_seq_len=opt["num_layers"]
    ) 

    l = DataLoader(
        dataset, 
        batch_size=opt["batch_size"], 
        collate_fn=dataset.batchify
    )

    for i, batch in enumerate(l):
        print(batch)
        if i>5: break

def test_inference():

    print("===== TEST INFERENCE =====")
    dataset_dir = opt["data_dir"] + opt["dataset_dir"]
    vocab = Vocab()

    # dataset for validation
    dataset = SNLIdataset(
        dataset_dir + opt["dataset_file"] + "_dev.jsonl",
        tokenizer=vocab.tokenize,
        encoder=vocab.encode,
        max_seq_len=opt["num_layers"]
    )

    # load vocabulary
    vocab.load(dataset_dir + opt["vocab_file"])

    # match dataset vocabulary with embeddings
    embedding = vocab.match_with_embeddings(path=dataset_dir + opt["snli_embeddings"], embedding_size=opt["embedding_size"])

    # initialise encoder
    enc = Encoder(embedding, opt)

    # test a few sentences
    num_samples = 8
    b = dataset.batchify(dataset[:num_samples])
    (p, h), t = b

    o = enc(p, h)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(o, t)
    print("Label {}, output {}, loss {}".format(t, o, loss))

    for i, ex in enumerate(dataset[:num_samples]):
        print(i, ex, t[i], o[i])




test_models()
test_dataset()
test_vocab()
test_encoder()
test_dataloader()
test_inference()