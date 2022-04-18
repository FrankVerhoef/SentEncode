from encoder import Encoder, ENCODER_TYPES, ENCODERS
from data import SNLIdataset
from vocab import Vocab

from torch.utils.data import DataLoader
import torch
import numpy as np


opt = {
    "vocab_path": None,
    "dataset_path": "data/snli_1_0/snli_small_train.jsonl",
    "dataset_dir": "data/snli_1_0/",
    "glove_path": "data/glove.840B.300d.txt",
    "snli_embeddings": None,
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
    dataset = SNLIdataset("data/snli_1_0/snli_small_train.json", tokenizer=None, encoder=None, max_seq_len=opt["num_layers"])
    for i in range(3):
        print(i,dataset[i])


def embedding(x):
    B, L = x.shape
    # TODO in the meantime, return random embedding
    return torch.rand((B, L, opt["embedding_size"]))


def test_vocab():

    print("===== TEST VOCAB =====")
    vocab = Vocab()
    dataset = SNLIdataset(
        opt["dataset_path"],
        tokenizer=vocab.tokenize,
        encoder=vocab.encode,
        max_seq_len=opt["num_layers"]
    )
    if opt["vocab_path"] == None:
        try:
            vocab.load(opt["dataset_dir"] + "snli_vocab.json")
        except:
            corpus = [ex["premise"] for ex in dataset[:1000]]
            corpus += [ex["hypothesis"] for ex in dataset[:1000]]        
            vocab.add_to_vocab(corpus)
            vocab.save(opt["dataset_dir"] + "snli_vocab.json")
    else:
        vocab.load(opt["vocab_path"])

    # embeddings = read_embeddings(path="data/glove.840B.300d.txt", embedding_size=opt["embedding_size"])
    # vocab.compare_vocab_and_embeddings(embeddings=embeddings)

    if opt["snli_embeddings"] == None:
        try:
            embedding = vocab.match_with_embeddings(
                path="data/glove.snli.300d.txt", 
                embedding_size=opt["embedding_size"], 
                savepath=None
            )
        except:       
            embedding = vocab.match_with_embeddings(
                path=opt["glove_path"], 
                embedding_size=opt["embedding_size"], 
                savepath="data/glove.snli.300d.txt"
            )
    else:
        embedding = vocab.match_with_embeddings(
            path=opt["snli_embeddings"],
            embedding_size=opt["embedding_size"], 
            savepath=None
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

    print("===== TEST ENCODER =====")
    num_samples = 3
    vocab = Vocab()
    vocab.load(opt["vocab_path"])
    dataset = SNLIdataset(
        opt["dataset_path"],
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
    vocab = Vocab()
    vocab.load(opt["vocab_path"])
    dataset = SNLIdataset(
        opt["dataset_path"],
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



#test_models()
#test_dataset()
test_vocab()
#test_encoder()
#test_dataloader()