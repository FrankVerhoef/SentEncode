from data import SNLIdataset
from vocab import Vocab


opt = {
    "data_dir": "data/",
    "dataset_dir": "snli_1_0/",
    "vocab_file": "vocab.json",
    "dataset_file": "snli_small",
    "embedding_size": 300,
    "num_layers": 64
}


def build_vocab():

    print("===== BUILD VOCAB =====")
    dataset_dir = opt["data_dir"] + opt["dataset_dir"]
    vocab = Vocab()

    # dataset for training
    dataset_train = SNLIdataset(
        dataset_dir + opt["dataset_file"] + "_train.jsonl",
        tokenizer=vocab.tokenize,
        encoder=vocab.encode,
        max_seq_len=opt["num_layers"]
    )

    # build vocabulary, based on train dataset
    corpus = [ex["premise"] for ex in dataset_train]
    corpus += [ex["hypothesis"] for ex in dataset_train]        
    vocab.add_to_vocab(corpus)
    vocab.save(dataset_dir + opt["vocab_file"])

build_vocab()