import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from argparse import ArgumentParser

from snli_lightning import SNLIModule
from models import ENCODER_TYPES
from encoder import CLASSIFIER_TYPES
from data import SNLIdataset
from vocab import Vocab


def main(opt):

    dataset_dir = opt["data_dir"] + opt["dataset_dir"]

    # initialize vocab with tokenizer and encoder
    vocab = Vocab()

    # get vocabulary from vocabfile
    vocab_path = dataset_dir + (opt["vocab_file"] if opt["vocab_file"] != None else "snli_vocab.json")
    assert vocab.load(vocab_path), print("Cannot load preprocessed vocab")

    # read matched embeddings from preprocessed file
    embed_path = dataset_dir + (opt["snli_embeddings"] if opt["snli_embeddings"] != None else "glove.snli.300d.txt")
    embedding = vocab.match_with_embeddings(path=embed_path, embedding_size=opt["embedding_size"])

    # get dataset for testing
    test_dataset = SNLIdataset(
        path=dataset_dir + opt["dataset_file"] + "_test.jsonl",
        tokenizer=vocab.tokenize,
        encoder=vocab.encode,
        max_seq_len=opt["num_layers"]
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=opt["batch_size"], 
        collate_fn=test_dataset.batchify
    )

    # init trainer
    trainer = pl.Trainer(
        gpus=1 if opt["device"] == "gpu" and torch.cuda.is_available() else 0,
        log_every_n_steps=1,
    )

    # load and test the model
    print("Start testing")
    snli_model = SNLIModule(embedding=embedding, opt=opt)
    snli_model.enc.sentence_encoder.load_state_dict(torch.load(opt["model_dir"] + "encoder_" + opt["encoder_type"]))
    snli_model.enc.classifier.load_state_dict(torch.load(opt["model_dir"] + "classifier_" + opt["encoder_type"]))
    trainer.test(model=snli_model, dataloaders=test_loader)


if __name__ == "__main__":
    parser = ArgumentParser()

    # files
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--dataset_dir", default="snli_1_0/")
    parser.add_argument("--dataset_file", default="snli_1.0")     # train, valid, test will be appended
    parser.add_argument("--vocab_file", default=None)
    parser.add_argument("--embeddings_file", default= "glove.840B.300d.txt")
    parser.add_argument("--snli_embeddings", default=None)
    parser.add_argument("--model_dir", default="models/")

    # device options
    parser.add_argument("--device", default="gpu")

    parser.add_argument("--batch_size", type=int, default=64)

    # model options
    parser.add_argument("--encoder_type", default="mean", choices=ENCODER_TYPES)
    parser.add_argument("--classifier", default="mlp", choices=CLASSIFIER_TYPES)
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=64)
    parser.add_argument("--aggregate_method", default="max", choices=["max", "avg"])

    args = parser.parse_args()
    opt = vars(args)
    print('Parameters')
    print('\n'.join(["{:20}\t{}".format(k,v) for k,v in opt.items()]))

    main(opt)
