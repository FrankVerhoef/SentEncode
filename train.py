import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

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

    # get datasets for training and validation
    train_dataset = SNLIdataset(
        path=dataset_dir + opt["dataset_file"] + "_train.jsonl",
        tokenizer=vocab.tokenize,
        encoder=vocab.encode,
        max_seq_len=opt["num_layers"]
    )
    # train_dataset.save_encoded(dataset_dir + opt["dataset_file"] + "_train_enc.json")
    valid_dataset = SNLIdataset(
        path=dataset_dir + opt["dataset_file"] + "_dev.jsonl",
        tokenizer=vocab.tokenize,
        encoder=vocab.encode,
        max_seq_len=opt["num_layers"]
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=opt["batch_size"], 
        collate_fn=train_dataset.batchify
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=opt["batch_size"], 
        collate_fn=valid_dataset.batchify
    )

    # get vocabulary from vocabfile or dataset
    vocab_path = dataset_dir + (opt["vocab_file"] if opt["vocab_file"] != None else "snli_vocab.json")
    if not vocab.load(vocab_path):
        print("Build vocab from training dataset")
        corpus = [ex["premise"] for ex in train_dataset]
        corpus += [ex["hypothesis"] for ex in train_dataset]        
        vocab.add_to_vocab(corpus)
        vocab.save(dataset_dir + "snli_vocab.json")

    # read matched embeddings from preprocessed file or else build from original embeddingsfile
    embed_path = dataset_dir + (opt["snli_embeddings"] if opt["snli_embeddings"] != None else "glove.snli.300d.txt")
    try:
        embedding = vocab.match_with_embeddings(path=embed_path, embedding_size=opt["embedding_size"])
    except:
        print("Start matching embeddings with unprocessed embeddingsfile")  
        embedding = vocab.match_with_embeddings(
            path=opt["data_dir"] + opt["embeddings_file"], 
            embedding_size=opt["embedding_size"], 
            savepath=dataset_dir + "glove.snli.300d.txt"
        )

    # init model and trainer
    snli_model = SNLIModule(embedding=embedding, opt=opt)

    trainer = pl.Trainer(
        gpus=1 if opt["device"]=="gpu" and torch.cuda.is_available() else 0,
        callbacks=[
            EarlyStopping(monitor="lr", stopping_threshold=opt["lr_limit"]),
            ModelCheckpoint(save_weights_only=True, monitor="val_acc", mode="max"),
        ],
        log_every_n_steps=1,
    )

    # train the model
    print("Start training")
    trainer.fit(model=snli_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    
    # save weights of encoder and classifier
    torch.save(snli_model.enc.sentence_encoder.state_dict(), "encoder_" + opt["encoder_type"])
    torch.save(snli_model.enc.classifier.state_dict(), "classifier_" + opt["encoder_type"])
    print("Saved sentence encoder in {}".format("encoder_" + opt["encoder_type"]))


if __name__ == "__main__":
    parser = ArgumentParser()

    # files
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--dataset_dir", default="snli_1_0/")
    parser.add_argument("--dataset_file", default="snli_1.0")     # train, valid, test will be appended
    parser.add_argument("--vocab_file", default=None)
    parser.add_argument("--embeddings_file", default= "glove.840B.300d.txt")
    parser.add_argument("--snli_embeddings", default=None)

    # device options
    parser.add_argument("--device", default="gpu")

    # train options
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_limit", type=float, default=1E-5)
    parser.add_argument("--weight_decay", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=0)

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
