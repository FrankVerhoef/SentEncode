import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch.utils.data import DataLoader
from argparse import ArgumentParser

from train import SNLIModule, LearningRateAdjustment
from data import SNLIdataset
from vocab import Vocab
from encoder import ENCODER_TYPES

import torch.nn as nn



def main(opt):

    vocab = Vocab()
    vocab.load("vocab_test.json")

    embedding = nn.Embedding(len(vocab.id2t), embedding_dim=opt["embedding_size"], padding_idx=0)

    dataset = SNLIdataset(
        "data/snli_1_0/snli_small_train.json",
        tokenizer=vocab.tokenize,
        encoder=vocab.encode,
        max_seq_len=opt["num_layers"],
        max=320
    )

    train_loader = DataLoader(
        dataset[:256], 
        batch_size=opt["batch_size"], 
        collate_fn=dataset.batchify,
        num_workers=8,
        drop_last=True
    )

    valid_loader = DataLoader(
        dataset[256:320], 
        batch_size=opt["batch_size"], 
        collate_fn=dataset.batchify,
        num_workers=1,
        drop_last=True
    )

    # init model
    snli_model = SNLIModule(embedding=embedding, opt=opt)

    trainer = pl.Trainer(
        accelerator=opt["accelerator"],
        devices=opt["devices"],
        max_epochs=opt["max_epochs"],
        callbacks=[
            LearningRateAdjustment(patience=opt["patience"]),
            EarlyStopping(monitor="lr", stopping_threshold=opt["lr_limit"])
        ],
        log_every_n_steps=10,
    )
    trainer.fit(model=snli_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
    parser = ArgumentParser()

    # device options
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default=8)

    # train options
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_limit", type=float, default=1E-5)
    parser.add_argument("--weight_decay", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=1)

    # model options
    parser.add_argument("--encoder_type", default="mean", choices=["mean", "lstm", "bilstm", "poolbilstm"])
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=64)
    parser.add_argument("--aggregate_method", default="max", choices=["max", "avg"])

    args = parser.parse_args()
    opt = vars(args)
    print(opt)

    main(opt)