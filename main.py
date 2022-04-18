import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy

from torch.utils.data import DataLoader
from argparse import ArgumentParser

from train import SNLIModule, LearningRateAdjustment
from data import SNLIdataset
from vocab import Vocab


def main(opt):

    dataset_dir = opt["data_dir"] + opt["dataset_dir"]
    vocab = Vocab()

    # dataset for training
    dataset_train = SNLIdataset(
        dataset_dir + opt["dataset_file"] + "_train.jsonl",
        tokenizer=vocab.tokenize,
        encoder=vocab.encode,
        max_seq_len=opt["num_layers"]
    )
    train_loader = DataLoader(
        dataset_train, 
        batch_size=opt["batch_size"], 
        collate_fn=dataset_train.batchify,
        num_workers=4,
        drop_last=True
    )

    # dataset for validation
    dataset_valid = SNLIdataset(
        dataset_dir + opt["dataset_file"] + "_dev.jsonl",
        tokenizer=vocab.tokenize,
        encoder=vocab.encode,
        max_seq_len=opt["num_layers"]
    )
   
    valid_loader = DataLoader(
        dataset_valid, 
        batch_size=opt["batch_size"], 
        collate_fn=dataset_valid.batchify,
        num_workers=4,
        drop_last=True
    )

    # load or build vocabulary, based on train dataset
    if opt["vocab_file"] != None:
        vocab.load(dataset_dir + opt["vocab_file"])
    else:
        try:
            vocab.load(dataset_dir + "snli_vocab.json")
        except:
            corpus = [ex["premise"] for ex in dataset_train]
            corpus += [ex["hypothesis"] for ex in dataset_train]        
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

    # init model and trainer
    snli_model = SNLIModule(embedding=embedding, opt=opt)

    trainer = pl.Trainer(
        accelerator=opt["accelerator"],
        devices=opt["devices"],
        strategy = DDPStrategy(find_unused_parameters=False),
        max_epochs=opt["max_epochs"],
        callbacks=[
            LearningRateAdjustment(patience=opt["patience"]),
            EarlyStopping(monitor="lr", stopping_threshold=opt["lr_limit"])
        ],
        log_every_n_steps=10,
    )

    # train the model
    print("Start training")
    trainer.fit(model=snli_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


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
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default=1)

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
    print('Parameters')
    print('\n'.join(["{:20}\t{}".format(k,v) for k,v in opt.items()]))

    main(opt)
