import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy

from torch.utils.data import DataLoader
from argparse import ArgumentParser

from train import SNLIModule
from encoder import ENCODER_TYPES, CLASSIFIER_TYPES
from data import SNLIdataset
from vocab import Vocab


def get_vocab_and_embedding(dataset_dir, dataset):

    vocab = Vocab()

    # load or build vocabulary, based on dataset
    if opt["vocab_file"] != None:
        vocab.load(dataset_dir + opt["vocab_file"])
    else:
        try:
            vocab.load(dataset_dir + "snli_vocab.json")
        except:
            corpus = [ex["premise"] for ex in dataset]
            corpus += [ex["hypothesis"] for ex in dataset]        
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
    
    return vocab, embedding


def get_dataset(path, vocab, opt):

    dataset = SNLIdataset(
        path,
        tokenizer=vocab.tokenize,
        encoder=vocab.encode,
        max_seq_len=opt["num_layers"]
    )
    return dataset


def get_dataloader(dataset, opt):
    data_loader = DataLoader(
        dataset, 
        batch_size=opt["batch_size"], 
        collate_fn=dataset.batchify,
        num_workers=3,
        drop_last=True
    )
    return data_loader


def main(opt):

    dataset_dir = opt["data_dir"] + opt["dataset_dir"]

    # initialize vocab with tokenizer and encoder
    vocab = Vocab()

    # get datasets for training and validation
    train_dataset = get_dataset(dataset_dir + opt["dataset_file"] + "_train.jsonl", vocab, opt)
    valid_dataset = get_dataset(dataset_dir + opt["dataset_file"] + "_dev.jsonl", vocab, opt)
    train_loader = get_dataloader(train_dataset, opt)
    valid_loader = get_dataloader(valid_dataset, opt)

    # get vocabulary based on dataset and matching embedding
    vocab, embedding = get_vocab_and_embedding(dataset_dir, train_dataset)

    # init model and trainer
    snli_model = SNLIModule(embedding=embedding, opt=opt)

    trainer = pl.Trainer(
        accelerator=opt["accelerator"],
        devices=opt["devices"],
        strategy = DDPStrategy(find_unused_parameters=False),
        callbacks=[
            EarlyStopping(monitor="lr", stopping_threshold=opt["lr_limit"])
        ],
        log_every_n_steps=1,
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
