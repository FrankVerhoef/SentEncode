# SentEncode
Sentence encoder made as part of practical assignment for UvA AI course "Advanced Topics in Computational sementics

## Goal
The goal is to train a sentence encoder and classifier on a natural language inference (NLI) task, and then transfer the trained sentence encoder to another task. Performance in the transfer learning tasks can be measured with SentEval.

## Structure of the code

### train.py

Contains the main code to train the model, using pytorch lightning Trainer.
Usage: python train.py --encoder_type ...
Other command line options give flexibility to specify training parameters, model parameters and location of source files.

files

    --data_dir, default="data/"
    --dataset_dir, default="snli_1_0/"
    --dataset_file, default="snli_1.0"  # train.jsonl, valid.jsonl, test.jsonl will be appended
    --vocab_file, default=None
    --embeddings_file, default= "glove.840B.300d.txt"
    --snli_embeddings, default=None

device options

    --device, default="gpu"

train options

    --lr, type=float, default=0.1
    --lr_limit, type=float, default=1E-5
    --weight_decay, type=float, default=0.99
    --batch_size, type=int, default=64
    --patience, type=int, default=0

model options

    --encoder_type, default="mean", choices=["mean", "lstm", "bilstm", "poolbilstm"]
    --classifier, default="mlp", choices=["mlp", "linear"]
    --embedding_size, type=int, default=300
    --hidden_size, type=int, default=2048
    --num_layers, type=int, default=64
    --aggregate_method, default="max", choices=["max", "avg"]

After training, two model files are saved:
- encoder_<encoder_type>: contains the state dict of the sentence encoder
- classifier_<encoder_type>: contains the state dict of the classier
These state dicts can be used by test.py to calculate accuracy on the test dataset.
The sentence encoder state dict is also used by eval.py, to test performance of the sentence encoder on the SentEval tasks.

### test.py

Loads state dicts of sentence encoder and classifier and tests performance of the combination on the SNLI test dataset.
Command line options similar to train.py (see above).

In addition, there is the following option:

    --models_dir, default="models/"

### eval.py

Loads state dict of sentence encoder and tests performance of the sentence encoder against the SentEval tasks
Command line options are similar to train.py (see above).

In addition, there are the following options:

    --tasks, nargs="*", type=str, default=['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'SICKEntailment']
    --models_dir, default="models/"

### snli_lightning.py

A pytorch lightning wrapper for the embedding, the sentence encoder and the classifier.
Contains specification of the optimizer, learning rate scheduler, and the train, validation and test steps.

### encoder.py

Code to encode two sentences and generate a score that signals if/how the two sentences are related: entailment, neutral, contradiction

### models.py

Code that defines the four available options for the sentence encoder:
- MeanEmbedding: calculate the mean of the token embeddings;
- UniLSTM: use the last hidden state as sentence representation;
- BiLSTM: use the concatenation of the hidden state of last token of the forward pass and the hidden state of first token on the backward pass as sentence representation;
- PoolBiLSTM: concatenates the forward and backward hidden states of each token, and use pooling (max of average) over all tokens to create sentence representation.
All modules are initialised using a dict 'opt' with the relevant parameters for the model.

### vocab.py

Code for building the vocabulary from source sentences, and for matching of vocabulary with word embeddings

### data.py

Code to load the SNLI dataset.
Also contains the function 'batchify' that is used by the dataloader to transform a list of examples from the dataset to the batch format that is used by to train/evaluate the model

