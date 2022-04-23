"""
Code to load the SNLI dataset
"""
from vocab import PAD_TOKEN, UNK_TOKEN
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import torch

from tqdm import tqdm

LABEL_VALUE = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

class SNLIdataset(Dataset):

    def __init__(self, path, tokenizer, encoder, max_seq_len, max=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.max_seq_len = max_seq_len
        self.dataset = self._load_dataset(path, max)

    def _load_dataset(self, path, max=None):
        dataset = []
        i = 0
        with open(path, 'r') as f:
            for line in tqdm(f, desc="Load and tokenize dataset"):
                ex = json.loads(line)

                # only include examples with valid label
                if ex["gold_label"] in LABEL_VALUE.keys():
                    dataset.append({
                        "premise": self.tokenizer(ex["sentence1"])[:self.max_seq_len],
                        "hypothesis": self.tokenizer(ex["sentence2"])[:self.max_seq_len],
                        "gold_label": ex["gold_label"], 
                    })
                i += 1
                if max != None: 
                    if i>=max: break
        print("Loaded dataset from {} with {} examples".format(path, len(dataset)))
        return dataset


    def __getitem__(self, key):
        return self.dataset[key]


    def __len__(self):
        return len(self.dataset)


    def encode(self, ex):
        p = self.encoder(ex["premise"])
        h = self.encoder(ex["hypothesis"])
        l = LABEL_VALUE[ex["gold_label"]]
        return (p, h), l


    def batchify(self, examples):
        """
            Transforms a list of dataset elements to batch of consisting of (premises, hypotheses), labels
            Premises and hypotheses are both tuples (padded_batch, seqence_lengths)
        """
        p_ids, p_lens, h_ids, h_lens, labels = [], [], [], [], []
        for (p, h), l in [self.encode(ex) for ex in examples]:
            p_ids.append(torch.tensor(p, dtype=torch.int))
            h_ids.append(torch.tensor(h, dtype=torch.int))
            p_lens.append(len(p))
            h_lens.append(len(h))
            labels.append(l)

        p_padded = pad_sequence(p_ids, batch_first=True, padding_value=self.encoder(PAD_TOKEN))
        h_padded = pad_sequence(h_ids, batch_first=True, padding_value=self.encoder(PAD_TOKEN))
        labels = torch.tensor(labels)

        return ((p_padded, p_lens), (h_padded, h_lens)), labels
