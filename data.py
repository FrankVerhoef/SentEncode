"""
Code to load the SNLI dataset
"""
from vocab import PAD_TOKEN, UNK_TOKEN
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import torch

from timeit import default_timer as timer
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
            for line in tqdm(f, desc="Load and encode dataset"):
                ex = json.loads(line)

                # only include examples with valid label
                if ex["gold_label"] in LABEL_VALUE.keys():
                    dataset.append({
                        "premise": ex["sentence1"],
                        "hypothesis": ex["sentence2"],
                        "gold_label": ex["gold_label"], 
                        "p_ids": self.encoder(self.tokenizer(ex["sentence1"]))[:self.max_seq_len],
                        "h_ids": self.encoder(self.tokenizer(ex["sentence2"]))[:self.max_seq_len],
                        "label": LABEL_VALUE[ex["gold_label"]]
                    })
                i += 1
                if max != None: 
                    if i>=max: break
        print("Loaded dataset from {} with {} examples".format(path, len(dataset)))
        return dataset if max==None else dataset[:max]
        

    def __getitem__(self, key):
        return self.dataset[key]


    def __len__(self):
        return len(self.dataset)


    def batchify(self, examples):
        """
            Transforms a list of dataset elements to batch of consisting of (premises, hypotheses), labels
            Premises and hypotheses are both tuples (padded_batch, seqence_lengths)
        """
        start = timer()

        p_ids = [torch.tensor(ex["p_ids"], dtype=torch.int) for ex in examples]
        h_ids = [torch.tensor(ex["h_ids"], dtype=torch.int) for ex in examples]
        p_padded = pad_sequence(p_ids, batch_first=True, padding_value=self.encoder(PAD_TOKEN))
        h_padded = pad_sequence(h_ids, batch_first=True, padding_value=self.encoder(PAD_TOKEN))
        p_lens = torch.tensor([len(ex["p_ids"]) for ex in examples])
        h_lens = torch.tensor([len(ex["h_ids"]) for ex in examples])
        labels = torch.tensor([ex["label"] for ex in examples])

        end = timer()
        print("Batchify took {:8.2f} milliseconds".format(1000*(end-start)))

        return ((p_padded, p_lens), (h_padded, h_lens)), labels
