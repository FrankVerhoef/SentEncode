"""
Code to load the SNLI dataset
"""
from vocab import PAD_TOKEN
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import torch


LABEL_VALUE = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2
}

class SNLIdataset(Dataset):

    def __init__(self, path, tokenizer, encoder, max_seq_len, max=None):
        super().__init__()
        self.dataset = self._load_dataset(path, max)
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.max_seq_len = max_seq_len


    def _load_dataset(self, path, max=None):
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                ex = json.loads(line)

                # only include examples with valid label
                if ex["gold_label"] in LABEL_VALUE.keys():
                    dataset.append({
                        "premise": ex["sentence1"],
                        "hypothesis": ex["sentence2"],
                        "label": ex["gold_label"]
                    })
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

        # convert premises and hypotheses from text to token id's, and truncate to max sequence length
        p_ids = [self.encoder(self.tokenizer(ex["premise"]))[:self.max_seq_len] for ex in examples]
        h_ids = [self.encoder(self.tokenizer(ex["hypothesis"]))[:self.max_seq_len] for ex in examples]
        p_lengths = [len(sequence) for sequence in p_ids]
        h_lengths = [len(sequence) for sequence in h_ids]

        # convert to tensor and add padding
        p_ids = [torch.tensor(sequence, dtype=torch.int) for sequence in p_ids]
        h_ids = [torch.tensor(sequence, dtype=torch.int) for sequence in h_ids]
        p_padded = pad_sequence(p_ids, batch_first=True, padding_value=self.encoder(PAD_TOKEN))
        h_padded = pad_sequence(h_ids, batch_first=True, padding_value=self.encoder(PAD_TOKEN))

        # convert labels to label values and then to one-hot vectors
        labels = torch.tensor([LABEL_VALUE[ex["label"]] for ex in examples])
        ys = torch.eye(3).gather(dim=0, index=labels.unsqueeze(dim=1).expand(-1,3))

        return ((p_padded, p_lengths), (h_padded, h_lengths)), ys
