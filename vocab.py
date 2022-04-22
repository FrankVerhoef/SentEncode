import spacy
import json
import torch
import torch.nn as nn
from tqdm import tqdm

#nlp = spacy.load(r'/Users/FrankVerhoef/opt/anaconda3/envs/pai_parlai/lib/python3.9/site-packages/en_core_web_sm/en_core_web_sm-3.2.0')
nlp = spacy.load("en_core_web_sm")

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'


class Vocab:

    def __init__(self):

        # initialize vocab with padding and unk tokens
        self.t2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.count = {PAD_TOKEN: 0, UNK_TOKEN: 0}
        self.id2t = [PAD_TOKEN, UNK_TOKEN]


    def add_to_vocab(self, sentences):

        count_start = len(self.id2t)

        # add new tokens to the indices
        for s in tqdm(sentences):
            for t in self.tokenize(s):
                if t not in self.t2id.keys():
                    self.t2id[t] = len(self.id2t)
                    self.id2t.append(t)
                    self.count[t] = 1
                else:
                    self.count[t] += 1
        print("Added {} tokens to vocabulary".format(len(self.id2t) - count_start))


    def tokenize(self, sentence):

        tokens = [token.text.lower() for token in nlp(sentence)]
        return tokens


    def encode(self, tokens):
        if isinstance(tokens, list):
            return [self._encode(t) for t in tokens]
        else:
            return self._encode(tokens)


    def _encode(self, token):
        try:
            return self.t2id[token]
        except:
            print("UNK: {}".format(token))
            return self.t2id[UNK_TOKEN]


    def save(self, path):
        with open(path, "w") as f:
            f.write(json.dumps(self.count))
            f.write('\n')
        print("Saved vocabulary with {} tokens".format(len(self.id2t)))
    

    def load(self, path):
        try:
            with open(path, "r") as f:
                self.count = json.loads(f.readline())
                self.id2t = list(self.count.keys())
                self.t2id = {self.id2t[id]: id for id in range(len(self.id2t))}
            print("Loaded vocabulary with {} tokens".format(len(self.id2t)))
            return True
        except:
            return False


    def match_with_embeddings(self, path, embedding_size, savepath=None):

        def get_token_and_vector(line, embedding_size):
            fields = line.split()
            token = fields[0]
            try:
                vector = torch.tensor([float(value) for value in fields[1:]])
                assert len(vector) == embedding_size
                return token, vector
            except:
                print("Error creating embedding of size {} for <{}>\n{}".format(embedding_size, token, fields[1:]))
                return None, None

        def print_coverage_stats(oov):
            num_common = len(self.id2t)
            freq_common = sum(self.count.values())
            num_oov = len(oov.keys())
            freq_oov = sum(oov.values())
            num_total = num_common + num_oov
            freq_total = freq_common + freq_oov

            print("Vocab coverage:  total {}, common {} oov {} ({:2.2%})".format(
                num_total, num_common, num_oov, num_oov/num_total
            ))
            print("Corpus coverage: total {}, common {} oov {} ({:2.2%})".format(
                freq_total, freq_common, freq_oov, freq_oov/freq_total
            ))
            print("Most frequent out-of-vocabulary tokens:")
            for t,c in sorted(oov.items(), key=lambda x: x[1], reverse=True)[:10]:
                print("\t{:20} \t{}".format(t[:20], c))

        # Initialize new indices
        t2id_new, id2t_new, id2emb = {}, [], []

        # Accumulate embeddings for calculation of average
        num_embeddings = 0
        embedding_sum = torch.zeros(embedding_size)

        # Read embeddings from file and match with vocab
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                token, vector = get_token_and_vector(line, embedding_size)
                if token != None:
                    embedding_sum += vector
                    num_embeddings += 1
                    if token in self.id2t:
                        t2id_new[token] = len(id2t_new)
                        id2t_new.append(token)        
                        id2emb.append(vector)

        # If no PAD token found in file, set embedding for PAD_TOKEN to zero vector
        if not PAD_TOKEN in id2t_new:
            t2id_new[PAD_TOKEN] = len(id2t_new)
            id2t_new.append(PAD_TOKEN)        
            id2emb.append(torch.zeros(embedding_size))

        # If no UNK token found in file, set embedding for UNK_TOKEN to average of all embeddings
        if not UNK_TOKEN in id2t_new:
            t2id_new[UNK_TOKEN] = len(id2t_new)
            id2t_new.append(UNK_TOKEN)        
            id2emb.append(embedding_sum / num_embeddings)

        oov = {token: count for token, count in self.count.items() if token not in id2t_new}
        self.id2t = id2t_new
        self.t2id = t2id_new
        self.count = {token: count for token, count in self.count.items() if token in self.id2t}
        self.count[UNK_TOKEN] = sum(oov.values())

        print("Loaded {} embeddings from {}".format(num_embeddings, path))
        print_coverage_stats(oov=oov)

        if savepath != None:
            with open(savepath, "w") as f:
                for t, e in zip(self.id2t, id2emb):
                    f.write(t + ' ' + ' '.join(["{:1.4}".format(v) for v in e]))
                    f.write('\n')
            print("Saved {} matched snli token embeddings in ".format(len(id2emb), savepath))

        embedding = nn.Embedding.from_pretrained(
            torch.stack(id2emb, dim=0), 
            padding_idx=self.t2id[PAD_TOKEN]
        )

        return embedding
