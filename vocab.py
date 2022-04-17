import spacy
import json
import torch
import torch.nn as nn

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
        self.embeddings = None


    def add_to_vocab(self, sentences):

        # collect all tokens
        tokens = [t for s in sentences for t in self.tokenize(s)]

        # add new tokens to the indices
        for t in tokens:
            if t not in self.t2id.keys():
                self.t2id[t] = len(self.id2t)
                self.id2t.append(t)
                self.count[t] = 1
            else:
                self.count[t] += 1


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
            return self.t2id[UNK_TOKEN]


    def match_with_embeddings(self, embeddings):

        embeddings[UNK_TOKEN] = torch.stack(list(embeddings.values()), dim=0).mean(dim=0)
        embeddings[PAD_TOKEN] = torch.zeros(len(embeddings[UNK_TOKEN]))

        # Match vectors with tokens in vocabulary and rebuild index
        old_vocab = self.t2id.keys()
        self.t2id = {}
        self.id2t = []
        matched_embeddings = []
        dropped = {}

        for token in old_vocab:
            if token in embeddings.keys():
                self.t2id[token] = len(self.id2t)
                self.id2t.append(token)        
                matched_embeddings.append(embeddings[token])
            else:
                dropped[token] = self.count[token]

        print("Dropped {} tokens from vocabulary, with total count of {}".format(len(dropped.keys()), sum(dropped.values())))
        print("Examples: ", *(sorted(dropped.items(), key=lambda x:x[1], reverse=True)[:10]))
            
        self.embeddings = torch.stack(matched_embeddings, dim=0)

        embedding = nn.Embedding.from_pretrained(
            self.embeddings, 
            padding_idx=self.t2id[PAD_TOKEN]
        )

        return embedding


    def save(self, path):
        with open(path, "w") as f:
            f.write(json.dumps(self.count))
            f.write('\n')
        print("Saved vocabulary with {} tokens".format(len(self.id2t)))
    

    def load(self, path):
        with open(path, "r") as f:
            self.count = json.loads(f.readline())
            self.id2t = list(self.count.keys())
            self.t2id = {self.id2t[id]: id for id in range(len(self.id2t))}
        print("Loaded vocabulary with {} tokens".format(len(self.id2t)))


    def compare_vocab_and_embeddings(self, embeddings):

        oov, common = {}, {}
        for word in self.t2id.keys():
            if word in embeddings.keys():
                common[word] = self.count[word]
            else: 
                oov[word] = self.count[word]
        
        num_tokens = len(self.id2t)
        freq_tokens = sum(self.count.values())
        num_common = len(common.keys())
        freq_common = sum(common.values())
        num_oov = len(oov.keys())
        freq_oov = sum(oov.values())

        print("Count:     total {}, common {} oov {} ({:2.2%})".format(
            num_tokens, num_common, num_oov, num_oov/num_tokens
        ))
        print("Frequency: total {}, common {} oov {} ({:2.2%})".format(
            freq_tokens, freq_common, freq_oov, freq_oov/freq_tokens
        ))
        print("Most frequent out-of-vocabulary tokens:")
        for t,c in sorted(oov.items(), key=lambda x: x[1], reverse=True)[:20]:
            print("\t{:20} \t{}".format(t[:20], c))


def read_embeddings(path, embedding_size):

    # Read embeddings from file
    embeddings = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.split(' ')
            token = fields[0]
            try:
                vector = torch.tensor([float(value) for value in fields[1:]])
                assert \
                    len(vector) == embedding_size, \
                    "Embedding size from file {} does not match required embedding size {}".format(
                        len(vector), embedding_size
                    )
                embeddings[token] = vector
            except:
                print("Error converting embedding for <{}>\n{}".format(token, fields[1:]))
    print("Loaded {} embeddings".format(len(embeddings.keys())))

    return embeddings
