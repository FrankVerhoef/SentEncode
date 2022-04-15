import spacy
import tqdm as tqdm
import json
import itertools

nlp = spacy.load(r'/Users/FrankVerhoef/opt/anaconda3/envs/pai_parlai/lib/python3.9/site-packages/en_core_web_sm/en_core_web_sm-3.2.0')

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
#stoplist = set('for a of the and to in on of to are at'.split(' '))

class Vocab:

    def __init__(self):

        # initialize vocab with padding and unk tokens
        self.t2id, self.id2t = {}, []
        self.t2id[PAD_TOKEN] = 0
        self.id2t.append(PAD_TOKEN)
        self.t2id[UNK_TOKEN] = 1
        self.id2t.append(UNK_TOKEN)


    def add_to_vocab(self, sentences):

        # collect all tokens in dataset
        tokensets = [set(self.tokenize(s)) for s in sentences]
        tokens = list(set(itertools.chain.from_iterable(tokensets)))

        # add new tokens to the indices
        for t in tokens:
            if t not in self.t2id.keys():
                self.t2id[t] = len(self.id2t)
                self.id2t.append(t)


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

    def save(self, path):
        with open(path, "w") as f:
            f.write(json.dumps(self.t2id))
            f.write('\n')
        print("Saved vocabulary with {} tokens".format(len(self.id2t)))
    

    def load(self, path):
        with open(path, "r") as f:
            self.t2id = json.loads(f.readline())
            self.id2t = list(self.t2id.keys())
        print("Loaded vocabulary with {} tokens".format(len(self.id2t)))