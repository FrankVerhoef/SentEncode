import json

datadir = "/Users/FrankVerhoef/Programming/ATCS/data/snli_1_0/"
snli_train = "snli.json"
newfile = "snli_small_train.json"

small_dataset = []

with open(datadir + snli_train, "r") as source:
    i = 0
    for line in source:
        if i % 10 == 0:
            ex = json.loads(line)
            small_dataset.append({
                "annotator_labels": ex["annotator_labels"],
                "gold_label": ex["gold_label"],
                "sentence1": ex["sentence1"],
                "sentence2": ex["sentence2"]
            })
        i += 1

with open(datadir + newfile, 'w') as target:
    for ex in small_dataset:
        target.write(json.dumps(ex))
        target.write('\n')


