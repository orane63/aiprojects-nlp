import torch
import pandas as pd

class BagofwordsDataset(torch.utils.data.Dataset):
    def __init__(self, data, vectorizer):
        self.df = data
        self.sequences = vectorizer.transform(self.df.question_text.tolist()) # matrix of word counts for each sample
        self.labels = self.df.target.tolist() # list of labels
        self.token2idx = vectorizer.vocabulary_ # dictionary converting words to their counts
        self.idx2token = {idx: token for token, idx in self.token2idx.items()} # same dictionary backwards
    def __getitem__(self, i):
        # return the ith sample's list of word counts and label
        return self.sequences[i, :].toarray(), self.labels[i]

    def __len__(self):
        return self.sequences.shape[0]
