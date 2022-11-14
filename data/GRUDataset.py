import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import torch

class GRUDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.df = data
        self.inputs = self.df.question_text.tolist() # list of questions
        self.labels = self.df.target.tolist() # list of labels
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=0.0014)
        self.sequences = self.vectorizer.fit_transform(self.df.question_text.tolist()) # matrix of word counts for each sample
        self.token2idx = self.vectorizer.vocabulary_ # dictionary converting words to their counts
        self.idx2token = {idx: token for token, idx in self.token2idx.items()} # same dictionary backwards
    def __getitem__(self, i):
        # return the ith sample's string and label
        return self.inputs[i], self.labels[i]

    def __len__(self):
        return len(self.labels)