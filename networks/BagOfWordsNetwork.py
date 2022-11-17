import torch
import torch.nn as nn


class BagOfWordsNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, vectorizer):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.vectorizer = vectorizer

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
