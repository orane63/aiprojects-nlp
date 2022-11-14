import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class GRUNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, sequence_length):
        super().__init__()
        self.hidden_size = hidden_dim
        self.sequence_length = sequence_length
        self.recurrent_layer = nn.GRU(input_size = input_dim, hidden_size = hidden_dim, num_layers = layers, batch_first = False)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output, hn = self.recurrent_layer(x)
        answer = self.classifier(output[25-1])
        answer = self.sigmoid(answer)

        return answer