import os
import pandas as pd

import constants
from sklearn.model_selection import train_test_split

from data.GRUDataset import GRUDataset
from networks.GRUNetwork import GRUNetwork
from train_functions.GRU_Train import starting_train
max_words = 25

data_path = 'quora_questions_train.csv'
data_pd = pd.read_csv(data_path)
data, val = train_test_split(data_pd, test_size = 0.05, stratify = data_pd['target'], shuffle = True, random_state = constants.SEED)
train_dataset = GRUDataset(data)
val_dataset = GRUDataset(val)
model = GRUNetwork(len(train_dataset.token2idx), constants.HIDDEN_DIM, 1, max_words)
hyperparameters = {"batch_size": constants.BATCH_SIZE, "epochs": constants.EPOCHS}
starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
    )