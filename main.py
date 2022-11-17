import os

import constants
from data.BagOfWordsDataset import BagOfWordsDataset
from networks.BagOfWordsNetwork import BagOfWordsNetwork
from train_functions.bagofwords_train import bagofwords_train

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs: ", constants.EPOCHS)
    print("Batch size: ", constants.BATCH_SIZE)
    print("Hidden layer size: ", constants.HIDDEN_DIM)

    # Initalize dataset and model. Then train the model!
    # Make sure you have train.csv downloaded in your project
    data_path = "train.csv"

    train_bagofwords = False

    if train_bagofwords:
        data_pd = pd.read_csv(data_path)
        data, val = train_test_split(data_pd, test_size = 0.05, stratify = data_pd['target'], shuffle = True, random_state = SEED)
        vectorizer = CountVectorizer(stop_words='english', max_df=0.99, min_df=0.005)
        fit = vectorizer.fit(data.question_text.tolist())
        train_dataset = BagofwordsDataset(data, fit)
        val_dataset = BagofwordsDataset(val, fit)
        model = BagOfWordsNetwork(train_dataset.sequences.shape[1], constants.HIDDEN_DIM)
        starting_train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=model,
            hyperparameters=hyperparameters,
            n_eval=constants.N_EVAL,
        )


if __name__ == "__main__":
    main()
