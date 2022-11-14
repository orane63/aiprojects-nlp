import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm
from torchtext.data import get_tokenizer
import numpy as np
tokenizer = get_tokenizer("basic_english")

max_words = 25
def vectorize_batch(X,token2idx):
    one_hot_len = len(token2idx)
    # separate the question into individual tokens (words)
    X = [tokenizer(x) for x in X]
    # make all sentences have the same number of tokens, pad with empty string or cut as needed
    X = [tokens+[""] * (max_words-len(tokens))  if len(tokens)<max_words else tokens[:max_words] for tokens in X]
    # note that this shape will require batch_first = true for the lstm, so we will transpose it at the end
    X_tensor = torch.zeros(len(X), max_words, one_hot_len)
    for i, tokens in enumerate(X):
      for j in range(len(tokens)):
        if(tokens[j] in token2idx):
          X_tensor[i][j][token2idx[tokens[j]]] = 50
    # with the transpose, we can have batch_first = false for the lstm
    return torch.transpose(X_tensor, 0, 1)
def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.
    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # Note: batch_size = len(val_dataset), so that's the whole validation set
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=len(val_dataset), shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch, (X, y) in tqdm(enumerate(train_loader)):
            # Predictions and loss
            inputs = vectorize_batch(X,train_dataset.token2idx)
            y = y.type(torch.float)

            pred = model(inputs)
            pred = np.squeeze(pred)
            loss = loss_fn(pred, y)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Periodically evaluate our model + log to Tensorboard
            if batch % n_eval == 0:
                # Compute training loss and accuracy.

                # CHANGE PATH AS NECESSARY
                path = "/Users/williamzhao/Documents/GRUSaves"
                name = "RNN" + str(batch) + ".pt"
                torch.save(model, path + name)
                accuracy = compute_accuracy(pred, y)
                print("batch loss: ", loss)
                print("batch accuracy: ", accuracy)

                # Compute validation loss and accuracy.
                val_loss, val_accuracy,val_f1,val_precision,val_recall = evaluate(val_loader, model, loss_fn,train_dataset)
                
                print("validation loss: ", val_loss)
                print("validation accuracy: ", val_accuracy)
                print("validation precision: ", val_precision)
                print("validation recall: ", val_recall)
                print("f1 score: ", val_f1)
                # TODO: Log the results to Tensorboard.



def compute_accuracy(outputs, labels):
    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total
def compute_precision(outputs, labels):
    tot_correct = 0
    tot_positive = 0
    rounds = torch.round(outputs)
    for i in range(len(outputs)):
      if(rounds[i]==1):
        tot_positive+=1.0
        if(labels[i]==1):
          tot_correct +=1.0
    if(tot_positive == 0):
      return 0
    return tot_correct / tot_positive
def compute_recall(outputs, labels):
    tot_correct = 0
    tot_positive = 0
    rounds = torch.round(outputs)
    for i in range(len(outputs)):
      if(labels[i]==1):
        tot_positive+=1.0
        if(rounds[i]==1):
          tot_correct +=1.0
    if(tot_positive == 0):
      return 0
    return tot_correct / tot_positive
def evaluate(val_loader, model, loss_fn,dataset):
    with torch.no_grad():
        # There should only be one batch (the entire validation set)
        for (X, y) in val_loader:
            
            inputs = vectorize_batch(X,dataset.token2idx)
            y = y.type(torch.float)
            
            pred = model(inputs)
            pred = torch.flatten(pred)
            loss = loss_fn(pred, y)
            accuracy = compute_accuracy(pred, y)
            f1 = f1_score(torch.round(pred), y, average='macro')
            precision, recall = compute_precision(pred,y),compute_recall(pred,y)
            return loss, accuracy, f1,precision,recall