import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def bagofwords_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
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
            X = X.type(torch.float)
            y = y.type(torch.float)

            pred = model(X)
            pred = np.squeeze(pred)
            loss = loss_fn(pred, y)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Periodically evaluate our model
            if batch % n_eval == 0:
                # Set the save_dir to the directory where you want to save your models
                save_dir = ""
                name = "BagOfWords" + str(batch) + ".pt"
                torch.save(model, save_dir + name)

                # Compute training loss and accuracy.
                accuracy = compute_accuracy(pred, y)
                print("batch loss: ", loss)
                print("batch accuracy: ", accuracy)

                # Compute validation loss and accuracy.
                val_loss, val_accuracy, f1 = bagofwords_evaluate(val_loader, model, loss_fn)
                print("validation loss: ", val_loss)
                print("validation accuracy: ", val_accuracy)
                print("f1 score: ", f1)
                # TODO: Log the results to Tensorboard.



def compute_accuracy(outputs, labels):
    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def bagofwords_evaluate(val_loader, model, loss_fn):
    with torch.no_grad():
        # There should only be one batch (the entire validation set)
        for (X, y) in val_loader:
            X = X.type(torch.float)
            y = y.type(torch.float)

            pred = model(X)
            pred = np.squeeze(pred)
            loss = loss_fn(pred, y)
            f1 = f1_score(torch.round(pred), y, average='macro')
            accuracy = compute_accuracy(pred, y)
            return loss, accuracy, f1
