import torch
import torch.nn as nn
from torch import optim

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(x_batch, y_batch, model, criterion, model_optimizer):
    # Clean existing gradients
    model_optimizer.zero_grad()

    # Forward pass
    output = model(x_batch)
    _, y_pred = torch.max(output.data, 1)
    _, y_truth = torch.max(y_batch, 1)

    # Compute loss
    loss = criterion(output, y_truth)

    # Backpropagate the gradients
    loss.backward()

    # Update the parameters
    model_optimizer.step()

    # Compute accuracy
    correct_counts = y_pred.eq(y_truth.data.view_as(y_pred))

    # Convert correct_counts to float and then compute the mean
    accuracy = torch.mean(correct_counts.type(torch.FloatTensor))

    return accuracy, loss


def run(
    model,
    x_train,
    y_train,
    batch_size=64,
    n_epochs=10,
    print_every=100,
    plot_every=100,
    save_every=3000,
    learning_rate=0.001,
):
    # Initialize criterion and optimizers
    criterion = nn.CrossEntropyLoss()
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Ensure dropout layers are in train mode
    model.train()

    for epoch in range(n_epochs):
        for iter in range(0, len(x_train), batch_size):
            x_batch = x_train[iter : iter + batch_size]
            y_batch = y_train[iter : iter + batch_size]
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            accuracy, loss = train(x_batch, y_batch, model, criterion, model_optimizer)
