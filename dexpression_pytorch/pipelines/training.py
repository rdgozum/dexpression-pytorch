import torch
import torch.nn as nn
from torch import optim

import math
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


def test(x_batch, y_batch, model, criterion):
    # Forward pass
    output = model(x_batch)
    _, y_pred = torch.max(output.data, 1)
    _, y_truth = torch.max(y_batch, 1)

    # Compute loss
    loss = criterion(output, y_truth)

    # Calculate validation accuracy
    correct_counts = y_pred.eq(y_truth.data.view_as(y_pred))

    # Convert correct_counts to float and then compute the mean
    accuracy = torch.mean(correct_counts.type(torch.FloatTensor))

    return accuracy, loss


def print_progress(
    epoch,
    iteration,
    n_epochs,
    n_iters_train,
    avg_train_accuracy,
    avg_train_loss,
    avg_test_accuracy,
    avg_test_loss,
    start,
):
    training_time = timeSince(start, iteration / n_iters_train)

    print(f"Epoch {epoch+1}/{n_epochs} | Iteration {iteration}")
    print(f"Train Accuracy: {avg_train_accuracy*100:.2f}%")
    print(f"Train Loss: {avg_train_loss:.3f}")
    print(f"Test Accuracy: {avg_test_accuracy*100:.2f}%")
    print(f"Test Loss: {avg_test_loss:.3f}")
    print(f"Training Time: {training_time} seconds")
    print("")


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    return "%s" % asMinutes(s)


def run(
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=32,
    n_epochs=5,
    print_every=20,
    plot_every=100,
    save_every=3000,
    learning_rate=0.001,
):
    # Initialize variables
    start = time.time()
    history = []

    # Initialize criterion and optimizers
    criterion = nn.NLLLoss()
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Ensure dropout layers are in train mode
    model.train()

    for epoch in range(n_epochs):
        train_accuracy = 0.0
        train_loss = 0.0

        test_accuracy = 0.0
        test_loss = 0.0

        iteration = 0
        n_iters_train = len(x_train) / batch_size
        n_iters_test = len(x_test) / batch_size
        for index in range(0, len(x_train), batch_size):
            x_batch = x_train[index : index + batch_size]
            y_batch = y_train[index : index + batch_size]
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            accuracy, loss = train(x_batch, y_batch, model, criterion, model_optimizer)
            train_accuracy += accuracy.item()
            train_loss += loss.item()

            # Validation
            iteration += 1
            if iteration % print_every == 0:
                with torch.no_grad():
                    model.eval()
                    for index in range(0, len(x_test), batch_size):
                        x_batch = x_test[index : index + batch_size]
                        y_batch = y_test[index : index + batch_size]
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                        accuracy, loss = test(x_batch, y_batch, model, criterion)
                        test_accuracy += accuracy.item()
                        test_loss += loss.item()

                # Track metrics
                avg_train_accuracy = train_accuracy / print_every
                avg_train_loss = train_loss / print_every

                avg_test_accuracy = test_accuracy / n_iters_test
                avg_test_loss = test_loss / n_iters_test

                history.append(
                    [
                        avg_train_accuracy,
                        avg_test_accuracy,
                        avg_train_loss,
                        avg_test_loss,
                    ]
                )

                # Print results
                print_progress(
                    epoch,
                    iteration,
                    n_epochs,
                    n_iters_train,
                    avg_train_accuracy,
                    avg_train_loss,
                    avg_test_accuracy,
                    avg_test_loss,
                    start,
                )

                train_accuracy = 0.0
                train_loss = 0.0

                test_accuracy = 0.0
                test_loss = 0.0

                start = time.time()
                model.train()
