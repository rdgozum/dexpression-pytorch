import torch
import torch.nn as nn
from torch import optim

import math
from datetime import datetime

from dexpression_pytorch import settings

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


def run(
    fold,
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=32,
    n_epochs=15,
    print_every=20,
    save_every=70,
    learning_rate=0.001,
):
    # Initialize variables
    history = []

    # Initialize criterion and optimizers
    criterion = nn.NLLLoss()
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Ensure dropout layers are in train mode
    model.train()

    # Perform training
    for epoch in range(n_epochs):
        running_train_accuracy = 0.0
        running_train_loss = 0.0

        running_test_accuracy = 0.0
        running_test_loss = 0.0

        iteration = 0
        n_iters_train = len(x_train) / batch_size
        n_iters_test = len(x_test) / batch_size
        for index in range(0, len(x_train), batch_size):
            x_batch = x_train[index : index + batch_size]
            y_batch = y_train[index : index + batch_size]
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            train_accuracy, train_loss = train(
                x_batch, y_batch, model, criterion, model_optimizer
            )
            running_train_accuracy += train_accuracy.item()
            running_train_loss += train_loss.item()

            # Perform testing
            iteration += 1
            if iteration % print_every == 0:
                with torch.no_grad():
                    model.eval()
                    for index in range(0, len(x_test), batch_size):
                        x_batch = x_test[index : index + batch_size]
                        y_batch = y_test[index : index + batch_size]
                        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                        test_accuracy, test_loss = test(
                            x_batch, y_batch, model, criterion
                        )
                        running_test_accuracy += test_accuracy.item()
                        running_test_loss += test_loss.item()

                # Track metrics
                avg_train_accuracy = running_train_accuracy / print_every
                avg_train_loss = running_train_loss / print_every

                avg_test_accuracy = running_test_accuracy / n_iters_test
                avg_test_loss = running_test_loss / n_iters_test

                history.append(
                    [
                        avg_train_accuracy,
                        avg_test_accuracy,
                        avg_train_loss,
                        avg_test_loss,
                    ]
                )

                # Reset metrics
                running_train_accuracy = 0.0
                running_train_loss = 0.0

                running_test_accuracy = 0.0
                running_test_loss = 0.0

                model.train()

                # Print progress
                print_progress(
                    fold,
                    epoch,
                    iteration,
                    n_epochs,
                    n_iters_train,
                    avg_train_accuracy,
                    avg_train_loss,
                    avg_test_accuracy,
                    avg_test_loss,
                )

            if iteration % save_every == 0 and fold == 0:  # remove fold == 0
                model_name = "cnn-fold{:d}-{:d}".format(
                    fold + 1, int(datetime.now().timestamp()),
                )
                checkpoint = "{:s}_{:d}_{:d}-{:.2f}.tar".format(
                    model_name, epoch + 1, iteration, train_accuracy,
                )

                print("Saving checkpoint {}...".format(settings.results(checkpoint)))
                print("")

                torch.save(
                    {
                        "fold": fold + 1,
                        "epoch": epoch + 1,
                        "iteration": iteration,
                        "model": model.state_dict(),
                        "model_opt": model_optimizer.state_dict(),
                        "loss": train_loss,
                    },
                    settings.results(checkpoint),
                )


def print_progress(
    fold,
    epoch,
    iteration,
    n_epochs,
    n_iters,
    avg_train_accuracy,
    avg_train_loss,
    avg_test_accuracy,
    avg_test_loss,
):
    print(
        "Fold: %d | Epoch: %d/%d | Iteration: %d/%d"
        % (fold + 1, epoch + 1, n_epochs, iteration, n_iters)
    )
    print("Train Accuracy: %.2f%%" % (avg_train_accuracy * 100))
    print("Train Loss: %.3f%%" % (avg_train_loss * 100))
    print("Test Accuracy: %.2f%%" % (avg_test_accuracy * 100))
    print("Test Loss: %.3f%%" % (avg_test_loss * 100))
    print("")
