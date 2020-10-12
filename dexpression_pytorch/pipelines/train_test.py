import torch
import torch.nn as nn
from torch import optim

from dexpression_pytorch.pipelines import pipeline_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

history = []


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

    # Get predicted and ground truth values
    y_pred = y_pred.to("cpu").tolist()
    y_truth = y_truth.to("cpu").tolist()

    return accuracy, loss, y_pred, y_truth


def run(
    fold,
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=32,
    n_epochs=25,
    learning_rate=0.001,
):
    # Initialize variables
    global history

    # Initialize criterion and optimizers
    criterion = nn.NLLLoss()
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):
        running_train_accuracy = 0.0
        running_train_loss = 0.0

        running_test_accuracy = 0.0
        running_test_loss = 0.0

        n_iters_train = len(x_train) / batch_size
        n_iters_test = len(x_test) / batch_size

        # Perform training
        model.train()
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
        with torch.no_grad():
            model.eval()

            test_pred, test_truth = [], []
            for index in range(0, len(x_test), batch_size):
                x_batch = x_test[index : index + batch_size]
                y_batch = y_test[index : index + batch_size]
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                test_accuracy, test_loss, pred, truth = test(
                    x_batch, y_batch, model, criterion
                )
                running_test_accuracy += test_accuracy.item()
                running_test_loss += test_loss.item()
                test_pred.extend(pred)
                test_truth.extend(truth)

        # Track metrics
        avg_train_accuracy = running_train_accuracy / n_iters_train
        avg_train_loss = running_train_loss / n_iters_train

        avg_test_accuracy = running_test_accuracy / n_iters_test
        avg_test_loss = running_test_loss / n_iters_test

        history.append(
            {
                "fold": fold + 1,
                "epoch": epoch + 1,
                "avg_train_accuracy": avg_train_accuracy * 100,
                "avg_test_accuracy": avg_test_accuracy * 100,
                "avg_train_loss": avg_train_loss,
                "avg_test_loss": avg_test_loss,
                "test_pred": test_pred,
                "test_truth": test_truth,
            }
        )

        # Print progress
        pipeline_utils.print_progress(
            fold,
            epoch,
            n_epochs,
            avg_train_accuracy,
            avg_train_loss,
            avg_test_accuracy,
            avg_test_loss,
        )

        # Save progress
        pipeline_utils.save_progress(
            fold, epoch, avg_test_accuracy, model, model_optimizer
        )
