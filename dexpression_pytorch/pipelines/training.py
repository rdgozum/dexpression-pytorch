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
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size=64,
    n_epochs=10,
    print_every=100,
    plot_every=100,
    save_every=3000,
    learning_rate=0.001,
):
    # Initialize variables
    history = []
    # best_accuracy = 0.0

    # Initialize criterion and optimizers
    criterion = nn.CrossEntropyLoss()
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Ensure dropout layers are in train mode
    model.train()

    for epoch in range(n_epochs):
        epoch_start = time.time()
        train_accuracy = 0.0
        train_loss = 0.0

        test_accuracy = 0.0
        test_loss = 0.0

        for iter in range(0, len(x_train), batch_size):
            x_batch = x_train[iter : iter + batch_size]
            y_batch = y_train[iter : iter + batch_size]
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            train_accuracy, train_loss = train(
                x_batch, y_batch, model, criterion, model_optimizer
            )
            train_accuracy += train_accuracy.item() * batch_size
            train_loss += train_loss.item() * batch_size

        with torch.no_grad():
            model.eval()

            for iter in range(0, len(x_test), batch_size):
                x_batch = x_test[iter : iter + batch_size]
                y_batch = y_test[iter : iter + batch_size]
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                test_accuracy, test_loss = test(x_batch, y_batch, model, criterion)
                test_accuracy += test_accuracy.item() * batch_size
                test_loss += test_loss.item() * batch_size

            avg_train_accuracy = test_accuracy / len(x_train)
            avg_train_loss = train_loss / len(x_train)

            avg_test_accuracy = test_accuracy / len(x_test)
            avg_test_loss = test_loss / len(x_test)

            history.append(
                [avg_train_accuracy, avg_test_accuracy, avg_train_loss, avg_test_loss]
            )

            epoch_end = time.time()

            print(
                f"Epoch : {epoch}, Training: Accuracy: {avg_train_accuracy*100}%, Loss: f{avg_train_loss}, \n\t\tValidation : Accuracy: {avg_test_accuracy*100}%, Loss : {avg_test_loss}, Time: {epoch_end-epoch_start}s"
            )

            # Save if the model has best accuracy till now
            torch.save(model.state_dict(), f"model_{epoch}.pth")
