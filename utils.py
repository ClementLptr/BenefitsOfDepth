import torch
import numpy as np


def target_function(x):
    """
    Target function with multiple oscillations.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output of the target function.
    """
    return torch.sin(10 * np.pi * x)


def target_function_slightly_oscillatory(x):
    """
    Quadratic target function: f(x) = x^2

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output of the quadratic target function.
    """
    return torch.sin(5 * np.pi * x)


def train_model(model, optimizer, criterion, x_train, y_train, epochs=1000):
    """
    Train the model on the given data.

    Args:
        model (nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        criterion (nn.Module): Loss function.
        x_train (torch.Tensor): Training data input.
        y_train (torch.Tensor): Training data target.
        epochs (int): Number of epochs to train the model.

    Returns:
        nn.Module: Trained model.
    """
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(x_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
    return model


def evaluate_model(model, x_test, y_test, criterion):
    """
    Evaluate the model and compute the MSE loss.

    Args:
        model (nn.Module): The model to evaluate.
        x_test (torch.Tensor): Test data input.
        y_test (torch.Tensor): Test data target.
        criterion (nn.Module): Loss function.

    Returns:
        float: Mean Squared Error loss.
        torch.Tensor: Predictions of the model.
    """
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
        mse = criterion(predictions, y_test).item()
    return mse, predictions
