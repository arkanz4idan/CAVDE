import torch


def token_accuracy(

    prediction,

    target,

):

    pred = prediction.argmax(dim=-1)

    correct = (pred == target).sum().item()

    total = target.numel()

    return correct / total