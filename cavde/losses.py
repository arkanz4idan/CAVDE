import torch.nn as nn


def create_loss(

    pad_id: int,

):

    return nn.CrossEntropyLoss(

        ignore_index=pad_id,

        label_smoothing=0.1,

    )