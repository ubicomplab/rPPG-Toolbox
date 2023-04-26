from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse, os
import pandas as pd
import numpy as np
import random
import math
from torchvision import transforms
from torch import nn


class Neg_Pearson(nn.Module):
    def __init__(self):
        super(Neg_Pearson, self).__init__()
        return

    def forward(self, preds, labels):
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        pearson = cos(preds - preds.mean(dim=0, keepdim=True), labels - labels.mean(dim=0, keepdim=True))
        return torch.mean(1 - pearson)


