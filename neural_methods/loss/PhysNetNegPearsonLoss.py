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
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])               
            sum_y = torch.sum(labels[i])             
            sum_xy = torch.sum(preds[i]*labels[i])       
            sum_x2 = torch.sum(torch.pow(preds[i],2))  
            sum_y2 = torch.sum(torch.pow(labels[i],2)) 
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss += 1 - pearson
            
            
        loss = loss/preds.shape[0]
        return loss


    # def forward(self, preds, labels):
    #     cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    #     pearson = cos(preds - preds.mean(dim=0, keepdim=True), labels - labels.mean(dim=0, keepdim=True))
    #     neg_pearson_sum = torch.mean(1-pearson)
    #     return neg_pearson_sum


