import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


class BaseTrainer():
    @staticmethod
    def add_trainer_args(parser):
        """Adds arguments to Paser for training process"""
        parser.add_argument('--lr', default=None, type=float)
        parser.add_argument(
            '--model_path', default=None, type=str)
        return parser

    def __init__(self):
        pass

    def train(self, data_loader):
        pass

    def valid(self, data_loader):
        pass

    def test(self):
        pass
