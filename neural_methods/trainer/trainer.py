import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt


class trainer():
    @staticmethod
    def add_trainer_args(parser):
        """Adds arguments to Paser for training process"""
        parser.add_argument("--frame_num", default=64, type=int)
        parser.add_argument('--learn_rate', default=1e-4, type=float)
        parser.add_argument(
            '--model_path', default="store_model", type=str)
        return parser

    def __init__(self):
        pass

    def train(self, data_loader):
        pass

    def valid(self, data_loader):
        pass

    def test(self):
        pass
