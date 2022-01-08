#TODO: Docstring
"""A one line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

  Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
from neural_methods.trainer.trainer import trainer
import torch
from torch.autograd import Variable
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
import torch.optim as optim
import numpy as np

class physnet_trainer(trainer):
    @staticmethod
    def add_trainer_args(parser):
        """Adds arguments to Paser for training process"""
        parser.add_argument("--frame_num", default=64, type=int)
        parser.add_argument('--learn_rate', default=1e-4, type=float)
        parser.add_argument('--round_num_max', default=20, type=int)
        return parser

    def __init__(self, args, twriter):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = device = torch.device("cuda:" + str(args.device)
                          if (args.device >= 0 and torch.cuda.is_available()) else "cpu")
        self.model =  PhysNet_padding_Encoder_Decoder_MAX(
            frames=args.frame_num).to(self.device)  # [3, T, 128,128]
        self.loss_model = Neg_Pearson()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learn_rate)
        self.round_num_max = args.round_num_max
        self.twriter = twriter
        print(self.device)

    def train(self,data_loader):
        """ TODO:Docstring"""
        for round in range(self.round_num_max):
            print(f"====training:ROUND{round}====")
            train_loss = []
            self.model.train()

            for i, batch in enumerate(data_loader["train"]):
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    Variable(batch[0]).to(torch.float32).to(self.device))
                BVP_label = Variable(batch[1]).to(torch.float32).to(self.device)
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                loss_ecg.backward()
                train_loss.append(loss_ecg.item())
                self.optimizer.step()
                self.optimizer.zero_grad()
            train_loss = np.asarray(train_loss)
            self.twriter.add_scalar("train_loss", scalar_value=float(
                loss_ecg), global_step=round)
            print(np.mean(train_loss))
            valid_loss = self.valid(data_loader)
            self.twriter.add_scalar("valid_loss", scalar_value=float(
                valid_loss), global_step=round)

    def valid(self,data_loader):
        """ Runs the model on valid sets."""
        print(" ====validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            for valid_i, valid_batch in enumerate(data_loader["valid"]):
                BVP_label = Variable(valid_batch[1]).to(torch.float32).to(self.device)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    Variable(valid_batch[0]).to(torch.float32).to(self.device))
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
            valid_loss = np.asarray(valid_loss)
            print(np.mean(valid_loss))
        return np.mean(valid_loss)

    def test(self,data_loader):
        """ Runs the model on test sets."""
        print(" ====testing===")
        test_step = 0
        test_loss = []
        self.model.eval()
        with torch.no_grad():
            for test_i, test_batch in enumerate(data_loader["test"]):
                BVP_label = Variable(test_batch[1]).to(torch.float32).to(self.device)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    Variable(test_batch[0]).to(torch.float32).to(self.device))
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                self.twriter.add_scalar("test_loss", scalar_value=float(
                    loss_ecg), global_step=test_step)
                test_step += 1
                print(loss_ecg.item())
                test_loss.append(test_loss)
        return np.mean(test_loss)
    #
    # def save_model(self):
    # def load_model(self):
