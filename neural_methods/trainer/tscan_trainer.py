"""Trainer for TSCAN, but also applies to 3D-CAN, Hybrid-CAN, and DeepPhys."""

from neural_methods.trainer.trainer import trainer
import torch
from neural_methods.model.ts_can import TSCAN
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
import torch.optim as optim
import numpy as np
import os


class tscan_trainer(trainer):

    def __init__(self, config, twriter):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.model = TSCAN(config.FRAME_DEPTH).to(self.device)
        self.criterion = Neg_Pearson()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.TRAIN.LR)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_path = config.MODEL.MODEL_PATH
        self.twriter = twriter
        print(self.device)

    def train(self, data_loader):
        """ TODO:Docstring"""
        min_valid_loss = 1
        for epoch in range(self.round_num_max):
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            self.model.train()
            # Model Training
            for idx, batch in enumerate(data_loader["train"]):
                inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                pred_ppg = self.model(inputs)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:  # print every 100 mini-batches
                    print(f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
                train_loss.append(loss_ecg.item())
                self.twriter.add_scalar("train_loss", scalar_value=float(loss), global_step=round)
            # Model Validation
            valid_loss = self.valid(data_loader)
            self.twriter.add_scalar("valid_loss", scalar_value=float(valid_loss), global_step=round)
            # Saving the best model checkpoint based on the validation loss.
            if valid_loss < min_valid_loss:
                print("Updating the best ckpt")
                self.save_model()

    def validate(self, data_loader):
        """ Model evaluation on the validation dataset."""
        print(" ====Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            for valid_idx, valid_batch in enumerate(data_loader["valid"]):
                inputs_valid, labels_valid = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                pred_ppg_valid = self.model(inputs_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        print(" ====Testing===")
        test_step = 0
        test_loss = []
        self.model.eval()
        with torch.no_grad():
            for test_idx, test_batch in enumerate(data_loader["test"]):
                inputs_test, labels_test = test_batch[0].to(self.device), test_batch[1].to(self.device)
                pred_ppg_test = self.model(inputs_test)
                loss = self.criterion(pred_ppg_test, labels_test)
                test_loss.append(loss.item())
                self.twriter.add_scalar("test_loss", scalar_value=float(
                    loss), global_step=test_step)
                test_step += 1
        return np.mean(test_loss)

    def save_model(self):
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        torch.save(self.model.state_dict(), os.path.join(
            self.model_path, "tscan_pretrained.pth"))
