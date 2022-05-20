"""Trainer for TSCAN, but also applies to 3D-CAN, Hybrid-CAN, and DeepPhys."""

from neural_methods.trainer.BaseTrainer import BaseTrainer
import torch
from neural_methods.model.EfficientPhys import EfficientPhys
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import logging


class EfficientPhysTrainer(BaseTrainer):

    def __init__(self, config):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
        self.model = EfficientPhys(frame_depth=self.frame_depth,
                                   img_size=config.DATA.PREPROCESS.H).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.TRAIN.LR)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME

    def train(self, data_loader):
        """ TODO:Docstring"""
        min_valid_loss = 1
        for epoch in range(self.max_epoch_num):
            logging.debug(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)

            for idx, batch in enumerate(tbar):
                tbar.set_description("Epoch %s" % epoch)
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N*D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N*D)//self.frame_depth*self.frame_depth]
                labels = labels[:(N*D)//self.frame_depth*self.frame_depth]
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                logging.debug(loss.item())
                if idx % 100 == 99:  # print every 100 mini-batches
                logging.debug(
                    f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())
            valid_loss = self.valid(data_loader)
            if valid_loss < min_valid_loss:
                logging.debug("Updating the best ckpt")
                min_valid_loss = valid_loss
                self.save_model()
            logging.debug('valid loss: ', valid_loss)
            logging.debug('min_valid_loss: ', min_valid_loss)

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        logging.debug(" ====Validating===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid = valid_batch[0].to(
                    self.device), valid_batch[1].to(self.device)
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                labels_valid = labels_valid.view(-1, 1)
                data_valid = data_valid[:(
                    N * D) // self.frame_depth * self.frame_depth]
                labels_valid = labels_valid[:(
                    N * D) // self.frame_depth * self.frame_depth]
                pred_ppg_valid = self.model(data_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        logging.debug(" ====Testing===")
        test_step = 0
        test_loss = []
        self.model.eval()
        with torch.no_grad():
            for test_idx, test_batch in enumerate(data_loader["test"]):
                data_test, labels_test = test_batch[0].to(
                    self.device), test_batch[1].to(self.device)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(
                    N * D) // self.frame_depth * self.frame_depth]
                labels_test = labels_test[:(
                    N * D) // self.frame_depth * self.frame_depth]
                pred_ppg_test = self.model(data_test)
                loss = self.criterion(pred_ppg_test, labels_test)
                test_loss.append(loss.item())
                self.twriter.add_scalar("test_loss", scalar_value=float(
                    loss), global_step=test_step)
                test_step += 1
        return np.mean(test_loss)

    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.model.state_dict(), os.path.join(
            self.model_dir, self.model_file_name))
