"""Trainer for EfficientPhys."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from metrics.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.EfficientPhys import EfficientPhys
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class EfficientPhysTrainer(BaseTrainer):

    def __init__(self, config):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.frame_depth = config.MODEL.EFFICIENTPHYS.FRAME_DEPTH
        self.model = EfficientPhys(frame_depth=self.frame_depth, img_size=config.TRAIN.DATA.PREPROCESS.H).to(
            self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        self.criterion = torch.nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu * self.frame_depth
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.best_epoch = 0

    def train(self, data_loader):
        """ TODO:Docstring"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        min_valid_loss = 1
        for epoch in range(self.max_epoch_num):
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(
                    self.device), batch[1].to(self.device)
                N, D, C, H, W = data.shape
                data = data.view(N * D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                data = torch.cat((data, last_frame), 0)
                labels = labels[:(N * D) // self.base_len * self.base_len]
                self.optimizer.zero_grad()
                pred_ppg = self.model(data)
                loss = self.criterion(pred_ppg, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                tbar.set_postfix(loss=loss.item())
            valid_loss = self.valid(data_loader)
            self.save_model(epoch)
            print('validation loss: ', valid_loss)
            if (valid_loss < min_valid_loss) or (valid_loss < 0):
                min_valid_loss = valid_loss
                self.best_epoch = epoch
                print("Update best model! Best epoch: {}".format(self.best_epoch))
                self.save_model(epoch)
        print("best trained epoch:{}, min_val_loss:{}".format(self.best_epoch, min_valid_loss))

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print("===Validating===")
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
                data_valid = data_valid[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data_valid[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                data_valid = torch.cat((data_valid, last_frame), 0)
                labels_valid = labels_valid[:(N * D) // self.base_len * self.base_len]
                pred_ppg_valid = self.model(data_valid)
                loss = self.criterion(pred_ppg_valid, labels_valid)
                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        print("===Testing===")
        predictions = dict()
        labels = dict()
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            best_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
            print("Testing uses non-pretrained model!")
            print(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                # Add one more frame for EfficientPhys since it does torch.diff for the input
                last_frame = torch.unsqueeze(data_test[-1, :, :, :], 0).repeat(self.num_of_gpu, 1, 1, 1)
                data_test = torch.cat((data_test, last_frame), 0)
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                pred_ppg_test = self.model(data_test)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels[subj_index][sort_index] = labels_test[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        calculate_metrics(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
