"""Trainer for TSCAN, but also applies to 3D-CAN, Hybrid-CAN, and DeepPhys."""

from neural_methods.trainer.BaseTrainer import BaseTrainer
import torch
from neural_methods.model.TS_CAN import TSCAN
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import logging
from metrics.metrics import calculate_metrics
from collections import OrderedDict


class TscanTrainer(BaseTrainer):

    def __init__(self, config):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.frame_depth = config.MODEL.TSCAN.FRAME_DEPTH
        self.model = TSCAN(frame_depth=self.frame_depth, img_size=config.TRAIN.DATA.PREPROCESS.H).to(self.device)
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
        self.config = config
        self.best_epoch = 0

    def train(self, data_loader):
        """ TODO:Docstring"""
        if data_loader["train"] is None:
            assert ValueError("No data for train")
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
                data = data.view(N*D, C, H, W)
                labels = labels.view(-1, 1)
                data = data[:(N*D)//self.base_len*self.base_len]
                labels = labels[:(N*D)//self.base_len*self.base_len]
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
            if(valid_loss < min_valid_loss) or (valid_loss < 0):
                min_valid_loss = valid_loss
                self.best_epoch = epoch
                print("Update best model! Best epoch: {}".format(self.best_epoch))
                self.save_model(epoch)
        print("best trained epoch:{}, min_val_loss:{}".format(self.best_epoch,min_valid_loss))


    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            assert ValueError("No data for valid")
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
            assert ValueError("No data for test")
        config = self.config
        print("===Testing===")
        predictions = dict()
        labels = dict()
        if config.TRAIN_OR_TEST == "only_test":
            self.model.load_state_dict(torch.load(config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            best_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
            print("Testing uses non-pretrained model!")
            print(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))
        self.model = self.model.to(config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(
                    config.DEVICE), test_batch[1].to(config.DEVICE)
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                labels_test = labels_test.view(-1, 1)
                data_test = data_test[:(N * D) // self.base_len * self.base_len]
                labels_test = labels_test[:(N * D) // self.base_len * self.base_len]
                pred_ppg_test = self.model(data_test)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx * config.TEST.DATA.PREPROCESS.CLIP_LENGTH:(
                                                    idx + 1) * config.TEST.DATA.PREPROCESS.CLIP_LENGTH]
                    labels[subj_index][sort_index] = labels_test[idx * config.TEST.DATA.PREPROCESS.CLIP_LENGTH:(
                                                    idx + 1) * config.TEST.DATA.PREPROCESS.CLIP_LENGTH]

        calculate_metrics(predictions, labels, config)


    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
