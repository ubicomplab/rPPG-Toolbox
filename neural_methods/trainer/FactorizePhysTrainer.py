"""
FactorizePhys: Matrix Factorization for Multidimensional Attention in Remote Physiological Sensing
NeurIPS 2024
Jitesh Joshi, Sos S. Agaian, and Youngjun Cho
"""

import os
import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys
from neural_methods.model.FactorizePhys.FactorizePhysBig import FactorizePhysBig
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class FactorizePhysTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.dropout_rate = config.MODEL.DROP_RATE
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        if torch.cuda.is_available() and config.NUM_OF_GPU_TRAIN > 0:
            dev_list = [int(d) for d in config.DEVICE.replace("cuda:", "").split(",")]
            self.device = torch.device(dev_list[0])     #currently toolbox only supports 1 GPU
            self.num_of_gpu = 1     #config.NUM_OF_GPU_TRAIN  # set number of used GPUs
        else:
            self.device = torch.device("cpu")  # if no GPUs set device is CPU
            self.num_of_gpu = 0  # no GPUs used

        frames = self.config.MODEL.FactorizePhys.FRAME_NUM
        in_channels = self.config.MODEL.FactorizePhys.CHANNELS
        model_type = self.config.MODEL.FactorizePhys.TYPE
        model_type = model_type.lower()

        md_config = {}
        md_config["FRAME_NUM"] = self.config.MODEL.FactorizePhys.FRAME_NUM
        md_config["MD_TYPE"] = self.config.MODEL.FactorizePhys.MD_TYPE
        md_config["MD_FSAM"] = self.config.MODEL.FactorizePhys.MD_FSAM
        md_config["MD_TRANSFORM"] = self.config.MODEL.FactorizePhys.MD_TRANSFORM
        md_config["MD_S"] = self.config.MODEL.FactorizePhys.MD_S
        md_config["MD_R"] = self.config.MODEL.FactorizePhys.MD_R
        md_config["MD_STEPS"] = self.config.MODEL.FactorizePhys.MD_STEPS
        md_config["MD_INFERENCE"] = self.config.MODEL.FactorizePhys.MD_INFERENCE
        md_config["MD_RESIDUAL"] = self.config.MODEL.FactorizePhys.MD_RESIDUAL

        self.md_infer = self.config.MODEL.FactorizePhys.MD_INFERENCE
        self.use_fsam = self.config.MODEL.FactorizePhys.MD_FSAM

        if model_type == "standard":
            self.model = FactorizePhys(frames=frames, md_config=md_config, in_channels=in_channels,
                                    dropout=self.dropout_rate, device=self.device)  # [3, T, 72,72]
        elif model_type == "big":
            self.model = FactorizePhysBig(frames=frames, md_config=md_config, in_channels=in_channels,
                                       dropout=self.dropout_rate, device=self.device)  # [3, T, 144,144]
        else:
            print("Unexpected model type specified. Should be standard or big, but specified:", model_type)
            exit()

        if torch.cuda.device_count() > 0 and self.num_of_gpu > 0:  # distribute model across GPUs
            self.model = torch.nn.DataParallel(self.model, device_ids=[self.device])  # data parallel model
        else:
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        if self.config.TOOLBOX_MODE == "train_and_test" or self.config.TOOLBOX_MODE == "only_train":
            self.num_train_batches = len(data_loader["train"])
            self.criterion = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.config.TRAIN.LR, epochs=self.config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif self.config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("FactorizePhys trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        mean_appx_error = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            appx_error_list = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                
                data = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                
                if len(labels.shape) > 2:
                    labels = labels[..., 0]     # Compatibility wigth multi-signal labelled data
                labels = (labels - torch.mean(labels)) / torch.std(labels)  # normalize
                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, max(self.num_of_gpu, 1), 1, 1)
                data = torch.cat((data, last_frame), 2)

                # last_sample = torch.unsqueeze(labels[-1, :], 0).repeat(max(self.num_of_gpu, 1), 1)
                # labels = torch.cat((labels, last_sample), 0)
                # labels = torch.diff(labels, dim=0)
                # labels = labels/ torch.std(labels)  # normalize
                # labels[torch.isnan(labels)] = 0

                self.optimizer.zero_grad()
                if self.model.training and self.use_fsam:
                    pred_ppg, vox_embed, factorized_embed, appx_error = self.model(data)
                else:
                    pred_ppg, vox_embed = self.model(data)
                
                pred_ppg = (pred_ppg - torch.mean(pred_ppg)) / torch.std(pred_ppg)  # normalize

                loss = self.criterion(pred_ppg, labels)
                
                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                if self.use_fsam:
                    appx_error_list.append(appx_error.item())

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                
                if self.use_fsam:
                    tbar.set_postfix({"appx_error": appx_error.item()}, loss=loss.item())
                else:
                    tbar.set_postfix(loss=loss.item())

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))
            if self.use_fsam:
                mean_appx_error.append(np.mean(appx_error_list))
                print("Mean train loss: {}, Mean appx error: {}".format(
                    np.mean(train_loss), np.mean(appx_error_list)))
            else:
                print("Mean train loss: {}".format(np.mean(train_loss)))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")

                data, labels = valid_batch[0].to(self.device), valid_batch[1].to(self.device)
                if len(labels.shape) > 2:
                    labels = labels[..., 0]     # Compatibility wigth multi-signal labelled data
                labels = (labels - torch.mean(labels)) / torch.std(labels)  # normalize

                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, max(self.num_of_gpu, 1), 1, 1)
                data = torch.cat((data, last_frame), 2)

                # last_sample = torch.unsqueeze(labels[-1, :], 0).repeat(max(self.num_of_gpu, 1), 1)
                # labels = torch.cat((labels, last_sample), 0)
                # labels = torch.diff(labels, dim=0)
                # labels = labels/ torch.std(labels)  # normalize
                # labels[torch.isnan(labels)] = 0

                if self.md_infer and self.use_fsam:
                    pred_ppg, vox_embed, factorized_embed, appx_error = self.model(data)
                else:
                    pred_ppg, vox_embed = self.model(data)
                pred_ppg = (pred_ppg - torch.mean(pred_ppg)) / torch.std(pred_ppg)  # normalize
                loss = self.criterion(pred_ppg, labels)

                valid_loss.append(loss.item())
                valid_step += 1
                # vbar.set_postfix(loss=loss.item())
                if self.md_infer and self.use_fsam:
                    vbar.set_postfix({"appx_error": appx_error.item()}, loss=loss.item())
                else:
                    vbar.set_postfix(loss=loss.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH, map_location=self.device), strict=False)
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path, map_location=self.device), strict=False)
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device), strict=False)

        self.model = self.model.to(self.device)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, labels_test = test_batch[0].to(self.device), test_batch[1].to(self.device)

                if len(labels_test.shape) > 2:
                    labels_test = labels_test[..., 0]     # Compatibility wigth multi-signal labelled data
                labels_test = (labels_test - torch.mean(labels_test)) / torch.std(labels_test)  # normalize

                last_frame = torch.unsqueeze(data[:, :, -1, :, :], 2).repeat(1, 1, max(self.num_of_gpu, 1), 1, 1)
                data = torch.cat((data, last_frame), 2)

                # last_sample = torch.unsqueeze(labels_test[-1, :], 0).repeat(max(self.num_of_gpu, 1), 1)
                # labels_test = torch.cat((labels_test, last_sample), 0)
                # labels_test = torch.diff(labels_test, dim=0)
                # labels_test = labels_test/ torch.std(labels_test)  # normalize
                # labels_test[torch.isnan(labels_test)] = 0

                if self.md_infer and self.use_fsam:
                    pred_ppg_test, vox_embed, factorized_embed, appx_error = self.model(data)
                else:
                    pred_ppg_test, vox_embed = self.model(data)
                pred_ppg_test = (pred_ppg_test - torch.mean(pred_ppg_test)) / torch.std(pred_ppg_test)  # normalize

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    labels_test = labels_test.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = labels_test[idx]


        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs 
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
