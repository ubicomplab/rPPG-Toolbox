"""Trainer for Physformer.

Based on open-source code from the original PhysFormer authors below:
https://github.com/ZitongYu/PhysFormer/blob/main/train_Physformer_160_VIPL.py

We also thank the PhysBench authors for their open-source code based on the code
of the original authors. Their code below provided a better reference for tuning loss
parameters of interest and utilizing RSME as a validation loss:
https://github.com/KegangWangCCNU/PhysBench/blob/main/benchmark_addition/PhysFormer_pure.ipynb

"""

import os
import numpy as np
import math
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.loss.PhysFormerLossComputer import TorchLossComputer
from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm
from scipy.signal import welch

from collections import OrderedDict


class PhysFormerTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.dropout_rate = config.MODEL.DROP_RATE
        self.patch_size = config.MODEL.PHYSFORMER.PATCH_SIZE
        self.dim = config.MODEL.PHYSFORMER.DIM
        self.ff_dim = config.MODEL.PHYSFORMER.FF_DIM
        self.num_heads = config.MODEL.PHYSFORMER.NUM_HEADS
        self.num_layers = config.MODEL.PHYSFORMER.NUM_LAYERS
        self.theta = config.MODEL.PHYSFORMER.THETA
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN if config.NUM_OF_GPU_TRAIN > 0 else 1
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.frame_rate = config.TRAIN.DATA.FS
        self.config = config 
        self.min_valid_loss = None
        self.best_epoch = 0

        if config.TOOLBOX_MODE == "train_and_test":
            self.model = ViT_ST_ST_Compact3_TDC_gra_sharp(
                image_size=(config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH,config.TRAIN.DATA.PREPROCESS.RESIZE.H,config.TRAIN.DATA.PREPROCESS.RESIZE.W), 
                patches=(self.patch_size,) * 3, dim=self.dim, ff_dim=self.ff_dim, num_heads=self.num_heads, num_layers=self.num_layers, 
                dropout_rate=self.dropout_rate, theta=self.theta).to(self.device)
            
            if config.NUM_OF_GPU_TRAIN > 0:
                self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))

            if self.config.INFERENCE.MODEL_PATH != "":
                self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH, map_location=self.device))
                print("Loaded Checkpoint:", self.config.INFERENCE.MODEL_PATH)

            self.num_train_batches = len(data_loader["train"])
            self.criterion_reg = torch.nn.MSELoss()
            self.criterion_L1loss = torch.nn.L1Loss()
            self.criterion_class = torch.nn.CrossEntropyLoss()
            self.criterion_Pearson = Neg_Pearson()
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0.00005)
            # TODO: In both the PhysFormer repo's training example and other implementations of a PhysFormer trainer, 
            # a step_size that doesn't end up changing the LR always seems to be used. This seems to defeat the point
            # of using StepLR in the first place. Consider investigating and using another approach (e.g., OneCycleLR).
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        elif config.TOOLBOX_MODE == "only_test":
            self.model = ViT_ST_ST_Compact3_TDC_gra_sharp(
                image_size=(config.TEST.DATA.PREPROCESS.CHUNK_LENGTH,config.TEST.DATA.PREPROCESS.RESIZE.H,config.TEST.DATA.PREPROCESS.RESIZE.W), 
                patches=(self.patch_size,) * 3, dim=self.dim, ff_dim=self.ff_dim, num_heads=self.num_heads, num_layers=self.num_layers, 
                dropout_rate=self.dropout_rate, theta=self.theta).to(self.device)
            
            if config.NUM_OF_GPU_TRAIN > 0:
                self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("Physformer trainer initialized in incorrect toolbox mode!")


    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        # a --> Pearson loss; b --> frequency loss
        a_start = 1.0
        b_start = 1.0
        exp_a = 0.5     # Unused
        exp_b = 1.0

        # TODO: Expand tracking and subsequent plotting of these losses for PhysFormer
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            loss_rPPG_avg = []
            loss_peak_avg = []
            loss_kl_avg_test = []
            loss_hr_mae = []

            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                hr = torch.tensor([self.get_hr(i) for i in batch[1]]).float().to(self.device)
                data, label = batch[0].float().to(self.device), batch[1].float().to(self.device)

                self.optimizer.zero_grad()

                gra_sharp = 2.0
                rPPG, _, _, _ = self.model(data, gra_sharp)
                rPPG = (rPPG-torch.mean(rPPG, axis=-1).view(-1, 1))/torch.std(rPPG, axis=-1).view(-1, 1)    # normalize
                loss_rPPG = self.criterion_Pearson(rPPG, label)

                fre_loss = 0.0
                kl_loss = 0.0
                train_mae = 0.0
                for bb in range(data.shape[0]):
                    loss_distribution_kl, \
                    fre_loss_temp, \
                    train_mae_temp = TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(
                        rPPG[bb],
                        hr[bb],
                        self.frame_rate,
                        std=1.0
                    )
                    fre_loss = fre_loss+fre_loss_temp
                    kl_loss = kl_loss+loss_distribution_kl
                    train_mae = train_mae+train_mae_temp
                fre_loss /= data.shape[0]
                kl_loss /= data.shape[0]
                train_mae /= data.shape[0]

                if epoch>10:
                    a = 0.05
                    b = 5.0
                else:
                    a = a_start
                    # exp ascend
                    b = b_start*math.pow(exp_b, epoch/10.0)

                loss = a*loss_rPPG + b*(fre_loss+kl_loss)
                loss.backward()
                self.optimizer.step()

                n = data.size(0)
                loss_rPPG_avg.append(float(loss_rPPG.data))
                loss_peak_avg.append(float(fre_loss.data))
                loss_kl_avg_test.append(float(kl_loss.data))
                loss_hr_mae.append(float(train_mae))
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(f'\nepoch:{epoch}, batch:{idx + 1}, total:{len(data_loader["train"]) // self.batch_size}, '
                        f'lr:0.0001, sharp:{gra_sharp:.3f}, a:{a:.3f}, NegPearson:{np.mean(loss_rPPG_avg[-2000:]):.4f}, '
                        f'\nb:{b:.3f}, kl:{np.mean(loss_kl_avg_test[-2000:]):.3f}, fre_CEloss:{np.mean(loss_peak_avg[-2000:]):.3f}, '
                        f'hr_mae:{np.mean(loss_hr_mae[-2000:]):.3f}')
                    
            # Append the current learning rate to the list
            lrs.append(self.scheduler.get_last_lr())
            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(loss_rPPG_avg))
            self.save_model(epoch)
            self.scheduler.step()
            self.model.eval()

            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print(f'Validation RMSE:{valid_loss:.3f}, batch:{idx+1}')
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
        print(" ====Validating===")
        self.optimizer.zero_grad()
        with torch.no_grad():
            hrs = []
            vbar = tqdm(data_loader["valid"], ncols=80)
            for val_idx, val_batch in enumerate(vbar):
                data, label = val_batch[0].float().to(self.device), val_batch[1].float().to(self.device)
                gra_sharp = 2.0
                rPPG, _, _, _ = self.model(data, gra_sharp)
                rPPG = (rPPG-torch.mean(rPPG, axis=-1).view(-1, 1))/torch.std(rPPG).view(-1, 1)
                for _1, _2 in zip(rPPG, label):
                    hrs.append((self.get_hr(_1.cpu().detach().numpy()), self.get_hr(_2.cpu().detach().numpy())))
            RMSE = np.mean([(i-j)**2 for i, j in hrs])**0.5
        return RMSE

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
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path, map_location=self.device))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                gra_sharp = 2.0
                pred_ppg_test, _, _, _ = self.model(data, gra_sharp)
                
                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()
                
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]

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

    # HR calculation based on ground truth label
    def get_hr(self, y, sr=30, min=30, max=180):
        p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
        return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60
