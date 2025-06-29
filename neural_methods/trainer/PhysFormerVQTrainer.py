from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.loss.PhysFormerLossComputer import TorchLossComputer
from neural_methods.model.PhysFormer_VQ import PhysFormerVQ
from .PhysFormerTrainer import PhysFormerTrainer


class PhysFormerVQTrainer(PhysFormerTrainer):
    """Trainer for PhysFormer with vector quantization."""

    def __init__(self, config, data_loader):
        super().__init__(config, data_loader)

        if config.TOOLBOX_MODE == "train_and_test":
            resize_h = config.TRAIN.DATA.PREPROCESS.RESIZE.H
            resize_w = config.TRAIN.DATA.PREPROCESS.RESIZE.W
        elif config.TOOLBOX_MODE == "only_test":
            resize_h = config.TRAIN.DATA.PREPROCESS.RESIZE.H
            resize_w = config.TRAIN.DATA.PREPROCESS.RESIZE.W
        else:
            raise ValueError("PhysFormerVQ trainer initialized in incorrect toolbox mode!")

        self.model = PhysFormerVQ(
            image_size=(self.chunk_len, resize_h, resize_w),
            patches=(self.patch_size,) * 3,
            dim=self.dim,
            ff_dim=self.ff_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            theta=self.theta,
        ).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        if config.TOOLBOX_MODE == "train_and_test":
            self.optimizer = optim.Adam(self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0.00005)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)

    def train(self, data_loader):
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        a_start = 1.0
        b_start = 1.0
        exp_b = 1.0

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        for epoch in range(self.max_epoch_num):
            print("")
            print(f"====Training Epoch: {epoch}====")
            loss_rPPG_avg = []
            loss_peak_avg = []
            loss_kl_avg_test = []
            loss_hr_mae = []
            vq_losses = []

            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                hr = torch.tensor([self.get_hr(i) for i in batch[1]]).float().to(self.device)
                data, label = batch[0].float().to(self.device), batch[1].float().to(self.device)

                self.optimizer.zero_grad()

                gra_sharp = 2.0
                rPPG, vq_loss, _ = self.model(data, gra_sharp)
                rPPG = (rPPG - torch.mean(rPPG, axis=-1).view(-1, 1)) / torch.std(rPPG, axis=-1).view(-1, 1)
                loss_rPPG = self.criterion_Pearson(rPPG, label)

                fre_loss = 0.0
                kl_loss = 0.0
                train_mae = 0.0
                for bb in range(data.shape[0]):
                    loss_distribution_kl, fre_loss_temp, train_mae_temp = TorchLossComputer.cross_entropy_power_spectrum_DLDL_softmax2(
                        rPPG[bb],
                        hr[bb],
                        self.frame_rate,
                        std=1.0,
                    )
                    fre_loss = fre_loss + fre_loss_temp
                    kl_loss = kl_loss + loss_distribution_kl
                    train_mae = train_mae + train_mae_temp
                fre_loss /= data.shape[0]
                kl_loss /= data.shape[0]
                train_mae /= data.shape[0]

                if epoch > 10:
                    a = 0.05
                    b = 5.0
                else:
                    a = a_start
                    b = b_start * math.pow(exp_b, epoch / 10.0)

                loss = a * loss_rPPG + b * (fre_loss + kl_loss) + vq_loss
                loss.backward()
                self.optimizer.step()

                loss_rPPG_avg.append(float(loss_rPPG.data))
                loss_peak_avg.append(float(fre_loss.data))
                loss_kl_avg_test.append(float(kl_loss.data))
                loss_hr_mae.append(float(train_mae))
                vq_losses.append(float(vq_loss.data))
                if idx % 100 == 99:
                    print(
                        f"\nepoch:{epoch}, batch:{idx + 1}, total:{len(data_loader['train']) // self.batch_size}, "
                        f"lr:0.0001, sharp:{gra_sharp:.3f}, a:{a:.3f}, NegPearson:{np.mean(loss_rPPG_avg[-2000:]):.4f}, "
                        f"\nb:{b:.3f}, kl:{np.mean(loss_kl_avg_test[-2000:]):.3f}, fre_CEloss:{np.mean(loss_peak_avg[-2000:]):.3f}, "
                        f"hr_mae:{np.mean(loss_hr_mae[-2000:]):.3f}, vq:{np.mean(vq_losses[-2000:]):.3f}"
                    )

            lrs.append(self.scheduler.get_last_lr())
            mean_training_losses.append(np.mean(loss_rPPG_avg))
            self.save_model(epoch)
            self.scheduler.step()
            self.model.eval()

            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print(f"Validation RMSE:{valid_loss:.3f}")
                if self.min_valid_loss is None or (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print(f"Update best model! Best epoch: {self.best_epoch}")
        if not self.config.TEST.USE_LAST_EPOCH:
            print(f"best trained epoch: {self.best_epoch}, min_val_loss: {self.min_valid_loss}")
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)
