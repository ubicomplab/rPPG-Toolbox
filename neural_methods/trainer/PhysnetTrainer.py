# TODO: Docstring
"""A one line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

  Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
from neural_methods.trainer.BaseTrainer import BaseTrainer
import torch
from torch.autograd import Variable
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.loss.NegPearsonLoss import Neg_Pearson
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from metrics.metrics import calculate_metrics
from collections import OrderedDict


class PhysnetTrainer(BaseTrainer):

    def __init__(self, config):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]
        self.loss_model = Neg_Pearson()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=config.TRAIN.LR)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
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
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    Variable(batch[0]).to(torch.float32).to(self.device))
                BVP_label = Variable(batch[1]).to(
                    torch.float32).to(self.device)
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                    torch.std(BVP_label)  # normalize
                loss = self.loss_model(rPPG, BVP_label)
                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch + 1}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                self.optimizer.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item())
            valid_loss = self.valid(data_loader)
            self.save_model(epoch)
            print('validation loss: ', valid_loss)
            if(valid_loss < min_valid_loss) or (valid_loss < 0):
                min_valid_loss = valid_loss
                self.best_epoch = epoch
                print("Update best model! Best epoch: {}".format(self.best_epoch))
                self.save_model(epoch)
        print("best trained epoch:{}, min_val_loss:{}".format(
            self.best_epoch, min_valid_loss))


    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = Variable(valid_batch[1]).to(
                    torch.float32).to(self.device)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    Variable(valid_batch[0]).to(torch.float32).to(self.device))
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                    torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test_(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        config = self.config
        print("===Testing===")
        predictions = dict()
        labels = dict()

        model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=128).to("cuda")  # [3, T, 128,128]
        # model = torch.nn.DataParallel(model, device_ids=list(range(1)))
        person_model_paths = "/data1/acsp/Yuzhe_Zhang/Toolbox_master_2/rPPG-Toolbox/PreTrai" \
                             "nedModels/PURE_SizeW128_SizeH128_ClipLength128_DataTypeStandardi" \
                             "zed_LabelTypeStandardized_Large_boxTrue_Large_size1.5_Dyamic_DetFa" \
                             "lse_det_len180/PURE_PURE_UBFC_physnet.pth_Epoch11.pth"
        model.load_state_dict(torch.load(person_model_paths, map_location=torch.device('cuda')))
        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    config.DEVICE), test_batch[1].to(config.DEVICE)
                pred_ppg_test, _, _, _ = model(data)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    # predictions[subj_index][sort_index] = prediction
                    # labels[subj_index][sort_index] = label
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    # print("p:",predictions[subj_index][sort_index].shape)
                    labels[subj_index][sort_index] = label[idx]
                    # print(labels[subj_index][sort_index].shape)
        calculate_metrics(predictions, labels, config)








    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            assert ValueError("No data for test")
        config = self.config
        print("===Testing===")
        predictions = dict()
        labels = dict()
        if config.TRAIN_OR_TEST == "only_test":
            self.model.load_state_dict(torch.load(config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(config.INFERENCE.MODEL_PATH)
        else:
            best_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
            print("Testing uses non-pretrained model!")
            print(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))
        self.model = self.model.to(config.DEVICE)
        # self.model.eval()
        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    config.DEVICE), test_batch[1].to(config.DEVICE)
                pred_ppg_test, _, _, _ = self.model(data)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    # predictions[subj_index][sort_index] = prediction
                    # labels[subj_index][sort_index] = label
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]
                    # print(labels[subj_index][sort_index].shape)
        calculate_metrics(predictions, labels, config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)