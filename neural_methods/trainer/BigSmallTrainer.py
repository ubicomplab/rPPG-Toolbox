"""Trainer for BigSmall Multitask Models"""

# Training / Eval Imports 
import torch
import torch.optim as optim
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods import loss
from neural_methods.model.Final_Models import BigSmall
from multitask_eval.metrics import calculate_bvp_metrics, calculate_resp_metrics, calculate_au_metrics

# Other Imports
from collections import OrderedDict
import numpy as np
import os
from tqdm import tqdm


class BigSmallTrainer(BaseTrainer):

    def define_model(self, config):

        # BIG SMALL SLOW FAST 
        model = BigSmallSlowFastWTSM(out_size=len(config.DATA.TRAIN.LABELS.USED_LABELS), n_segment=3)

        if self.using_TSM:
            self.frame_depth = 3 # 3 # default for TSCAN is 10 - consider changing later...
            self.base_len = self.num_of_gpu * self.frame_depth 
            print("USING TIME SHIFT MODULE LOGIC")  

        return model


    def format_data_shape(self, data, labels):
        # reshape big data
        data_big = data[0]
        N, D, C, H, W = data_big.shape
        data_big = data_big.view(N * D, C, H, W)

        # reshape small data
        data_small = data[1]
        N, D, C, H, W = data_small.shape
        data_small = data_small.view(N * D, C, H, W)

        # reshape labels 
        if len(labels.shape) != 3: # this training format requires labels that are of shape N_label, D_label, C_label
            labels = torch.unsqueeze(labels, dim=-1)
        N_label, D_label, C_label = labels.shape
        labels = labels.view(N_label * D_label, C_label)

        # TODO If using TSM modules - reshape for GPU - change how this is used...
        if self.using_TSM:
            data_big = data_big[:(N * D) // self.base_len * self.base_len]
            data_small = data_small[:(N * D) // self.base_len * self.base_len]
            labels = labels[:(N * D) // self.base_len * self.base_len]

        data[0] = data_big
        data[1] = data_small
        labels = torch.unsqueeze(labels, dim=-1)

        return data, labels



    def send_data_to_device(self, data, labels):
        big_data = data[0].to(self.device)
        small_data = data[1].to(self.device)
        labels = labels.to(self.device)
        data = (big_data, small_data)
        return data, labels



    # TODO add this to save model so data parallel is NOT innate
    def remove_data_parallel(self, old_state_dict):
        new_state_dict = OrderedDict()

        for k, v in old_state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        
        return new_state_dict



    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
        print('')



    def reform_data_from_dict(self, data, flatten):
        sort_data = sorted(data.items(), key=lambda x: x[0])
        sort_data = [i[1] for i in sort_data]
        sort_data = torch.cat(sort_data, dim=0)

        if flatten:
            sort_data = np.reshape(sort_data.cpu(), (-1))
        else:
            sort_data = np.array(sort_data.cpu())

        return sort_data



    def reform_preds_labels(self, predictions, labels, flatten=True):
        for index in predictions.keys():
            predictions[index] = self.reform_data_from_dict(predictions[index], flatten=flatten)
            labels[index] = self.reform_data_from_dict(labels[index], flatten=flatten)

        return predictions, labels



    def __init__(self, config, data_loader):

        print('')
        print('Init BigSmall Multitask Trainer')
        print('')

        self.config = config # save config file

        # SET UP GPU COMPUTE DEVICE (GPU OR CPU)
        if torch.cuda.is_available() and config.NUM_OF_GPU_TRAIN > 0:
            self.device = torch.device(config.DEVICE) # set device to primary GPU
            self.num_of_gpu = config.NUM_OF_GPU_TRAIN # set number of used GPUs
        else:
            self.device = "cpu" # if no GPUs set device is CPU
            self.num_of_gpu = 0 # no GPUs used

        # DEFINING MODEL
        self.using_TSM = True
        self.model = self.define_model(config) # define the model

        if torch.cuda.device_count() > 1 and config.NUM_OF_GPU_TRAIN > 1: # distribute model across GPUs
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN))) # data parallel model

        self.model = self.model.to(self.device) # send model to primary GPU

        # TRAINING PARAMETERS
        self.batch_size = config.MODEL_SPECS.TRAIN.BATCH_SIZE
        self.max_epoch_num = config.MODEL_SPECS.TRAIN.EPOCHS
        self.LR = config.MODEL_SPECS.TRAIN.LR
        self.num_train_batches = len(data_loader["train"])

        # Set Loss and Optimizer
        self.criterionAU = loss.loss_utils.set_loss(loss_name='BCEWithLogitsBP4DAU', device=self.device)
        self.criterionBVP = loss.loss_utils.set_loss(loss_name='MSE', device=self.device)
        self.criterionRESP = loss.loss_utils.set_loss(loss_name='MSE', device=self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.LR, weight_decay=0)
        
        # MODEL INFO (SAVED MODEL DIR, CHUNK LEN, BEST EPOCH)
        self.model_dir = config.MODEL_SPECS.MODEL.MODEL_DIR
        self.model_file_name = config.MODEL_SPECS.TRAIN.MODEL_FILE_NAME
        self.chunk_len = config.DATA.TRAIN.PREPROCESS.CHUNK_LENGTH

        self.run_validation = self.config.MODEL_SPECS.VALID.RUN_VALIDATION
        self.model_to_use =  self.config.MODEL_SPECS.TEST.MODEL_TO_USE # either 'last_epoch' or 'best_epoch'
        self.used_epoch = 0

        # SAVED OUTPUT LOGGING INFO
        self.save_data = config.SAVE_DATA.SAVE_DATA
        self.save_train = config.SAVE_DATA.SAVE_TRAIN
        self.save_test = config.SAVE_DATA.SAVE_TEST
        self.save_metrics = config.SAVE_DATA.SAVE_METRICS

        self.save_data_path = config.SAVE_DATA.PATH
        self.data_dict = dict() # dictionary to save
        self.data_dict['config'] = self.config # save config file

        # INDICES CORRESPONDING TO USED LABELS 
        train_au_label_list = [label for label in config.DATA.TRAIN.LABELS.USED_LABELS if 'AU' in label]
        valid_au_label_list = [label for label in config.DATA.VALID.LABELS.USED_LABELS if 'AU' in label]
        test_au_label_list = [label for label in config.DATA.TEST.LABELS.USED_LABELS if 'AU' in label]

        train_bvp_label_list = [label for label in config.DATA.TRAIN.LABELS.USED_LABELS if 'bvp' in label]
        valid_bvp_label_list = [label for label in config.DATA.VALID.LABELS.USED_LABELS if 'bvp' in label]
        test_bvp_label_list = [label for label in config.DATA.TEST.LABELS.USED_LABELS if 'bvp' in label]

        train_resp_label_list = [label for label in config.DATA.TRAIN.LABELS.USED_LABELS if 'resp' in label]
        valid_resp_label_list = [label for label in config.DATA.VALID.LABELS.USED_LABELS if 'resp' in label]
        test_resp_label_list = [label for label in config.DATA.TEST.LABELS.USED_LABELS if 'resp' in label]

        self.label_idx_train_au = self.get_label_idxs(config.DATA.TRAIN.LABELS.LABEL_LIST, train_au_label_list)
        self.label_idx_valid_au = self.get_label_idxs(config.DATA.VALID.LABELS.LABEL_LIST, valid_au_label_list)
        self.label_idx_test_au = self.get_label_idxs(config.DATA.TEST.LABELS.LABEL_LIST, test_au_label_list)

        self.label_idx_train_bvp = self.get_label_idxs(config.DATA.TRAIN.LABELS.LABEL_LIST, train_bvp_label_list)
        self.label_idx_valid_bvp = self.get_label_idxs(config.DATA.VALID.LABELS.LABEL_LIST, valid_bvp_label_list)
        self.label_idx_test_bvp = self.get_label_idxs(config.DATA.TEST.LABELS.LABEL_LIST, test_bvp_label_list)

        self.label_idx_train_resp = self.get_label_idxs(config.DATA.TRAIN.LABELS.LABEL_LIST, train_resp_label_list)
        self.label_idx_valid_resp = self.get_label_idxs(config.DATA.VALID.LABELS.LABEL_LIST, valid_resp_label_list)
        self.label_idx_test_resp = self.get_label_idxs(config.DATA.TEST.LABELS.LABEL_LIST, test_resp_label_list)

        print('Used Labels:', config.DATA.TRAIN.LABELS.USED_LABELS)
        print('Training Indices AU:', self.label_idx_train_au)
        print('Training Indices BVP:', self.label_idx_train_bvp)
        print('Training Indices Resp:', self.label_idx_train_resp)
        print('')



    def train(self, data_loader):
        """Model Training"""

        if data_loader["train"] is None:
            raise ValueError("No data for train")

        print('Starting Training Routine')
        print('')

        # Init min validation loss as infinity
        min_valid_loss = np.inf # minimum validation loss

        # ARRAYS TO SAVE (LOSS ARRAYS)
        train_loss_dict = dict()
        train_au_loss_dict = dict()
        train_bvp_loss_dict = dict()
        train_resp_loss_dict = dict()

        val_loss_dict = dict()
        val_au_loss_dict = dict()
        val_bvp_loss_dict = dict()
        val_resp_loss_dict = dict()

        # ITERATE THROUGH EPOCHS
        for epoch in range(self.max_epoch_num):
            print(f"====Training Epoch: {epoch}====")

            # INIT PARAMS FOR TRAINING
            running_loss = 0.0 # tracks avg loss over mini batches of 100
            train_loss = []
            train_au_loss = []
            train_bvp_loss = []
            train_resp_loss = []
            self.model.train() # put model in train mode

            # MODEL TRAINING
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)

                # GATHER AND FORMAT BATCH DATA
                data, labels = batch[0], batch[1]
                data, labels = self.format_data_shape(data, labels)
                data, labels = self.send_data_to_device(data, labels)

                # FOWARD AND BACK PROPOGATE THROUGH MODEL
                self.optimizer.zero_grad()
                au_out, bvp_out, resp_out = self.model(data)
                au_loss = self.criterionAU(au_out, labels[:, self.label_idx_train_au, 0]) # au loss 
                bvp_loss = self.criterionBVP(bvp_out, labels[:, self.label_idx_train_bvp, 0]) # bvp loss
                resp_loss =  self.criterionRESP(resp_out, labels[:, self.label_idx_train_resp, 0]) # resp loss 
                loss = au_loss  + bvp_loss + resp_loss # sum losses 
                loss.backward()

                self.optimizer.step() # Step the optimizer

                # UPDATE RUNNING LOSS AND PRINTED TERMINAL OUTPUT AND SAVED LOSSES
                train_loss.append(loss.item())
                train_au_loss.append(au_loss.item())
                train_bvp_loss.append(bvp_loss.item())
                train_resp_loss.append(resp_loss.item())

                running_loss += loss.item()
                if idx % 100 == 99: # print every 100 mini-batches
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

                 
                tbar.set_postfix({"loss:": loss.item(), "lr:": self.optimizer.param_groups[0]["lr"]})

            # APPEND EPOCH LOSS LIST TO TRAINING LOSS DICTIONARY
            train_loss_dict[epoch] = train_loss
            train_au_loss_dict[epoch] = train_au_loss
            train_bvp_loss_dict[epoch] = train_bvp_loss
            train_resp_loss_dict[epoch] = train_resp_loss
            
            print('')

            # SAVE MODEL FOR THIS EPOCH
            self.save_model(epoch)

            # VALIDATION (ENABLED)
            if self.run_validation or self.model_to_use == 'best_epoch':

                # Get validation losses
                valid_loss, valid_au_loss, valid_bvp_loss, valid_resp_loss = self.valid(data_loader)
                val_loss_dict[epoch] = valid_loss
                val_au_loss_dict[epoch] = valid_au_loss
                val_bvp_loss_dict[epoch] = valid_bvp_loss
                val_resp_loss_dict[epoch] = valid_resp_loss
                print('validation loss: ', valid_loss)

                # Update used model
                if self.model_to_use == 'best_epoch' and (valid_loss < min_valid_loss):
                    min_valid_loss = valid_loss
                    self.used_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.used_epoch))
                elif self.model_to_use == 'last_epoch':
                    self.used_epoch = epoch
            
            # VALIDATION (NOT ENABLED)
            else: 
                self.used_epoch = epoch

            print('')

        # IF SAVING OUTPUT DATA
        if self.save_data:
            self.data_dict['train_loss'] = train_loss_dict
            self.data_dict['train_au_loss'] = train_au_loss_dict
            self.data_dict['train_bvp_loss'] = train_bvp_loss_dict
            self.data_dict['train_resp_loss'] = train_resp_loss_dict
            self.data_dict['val_loss'] = val_loss_dict 
            self.data_dict['val_au_loss'] = val_au_loss_dict
            self.data_dict['val_bvp_loss'] = val_bvp_loss_dict
            self.data_dict['val_resp_loss'] = val_resp_loss_dict

        # PRINT MODEL TO BE USED FOR TESTING
        print("Used model trained epoch:{}, val_loss:{}".format(self.used_epoch, min_valid_loss))
        print('')



    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""

        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print("===Validating===")

        # INIT PARAMS FOR VALIDATION
        valid_loss = []
        valid_au_loss = []
        valid_bvp_loss = []
        valid_resp_loss = []
        self.model.eval()

        # MODEL VALIDATION
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")

                # GATHER AND FORMAT BATCH DATA
                data, labels = valid_batch[0], valid_batch[1]
                data, labels = self.format_data_shape(data, labels)
                data, labels = self.send_data_to_device(data, labels)

                au_out, bvp_out, resp_out = self.model(data)
                au_loss = self.criterionAU(au_out, labels[:, self.label_idx_valid_au, 0]) # au loss
                bvp_loss = self.criterionBVP(bvp_out, labels[:, self.label_idx_valid_bvp, 0]) # bvp loss
                resp_loss =  self.criterionRESP(resp_out, labels[:, self.label_idx_valid_resp, 0]) # resp loss 
                loss = au_loss + bvp_loss + resp_loss # sum losses

                # APPEND VAL LOSS
                valid_loss.append(loss.item())
                valid_au_loss.append(au_loss.item())
                valid_bvp_loss.append(bvp_loss.item())
                valid_resp_loss.append(resp_loss.item())
                vbar.set_postfix(loss=loss.item())

        valid_loss = np.asarray(valid_loss)
        valid_au_loss = np.asarray(valid_au_loss)
        valid_bvp_loss = np.asarray(valid_bvp_loss)
        valid_resp_loss = np.asarray(valid_resp_loss)
        return np.mean(valid_loss), np.mean(valid_au_loss), np.mean(valid_bvp_loss), np.mean(valid_resp_loss)



    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""

        print("===Testing===")
        print('')

        # SETUP
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        # Change chunk length to be test chunk length
        self.chunk_len = self.config.DATA.TEST.PREPROCESS.CHUNK_LENGTH 

        # ARRAYS TO SAVE (PREDICTIONS AND METRICS ARRAYS)
        preds_dict_au = dict()
        labels_dict_au = dict()
        preds_dict_bvp = dict()
        labels_dict_bvp = dict()
        preds_dict_resp = dict()
        labels_dict_resp = dict()

        # IF ONLY_TEST MODE LOAD PRETRAINED MODEL
        if self.config.TOOLBOX_MODE == "only_test":
            model_path = self.config.MODEL_SPECS.TEST.MODEL_PATH
            print("Testing uses pretrained model!")
            print('Model path:', model_path)
            if not os.path.exists(model_path):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")

        # IF USING MODEL FROM TRAINING
        else:
            model_path = os.path.join(self.model_dir, 
                                           self.model_file_name + '_Epoch' + str(self.used_epoch) + '.pth')
            print("Testing uses non-pretrained model!")
            print('Model path:', model_path)
            if not os.path.exists(model_path):
                raise ValueError("Something went wrong... cant find trained model...")
        print('')
            
        # LOAD ABOVED SPECIFIED MODEL FOR TESTING
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()

        # MODEL TESTING
        with torch.no_grad():
            for _, test_batch in enumerate(data_loader['test']):

                # PROCESSING - ANALYSIS, METRICS, SAVING OUT DATA
                batch_size = test_batch[1].shape[0] # get batch size

                # GATHER AND FORMAT BATCH DATA
                data, labels = test_batch[0], test_batch[1]
                data, labels = self.format_data_shape(data, labels)
                data, labels = self.send_data_to_device(data, labels)

                # Weird dataloader bug is causing the final training batch to be of size 0...
                if labels.shape[0] == 0:
                    continue

                # GET MODEL PREDICTIONS
                au_out, bvp_out, resp_out = self.model(data)
                au_out = torch.sigmoid(au_out) 

                # GATHER AND SLICE LABELS USED FOR TEST DATASET
                TEST_AU = False
                if len(self.label_idx_test_au) > 0: # if test dataset has AU
                    TEST_AU = True
                    labels_au = labels[:, self.label_idx_test_au] 
                else: # if not set whole AU labels array to -1
                    labels_au = np.ones((batch_size, len(self.label_idx_train_au)))
                    labels_au = -1 * labels_au
                    # labels_au = torch.from_numpy(labels_au)

                TEST_BVP = False
                if len(self.label_idx_test_bvp) > 0: # if test dataset has BVP
                    TEST_BVP = True
                    labels_bvp = labels[:, 0] # TODO use bpwave as label for BVP pseudo input
                else: # if not set whole BVP labels array to -1
                    labels_bvp = np.ones((batch_size, len(self.label_idx_train_bvp)))
                    labels_bvp = -1 * labels_bvp
                    # labels_bvp = torch.from_numpy(labels_bvp)

                TEST_RESP = False
                if len(self.label_idx_test_resp) > 0: # if test dataset has BVP
                    TEST_RESP = True
                    labels_resp = labels[:, self.label_idx_test_resp]
                else: # if not set whole BVP labels array to -1
                    labels_resp = np.ones((batch_size, len(self.label_idx_train_resp)))
                    labels_resp = -1 * labels_resp
                    # labels_resp = torch.from_numpy(labels_resp)
                
                # IF TEST PREDICTION DATA TO BE SAVED - MOVE FROM GPU TO CPU
                if self.save_data and self.save_test:
                    au_out = au_out.to('cpu')
                    labels_au = labels_au.to('cpu') 
                    bvp_out = bvp_out.to('cpu')
                    labels_bvp = labels_bvp.to('cpu')
                    resp_out = resp_out.to('cpu')
                    labels_resp = labels_resp.to('cpu')

                # ITERATE THROUGH BATCH, SORT, AND ADD TO CORRECT DICTIONARY
                for idx in range(batch_size):

                    # if the labels are cut off due to TSM dataformating
                    if idx * self.chunk_len >= labels.shape[0] and self.using_TSM:
                        continue 

                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])

                    # add subject to prediction / label arrays
                    if subj_index not in preds_dict_bvp.keys():
                        preds_dict_au[subj_index] = dict()
                        labels_dict_au[subj_index] = dict()
                        preds_dict_bvp[subj_index] = dict()
                        labels_dict_bvp[subj_index] = dict()
                        preds_dict_resp[subj_index] = dict()
                        labels_dict_resp[subj_index] = dict()

                    # append predictions and labels to subject dict
                    preds_dict_au[subj_index][sort_index] = au_out[idx * self.chunk_len:(idx + 1) * self.chunk_len] 
                    labels_dict_au[subj_index][sort_index] = labels_au[idx * self.chunk_len:(idx + 1) * self.chunk_len] 
                    preds_dict_bvp[subj_index][sort_index] = bvp_out[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels_dict_bvp[subj_index][sort_index] = labels_bvp[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    preds_dict_resp[subj_index][sort_index] = resp_out[idx * self.chunk_len:(idx + 1) * self.chunk_len]
                    labels_dict_resp[subj_index][sort_index] = labels_resp[idx * self.chunk_len:(idx + 1) * self.chunk_len]

        # REFORM DATA
        preds_dict_au, labels_dict_au = self.reform_preds_labels(preds_dict_au, labels_dict_au, flatten=False)
        preds_dict_bvp, labels_dict_bvp = self.reform_preds_labels(preds_dict_bvp, labels_dict_bvp)
        preds_dict_resp, labels_dict_resp = self.reform_preds_labels(preds_dict_resp, labels_dict_resp)

        # CALCULATE METRICS ON PREDICTIONS
        if self.config.MODEL_SPECS.TEST.BVP_METRICS: # run metrics, if not empty list
            print('BVP Metrics:')
            bvp_metric_dict = calculate_bvp_metrics(preds_dict_bvp, labels_dict_bvp, self.config)
            if self.save_metrics:
                self.data_dict['bvp_metrics'] = bvp_metric_dict
            print('')

        if self.config.MODEL_SPECS.TEST.RESP_METRICS: # run metrics, if not empty list
            print('Resp Metrics:')
            resp_metric_dict = calculate_resp_metrics(preds_dict_resp, labels_dict_resp, self.config)
            if self.save_metrics:
                self.data_dict['resp_metrics'] = resp_metric_dict
            print('')

        if self.config.MODEL_SPECS.TEST.AU_METRICS: # run metrics, if not empty list
            print('AU Metrics:')
            au_metric_dict = calculate_au_metrics(preds_dict_au, labels_dict_au, self.config)
            if self.save_metrics:
                self.data_dict['au_metrics'] = au_metric_dict
            print('')
        











