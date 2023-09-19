import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import pickle


class BaseTrainer:
    @staticmethod
    def add_trainer_args(parser):
        """Adds arguments to Paser for training process"""
        parser.add_argument('--lr', default=None, type=float)
        parser.add_argument('--model_file_name', default=None, type=float)
        return parser

    def __init__(self):
        pass

    def train(self, data_loader):
        pass

    def valid(self, data_loader):
        pass

    def test(self):
        pass

    def save_test_outputs(self, predictions, labels, config):
    
        output_dir = config.TEST.OUTPUT_SAVE_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Filename ID to be used in any results files (e.g., Bland-Altman plots) that get saved
        if config.TOOLBOX_MODE == 'train_and_test':
            filename_id = self.model_file_name
        elif config.TOOLBOX_MODE == 'only_test':
            model_file_root = config.INFERENCE.MODEL_PATH.split("/")[-1].split(".pth")[0]
            filename_id = model_file_root + "_" + config.TEST.DATA.DATASET
        else:
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        output_path = os.path.join(output_dir, filename_id + '_outputs.pickle')

        data = dict()
        data['predictions'] = predictions
        data['labels'] = labels
        data['label_type'] = config.TEST.DATA.PREPROCESS.LABEL_TYPE
        data['fs'] = config.TEST.DATA.FS

        with open(output_path, 'wb') as handle: # save out frame dict pickle file
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Saving outputs to:', output_path)
