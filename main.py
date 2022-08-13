""" The main function of rPPG deep learning pipeline.

TODO: Adds detailed description for models and datasets supported.
An end-to-end training pipleine for neural network methods.
  Typical usage example:


  python main_neural_method.py --config_file configs/COHFACE_TSCAN_BASIC.yaml --data_path "G:\\COHFACE"
"""
import argparse
from config import get_config
from torch.utils.data import DataLoader
from dataset import data_loader
from neural_methods import trainer
import torch
import random
import numpy as np
import time
from signal_methods.signal_predictor import signal_predict

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/UBFC_SIGNAL.yaml", type=str, help="The name of the model.")
    # Neural Method Sample YAMSL LIST:
    #   SCAMPS_SCAMPS_UBFC_TSCAN_BASIC.yaml
    #   SCAMPS_SCAMPS_UBFC_DEEPPHYS_BASIC.yaml
    #   SCAMPS_SCAMPS_UBFC_PHYSNET_BASIC.yaml
    #   SCAMPS_SCAMPS_PURE_DEEPPHYS_BASIC.yaml
    #   SCAMPS_SCAMPS_PURE_TSCAN_BASIC.yaml
    #   SCAMPS_SCAMPS_PURE_PHYSNET_BASIC.yaml
    #   PURE_PURE_UBFC_TSCAN_BASIC.yaml
    #   PURE_PURE_UBFC_DEEPPHYS_BASIC.yaml
    #   PURE_PURE_UBFC_PHYSNET_BASIC.yaml
    #   UBFC_UBFC_PURE_TSCAN_BASIC.yaml
    #   UBFC_UBFC_PURE_DEEPPHYS_BASIC.yaml
    #   UBFC_UBFC_PURE_PHYSNET_BASIC.yaml
    # Signal Method Sample YAMSL LIST:
    #   PURE_SIGNAL.yaml
    #   UBFC_SIGNAL.yaml
    return parser


def train_and_test(config, data_loader):
    """Trains the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.train(data_loader)
    model_trainer.test(data_loader)

def test(config, data_loader):
    """Tests the model."""
    if config.MODEL.NAME == "Physnet":
        model_trainer = trainer.PhysnetTrainer.PhysnetTrainer(config)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(config)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(config)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.test(data_loader)

def signal_method_inference(config, data_loader):
    if "pos" in config.SIGNAL.METHOD:
        signal_predict(config, data_loader, "pos")
    if "chrome" in config.SIGNAL.METHOD:
        signal_predict(config, data_loader, "chrome")
    if "ica" in config.SIGNAL.METHOD:
        signal_predict(config, data_loader, "ica")
    if "SSR" in config.SIGNAL.METHOD:
        signal_predict(config, data_loader, "SSR")
    if "LGI" in config.SIGNAL.METHOD:
        signal_predict(config, data_loader, "LGI")
    if "CHROM" in config.SIGNAL.METHOD:
        signal_predict(config, data_loader, "CHROM")
    if "POS2" in config.SIGNAL.METHOD:
        signal_predict(config, data_loader, "POS2")
    if "PBV" in config.SIGNAL.METHOD:
        signal_predict(config, data_loader, "PBV")
    if "PCA" in config.SIGNAL.METHOD:
        signal_predict(config, data_loader, "PCA")
    if "GREEN" in config.SIGNAL.METHOD:
        signal_predict(config, data_loader, "GREEN")
    if "OMIT" in config.SIGNAL.METHOD:
        signal_predict(config, data_loader, "OMIT")
    if "ICA2" in config.SIGNAL.METHOD:
        signal_predict(config, data_loader, "ICA2")
    else:
        raise ValueError("Not supported signal method!")

if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    print(config)

    data_loader_dict = dict()
    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test":
        # neural method dataloader
        # train_loader
        if config.TRAIN.DATA.DATASET == "COHFACE":
            train_loader = data_loader.COHFACELoader.COHFACELoader
        elif config.TRAIN.DATA.DATASET == "UBFC":
            train_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.TRAIN.DATA.DATASET == "PURE":
            train_loader = data_loader.PURELoader.PURELoader
        elif config.TRAIN.DATA.DATASET == "SYNTHETICS":
            train_loader = data_loader.SyntheticsLoader.SyntheticsLoader
        else:
            raise ValueError(
                "Unsupported dataset! Currently supporting COHFACE, UBFC and PURE.")

        # valid_loader
        if config.VALID.DATA.DATASET == "COHFACE":
            valid_loader = data_loader.COHFACELoader.COHFACELoader
        elif config.VALID.DATA.DATASET == "UBFC":
            valid_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.VALID.DATA.DATASET == "PURE":
            valid_loader = data_loader.PURELoader.PURELoader
        elif config.VALID.DATA.DATASET == "SYNTHETICS":
            valid_loader = data_loader.SyntheticsLoader.SyntheticsLoader
        else:
            raise ValueError(
                "Unsupported dataset! Currently supporting COHFACE, UBFC and PURE.")

        # test_loader
        if config.TEST.DATA.DATASET == "COHFACE":
            test_loader = data_loader.COHFACELoader.COHFACELoader
        elif config.TEST.DATA.DATASET == "UBFC":
            test_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.TEST.DATA.DATASET == "PURE":
            test_loader = data_loader.PURELoader.PURELoader
        elif config.TEST.DATA.DATASET == "SYNTHETICS":
            test_loader = data_loader.SyntheticsLoader.SyntheticsLoader
        else:
            raise ValueError(
                "Unsupported dataset! Currently supporting COHFACE, UBFC and PURE.")
        if config.TRAIN.DATA.DATA_PATH:
            train_data_loader = train_loader(
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA)
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=16,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g
            )
        else:
            data_loader_dict['train'] = None
        if config.TRAIN.DATA.DATA_PATH:
            valid_data = valid_loader(
                name="valid",
                data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA)
            data_loader_dict["valid"] = DataLoader(
                dataset=valid_data,
                num_workers=16,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=g
            )
        else:
            data_loader_dict['valid'] = None

        if config.TEST.DATA.DATA_PATH:
            test_data = test_loader(
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA)
            data_loader_dict["test"] = DataLoader(
                dataset=test_data,
                num_workers=16,
                batch_size=config.INFERENCE.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=g
            )
        else:
            data_loader_dict['test'] = None
    elif config.TOOLBOX_MODE == "signal_method":
        # signal method dataloader
        if config.SIGNAL.DATA.DATASET == "COHFACE":
            signal_loader = data_loader.COHFACELoader.COHFACELoader
        elif config.SIGNAL.DATA.DATASET == "UBFC":
            signal_loader = data_loader.UBFCLoader.UBFCLoader
        elif config.SIGNAL.DATA.DATASET == "PURE":
            signal_loader = data_loader.PURELoader.PURELoader
        elif config.SIGNAL.DATA.DATASET == "SYNTHETICS":
            signal_loader = data_loader.SyntheticsLoader.SyntheticsLoader
        else:
            raise ValueError(
                "Unsupported dataset! Currently supporting COHFACE, UBFC and PURE.")
        signal_data = signal_loader(
            name="signal",
            data_path=config.SIGNAL.DATA.DATA_PATH,
            config_data=config.SIGNAL.DATA)
        data_loader_dict["signal"] = DataLoader(
            dataset=signal_data,
            num_workers=16,
            batch_size=1,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g
        )
    else:
        raise ValueError("Unsupported toolbox_mode! Currently support train_and_test or only_test or signal_method.")

    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "signal_method":
        signal_method_inference(config,data_loader_dict)
    else:
        print("TOOLBOX_MODE only support train_and_test or only_test !")
