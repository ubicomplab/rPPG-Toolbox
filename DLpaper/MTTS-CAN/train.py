'''
Training Script for Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement
Author: Xin Liu, Daniel McDuff
'''
# %%
from __future__ import print_function

import argparse
import itertools
import json
import os

import numpy as np
import scipy.io
import tensorflow as tf

from data_generator import DataGenerator
from model import HeartBeat, CAN, CAN_3D, Hybrid_CAN, TS_CAN, MTTS_CAN, \
    MT_Hybrid_CAN, MT_CAN_3D, MT_CAN
from pre_process import get_nframe_video, split_subj, sort_video_list

np.random.seed(100)  # for reproducibility
tf.test.is_gpu_available()
tf.keras.backend.clear_session()
print(tf.__version__)

# %%
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-exp', '--exp_name', type=str,
                    help='experiment name')
parser.add_argument('-i', '--data_dir', type=str, help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='./rPPG-checkpoints',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-a', '--nb_filters1', type=int, default=32,
                    help='number of convolutional filters to use')
parser.add_argument('-b', '--nb_filters2', type=int, default=64,
                    help='number of convolutional filters to use')
parser.add_argument('-c', '--dropout_rate1', type=float, default=0.25,
                    help='dropout rates')
parser.add_argument('-d', '--dropout_rate2', type=float, default=0.5,
                    help='dropout rates')
parser.add_argument('-l', '--lr', type=float, default=1.0,
                            help='learning rate')
parser.add_argument('-e', '--nb_dense', type=int, default=128,
                    help='number of dense units')
parser.add_argument('-f', '--cv_split', type=int, default=0,
                    help='cv_split')
parser.add_argument('-g', '--nb_epoch', type=int, default=24,
                    help='nb_epoch')
parser.add_argument('-t', '--nb_task', type=int, default=12,
                    help='nb_task')
parser.add_argument('-fd', '--frame_depth', type=int, default=10,
                    help='frame_depth for CAN_3D, TS_CAN, Hybrid_CAN')
parser.add_argument('-temp', '--temporal', type=str, default='MTTS_CAN',
                    help='CAN, MT_CAN, CAN_3D, MT_CAN_3D, Hybrid_CAN, \
                    MT_Hybrid_CAN, TS_CAN, MTTS_CAN ')
parser.add_argument('-save', '--save_all', type=int, default=1,
                    help='save all or not')
parser.add_argument('-resp', '--respiration', type=int, default=0,
                    help='train with resp or not')

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

# %% Spliting Data

print('Spliting Data...')
subNum = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27])
taskList = list(range(1, args.nb_task+1))

# %% Training


def train(args, subTrain, subTest, cv_split, img_rows=36, img_cols=36):
    print('================================')
    print('Train...')
    print('subTrain', subTrain)
    print('subTest', subTest)

    input_shape = (img_rows, img_cols, 3)

    path_of_video_tr = sort_video_list(args.data_dir, taskList, subTrain)
    path_of_video_test = sort_video_list(args.data_dir, taskList, subTest)
    path_of_video_tr = list(itertools.chain(*path_of_video_tr))  # Fllaten the list
    path_of_video_test = list(itertools.chain(*path_of_video_test))

    print('sample path: ', path_of_video_tr[0])
    nframe_per_video = get_nframe_video(path_of_video_tr[0])
    print('Trian Length: ', len(path_of_video_tr))
    print('Test Length: ', len(path_of_video_test))
    print('nframe_per_video', nframe_per_video)

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        if strategy.num_replicas_in_sync == 4:
            print("Using 4 GPUs for training")
            if args.temporal == 'CAN' or args.temporal == 'MT_CAN':
                args.batch_size = 32
            elif args.temporal == 'CAN_3D' or args.temporal == 'MT_CAN_3D':
                args.batch_size = 12
            elif args.temporal == 'TS_CAN' or args.temporal == 'MTTS_CAN':
                args.batch_size = 32
            elif args.temporal == 'Hybrid_CAN' or args.temporal == 'MT_Hybrid_CAN':
                args.batch_size = 16
            else:
                raise ValueError('Unsupported Model Type!')
        elif strategy.num_replicas_in_sync == 8:
            print('Using 8 GPUs for training!')
            args.batch_size = args.batch_size * 2
        elif strategy.num_replicas_in_sync == 2:
            args.batch_size = args.batch_size // 2
        else:
            raise Exception('Only supporting 4 GPUs or 8 GPUs now. Please adjust learning rate in the training script!')

        if args.temporal == 'CAN':
            print('Using CAN!')
            model = CAN(args.nb_filters1, args.nb_filters2, input_shape, dropout_rate1=args.dropout_rate1,
                        dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
        elif args.temporal == 'MT_CAN':
            print('Using MT_CAN!')
            model = MT_CAN(args.nb_filters1, args.nb_filters2, input_shape, dropout_rate1=args.dropout_rate1,
                           dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
        elif args.temporal == 'CAN_3D':
            print('Using CAN_3D!')
            input_shape = (img_rows, img_cols, args.frame_depth, 3)
            model = CAN_3D(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
                           dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
        elif args.temporal == 'MT_CAN_3D':
            print('Using MT_CAN_3D!')
            input_shape = (img_rows, img_cols, args.frame_depth, 3)
            model = MT_CAN_3D(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
                              dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2,
                              nb_dense=args.nb_dense)
        elif args.temporal == 'TS_CAN':
            print('Using TS_CAN!')
            input_shape = (img_rows, img_cols, 3)
            model = TS_CAN(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
                           dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
        elif args.temporal == 'MTTS_CAN':
            print('Using MTTS_CAN!')
            input_shape = (img_rows, img_cols, 3)
            model = MTTS_CAN(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape,
                             dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2, nb_dense=args.nb_dense)
        elif args.temporal == 'Hybrid_CAN':
            print('Using Hybrid_CAN!')
            input_shape_motion = (img_rows, img_cols, args.frame_depth, 3)
            input_shape_app = (img_rows, img_cols, 3)
            model = Hybrid_CAN(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape_motion,
                               input_shape_app,
                               dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2,
                               nb_dense=args.nb_dense)
        elif args.temporal == 'MT_Hybrid_CAN':
            print('Using MT_Hybrid_CAN!')
            input_shape_motion = (img_rows, img_cols, args.frame_depth, 3)
            input_shape_app = (img_rows, img_cols, 3)
            model = MT_Hybrid_CAN(args.frame_depth, args.nb_filters1, args.nb_filters2, input_shape_motion,
                                  input_shape_app,
                                  dropout_rate1=args.dropout_rate1, dropout_rate2=args.dropout_rate2,
                                  nb_dense=args.nb_dense)
        else:
            raise ValueError('Unsupported Model Type!')

        optimizer = tf.keras.optimizers.Adadelta(learning_rate=args.lr)
        if args.temporal == 'MTTS_CAN' or args.temporal == 'MT_Hybrid_CAN' or args.temporal == 'MT_CAN_3D' or \
                args.temporal == 'MT_CAN':
            losses = {"output_1": "mean_squared_error", "output_2": "mean_squared_error"}
            loss_weights = {"output_1": 1.0, "output_2": 1.0}
            model.compile(loss=losses, loss_weights=loss_weights, optimizer=optimizer)
        else:
            model.compile(loss='mean_squared_error', optimizer=optimizer)
        print('learning rate: ', args.lr)

        # %% Create data genener
        training_generator = DataGenerator(path_of_video_tr, nframe_per_video, (img_rows, img_cols),
                                           batch_size=args.batch_size, frame_depth=args.frame_depth,
                                           temporal=args.temporal, respiration=args.respiration)
        validation_generator = DataGenerator(path_of_video_test, nframe_per_video, (img_rows, img_cols),
                                             batch_size=args.batch_size, frame_depth=args.frame_depth,
                                             temporal=args.temporal, respiration=args.respiration)
        # %%  Checkpoint Folders
        checkpoint_folder = str(os.path.join(args.save_dir, args.exp_name))
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        cv_split_path = str(os.path.join(checkpoint_folder, "cv_" + str(cv_split)))

        # %% Callbacks
        if args.save_all == 1:
            save_best_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=cv_split_path + "_epoch{epoch:02d}_model.hdf5",
                save_best_only=False, verbose=1)
        else:
            save_best_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cv_split_path + "_last_model.hdf5",
                                                                    save_best_only=False, verbose=1)
        csv_logger = tf.keras.callbacks.CSVLogger(filename=cv_split_path + '_train_loss_log.csv')
        hb_callback = HeartBeat(training_generator, validation_generator, args, str(cv_split), checkpoint_folder)

        # %% Model Training and Saving Results
        history = model.fit(x=training_generator, validation_data=validation_generator, epochs=args.nb_epoch, verbose=1,
                            shuffle=False, callbacks=[csv_logger, save_best_callback, hb_callback], validation_freq=4)

        val_loss_history = history.history['val_loss']
        val_loss = np.array(val_loss_history)
        np.savetxt((cv_split_path + '_val_loss_log.csv'), val_loss, delimiter=",")

        score = model.evaluate_generator(generator=validation_generator, verbose=1)

        print('****************************************')
        if args.temporal == 'MTTS_CAN' or args.temporal == 'MT_Hybrid_CAN' or args.temporal == 'MT_CAN_3D' \
                or args.temporal == 'MT_CAN':
            print('Average Test Score: ', score[0])
            print('PPG Test Score: ', score[1])
            print('Respiration Test Score: ', score[2])
        else:
            print('Test score:', score)
        print('****************************************')
        print('Start saving predicitions from the last epoch')

        training_generator = DataGenerator(path_of_video_tr, nframe_per_video, (img_rows, img_cols),
                                           batch_size=args.batch_size, frame_depth=args.frame_depth,
                                           temporal=args.temporal, respiration=args.respiration, shuffle=False)

        validation_generator = DataGenerator(path_of_video_test, nframe_per_video, (img_rows, img_cols),
                                             batch_size=args.batch_size, frame_depth=args.frame_depth,
                                             temporal=args.temporal, respiration=args.respiration, shuffle=False)

        yptrain = model.predict(training_generator, verbose=1)
        scipy.io.savemat(checkpoint_folder + '/yptrain_best_' + '_cv' + str(cv_split) + '.mat',
                         mdict={'yptrain': yptrain})
        yptest = model.predict(validation_generator, verbose=1)
        scipy.io.savemat(checkpoint_folder + '/yptest_best_' + '_cv' + str(cv_split) + '.mat',
                         mdict={'yptest': yptest})

        print('Finish saving the results from the last epoch')


# %% Training

print('Using Split ', str(args.cv_split))
subTrain, subTest = split_subj(args.data_dir, args.cv_split, subNum)
train(args, subTrain, subTest, args.cv_split)
