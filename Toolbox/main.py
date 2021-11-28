import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn
import torch.optim as optim
from model.Physnet.NegPearsonLoss import Neg_Pearson
from model.Physnet.PhysNetED_BMVC import *
from model.rPPGnet.rPPGNet import *
from torch.autograd import Variable
from dataset.data_loader import UBFC_loader
from tensorboardX import SummaryWriter
import argparse
import glob
from trainer import *
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--round_num_max', default=20, type=int)
    # parser.add_argument('--saving_steps', type=int, default=1000)
    parser.add_argument('--learn_rate', default=1e-4, type=float)
    parser.add_argument("--frame_num", default=64, type=int)
    parser.add_argument("--model", default="physnet", type=str)
    # parser.add_argument('--beta1', type=float, default=0.5,
    #                     help='beta1 for adam. default=0.5')
    parser.add_argument('--data_dir', default="/mnt/data0/UBFC/UBFC",
                        type=str, help='The path of the data directory')
    # parser.add_argument('--ckpt_dir', default='results',
    #                     type=str, help='The path of the checkpoint directory')
    # parser.add_argument('--log_dir', default='./runs', type=str)
    args = parser.parse_args()

    # The log will be saved in 'runs/exp'
    writer = SummaryWriter('runs/exp/'+args.model)
    device = torch.device("cuda:"+str(args.device)
                          if torch.cuda.is_available() else "cpu")
    print(device)
    criterion_Pearson = Neg_Pearson()   # rPPG singal
    bvp_files = glob.glob(args.data_dir+os.sep+"subject*/*.txt")
    video_files = glob.glob(args.data_dir+os.sep+"subject*/*.avi")

    bvp_train_files = bvp_files[:-10]
    bvp_valid_files = bvp_files[-10:-5]
    bvp_test_files = bvp_files[-5:]

    video_train_files = video_files[:-10]
    video_valid_files = video_files[-10:-5]
    video_test_files = video_files[-5:]

    train_data = UBFC_loader(video_train_files, bvp_train_files, "train")
    train_data.preprocessing()
    valid_data = UBFC_loader(video_valid_files, bvp_valid_files, "valid")
    valid_data.preprocessing()
    test_data = UBFC_loader(video_test_files, bvp_test_files, "test")
    test_data.preprocessing()

    train_loader = DataLoader(dataset=train_data, num_workers=2,
                              batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, num_workers=2,
                              batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, num_workers=2,
                             batch_size=args.batch_size, shuffle=True)
    if args.model == "physnet":
        model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=args.frame_num).to(device)  # [3, T, 128,128]
    elif args.model == "rppgnet":
        model = rPPGNet(frames=args.frame_num).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)
    valid_step = 0
    train_step = 0

    for round in range(args.round_num_max):
        print(f"====training:ROUND{round}====")
        train_loss = []
        model.train()
        for i, batch in enumerate(train_loader):
            if args.model == "physnet":
                loss_ecg = train_physnet(
                    model, criterion_Pearson, batch, device, train_step, writer)
            elif args.model == "rppgnet":
                loss_ecg = train_rppgnet(model, criterion_Pearson, batch,
                                         device, train_step, writer)
            train_loss.append(loss_ecg.item())
            train_step += 1
            optimizer.step()
            optimizer.zero_grad()
        train_loss = np.asarray(train_loss)
        print(np.mean(train_loss))
        print(" ====validing===")
        valid_loss = []
        model.eval()
        with torch.no_grad():
            for valid_i, valid_batch in enumerate(valid_loader):
                if args.model == "physnet":
                    loss_ecg = valid_physnet(
                        model, criterion_Pearson, valid_batch, device, valid_step, writer)
                elif args.model == "rppgnet":
                    loss_ecg = valid_rppgnet(model, criterion_Pearson,
                                             valid_batch, device, valid_step, writer)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
            valid_loss = np.asarray(valid_loss)
            print(np.mean(valid_loss))
    torch.save(model.state_dict(), "model_ubfc.pth")

print(" ====testing===")
test_step = 0
model.eval()
with torch.no_grad():
    for test_i, test_batch in enumerate(valid_loader):
        if args.model == "physnet":
            loss_ecg = test_physnet(
                model, criterion_Pearson, test_batch, device, test_step, writer)
        # elif args.model == "rppgnet":

        test_step += 1
        print(loss_ecg.item())
