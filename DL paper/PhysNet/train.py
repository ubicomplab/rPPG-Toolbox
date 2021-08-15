import numpy as np
import torch
import h5py
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch import nn
import torch.optim as optim
from NegPearsonLoss import Neg_Pearson
from PhysNetED_BMVC import *

class MyDataset(Dataset):
    def __init__(self, archive,xtag = "xsub",ytag='ysub'):
        self.archive = h5py.File(archive, 'r')
        self.data = self.archive[xtag]
        self.labels = self.archive[ytag]

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x,y

    def __len__(self):
        return len(self.labels)

criterion_Pearson = Neg_Pearson()   # rPPG singal
#paras:
train_filename= "void.h5"
test_filename = "void.h5"
batch_size = 8
round_num_max = 100
learn_rate = 1e-4

train = MyDataset(train_filename)
train_len = TODO
valid_len = TODO
train_data,valid_data = torch.utils.data.random_split(dataset= train, lengths=[train_len,valid_len])
test_data = MyDataset(test_filename)

train_loader = DataLoader(dataset=train_data,num_workers=0,batch_size=batch_size,shuffle=True)
valid_loader = DataLoader(dataset=valid_data,num_workers=0,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_data,num_workers=0,batch_size=batch_size,shuffle=True)




model = PhysNet_padding_Encoder_Decoder_MAX()
#TODO:没指定优化器
optimizer = optim.Adam(model.parameters(),lr=learn_rate)

for round in range(round_num_max):
    print("====training====")
    loss = 0
    for i,batch in enumerate(train_loader):
        rPPG, x_visual, x_visual3232, x_visual1616 = model(batch[0])
        BVP_label = batch[1]
        rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
        BVP_label = (BVP_label - torch.mean(BVP_label)) / torch.std(BVP_label)  # normalize
        loss_ecg = criterion_Pearson(rPPG, BVP_label)
        optimizer.zero_grad()
        loss_ecg.backward()
        optimizer.step()
        print("loss:", loss_ecg)

    print(" ====validing===")
    valid_loss = []
    model.eval()
    with torch.no_grad():
        for valid_i, valid_batch in enumerate(valid_loader):
            BVP_label = valid_batch[1]
            rPPG, x_visual, x_visual3232, x_visual1616 = model(valid_batch[0])
            rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
            BVP_label = (BVP_label - torch.mean(BVP_label)) / torch.std(BVP_label)  # normalize
            loss_ecg = criterion_Pearson(rPPG, BVP_label)
            valid_loss.append(float(loss_ecg))

        valid_loss = np.asarray(valid_loss)
        print(np.mean(valid_loss))
    model.train()

print(" ====testing===")
test_loss = []
model.eval()
with torch.no_grad():
    for test_i, test_batch in enumerate(test_loader):
        BVP_label = test_batch[1]
        rPPG, x_visual, x_visual3232, x_visual1616 = model(test_batch[0])
        rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
        BVP_label = (BVP_label - torch.mean(BVP_label)) / torch.std(BVP_label)  # normalize
        loss_ecg = criterion_Pearson(rPPG, BVP_label)
        test_loss.append(float(loss_ecg))

test_loss = np.asarray(test_loss)
print(np.mean(test_loss))