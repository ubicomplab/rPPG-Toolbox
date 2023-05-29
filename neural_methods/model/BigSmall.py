"""BigSmall: Multitask Network for AU / Respiration / PPG

BigSmall: Efficient Multi-Task Learning
For Physiological Measurements
Girish Narayanswamy, Yujia (Nancy) Liu, Yuzhe Yang, Chengqian (Jack) Ma, 
Xin Liu, Daniel McDuff, Shwetak Patel

https://arxiv.org/abs/2303.11573
"""

import torch
import torch.nn as nn


#####################################################
############ Wrapping Time Shift Module #############
#####################################################
class WTSM(nn.Module):
    def __init__(self, n_segment=3, fold_div=3):
        super(WTSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, -1, :fold] = x[:, 0, :fold] # wrap left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, 0, fold: 2 * fold] = x[:, -1, fold: 2 * fold]  # wrap right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # no shift for final fold
        return out.view(nt, c, h, w)



#######################################################################################
##################################### BigSmall Model ##################################
#######################################################################################
class BigSmall(nn.Module):

    def __init__(self, in_channels=3, nb_filters1=32, nb_filters2=64, kernel_size=3, 
                 dropout_rate1=0.25, dropout_rate2=0.5, dropout_rate3=0.5, pool_size1=(2, 2), pool_size2=(4,4),
                 nb_dense=128, out_size_bvp=1, out_size_resp=1, out_size_au=12, n_segment=3):

        super(BigSmall, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout_rate1 = dropout_rate1
        self.dropout_rate2 = dropout_rate2
        self.dropout_rate3 = dropout_rate3
        self.pool_size1 = pool_size1
        self.pool_size2 = pool_size2
        self.nb_filters1 = nb_filters1
        self.nb_filters2 = nb_filters2
        self.nb_dense = nb_dense

        self.out_size_bvp = out_size_bvp
        self.out_size_resp = out_size_resp
        self.out_size_au = out_size_au

        self.n_segment = n_segment

        # Big Convolutional Layers
        self.big_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv5 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)
        self.big_conv6 = nn.Conv2d(self.nb_filters2, self.nb_filters2, kernel_size=self.kernel_size, padding=(1, 1), bias=True)

        # Big Avg Pooling / Dropout Layers
        self.big_avg_pooling1 = nn.AvgPool2d(self.pool_size1)
        self.big_dropout1 = nn.Dropout(self.dropout_rate1)
        self.big_avg_pooling2 = nn.AvgPool2d(self.pool_size1)
        self.big_dropout2 = nn.Dropout(self.dropout_rate2)
        self.big_avg_pooling3 = nn.AvgPool2d(self.pool_size2)
        self.big_dropout3 = nn.Dropout(self.dropout_rate3)

        # TSM layers
        self.TSM_1 = WTSM(n_segment=self.n_segment)
        self.TSM_2 = WTSM(n_segment=self.n_segment)
        self.TSM_3 = WTSM(n_segment=self.n_segment)
        self.TSM_4 = WTSM(n_segment=self.n_segment)
        
        # Small Convolutional Layers
        self.small_conv1 = nn.Conv2d(self.in_channels, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv2 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv3 = nn.Conv2d(self.nb_filters1, self.nb_filters1, kernel_size=self.kernel_size, padding=(1,1), bias=True)
        self.small_conv4 = nn.Conv2d(self.nb_filters1, self.nb_filters2, kernel_size=self.kernel_size, padding=(1,1), bias=True)

        # AU Fully Connected Layers 
        self.au_fc1 = nn.Linear(5184, self.nb_dense, bias=True)
        self.au_fc2 = nn.Linear(self.nb_dense, self.out_size_au, bias=True)

        # BVP Fully Connected Layers 
        self.bvp_fc1 = nn.Linear(5184, self.nb_dense, bias=True)
        self.bvp_fc2 = nn.Linear(self.nb_dense, self.out_size_bvp, bias=True)

        # Resp Fully Connected Layers 
        self.resp_fc1 = nn.Linear(5184, self.nb_dense, bias=True)
        self.resp_fc2 = nn.Linear(self.nb_dense, self.out_size_resp, bias=True)


    def forward(self, inputs, params=None):

        big_input = inputs[0] # big res 
        small_input = inputs[1] # small res

        # reshape Big 
        nt, c, h, w = big_input.size()
        n_batch = nt // self.n_segment
        big_input = big_input.view(n_batch, self.n_segment, c, h, w)
        big_input = torch.moveaxis(big_input, 1, 2) # color channel to idx 1, sequence channel to idx 2
        big_input = big_input[:, :, 0, :, :] # use only first frame in sequences 


        # Big Conv block 1
        b1 = nn.functional.relu(self.big_conv1(big_input))
        b2 = nn.functional.relu(self.big_conv2(b1))
        b3 = self.big_avg_pooling1(b2)
        b4 = self.big_dropout1(b3)

        # Big Conv block 2
        b5 = nn.functional.relu(self.big_conv3(b4))
        b6 = nn.functional.relu(self.big_conv4(b5))
        b7 = self.big_avg_pooling2(b6)
        b8 = self.big_dropout2(b7)

        # Big Conv block 3
        b9 = nn.functional.relu(self.big_conv5(b8))
        b10 = nn.functional.relu(self.big_conv6(b9))
        b11 = self.big_avg_pooling3(b10)
        b12 = self.big_dropout3(b11)

        # Reformat Big Shape For Concat w/ Small Branch
        b13 = torch.stack((b12, b12, b12), 2) #TODO: this is hardcoded for num_segs = 3: change this...
        b14 = torch.moveaxis(b13, 1, 2)
        bN, bD, bC, bH, bW = b14.size()
        b15 = b14.reshape(int(bN*bD), bC, bH, bW)

        # Small Conv block 1
        s1 = self.TSM_1(small_input)
        s2 = nn.functional.relu(self.small_conv1(s1))
        s3 = self.TSM_2(s2)
        s4 = nn.functional.relu(self.small_conv2(s3))

        # Small Conv block 2
        s5 = self.TSM_3(s4)
        s6 = nn.functional.relu(self.small_conv3(s5))
        s7 = self.TSM_4(s6)
        s8 = nn.functional.relu(self.small_conv4(s7))

        # Shared Layers
        concat = b15 + s8 # sum layers

        # share1 = concat.view(concat.size(0), -1) # flatten entire tensors
        share1 = concat.reshape(concat.size(0), -1)

        # AU Output Layers
        aufc1 = nn.functional.relu(self.au_fc1(share1))
        au_out = self.au_fc2(aufc1)

        # BVP Output Layers
        bvpfc1 = nn.functional.relu(self.bvp_fc1(share1))
        bvp_out = self.bvp_fc2(bvpfc1)

        # Resp Output Layers
        respfc1 = nn.functional.relu(self.resp_fc1(share1))
        resp_out = self.resp_fc2(respfc1)

        return au_out, bvp_out, resp_out


