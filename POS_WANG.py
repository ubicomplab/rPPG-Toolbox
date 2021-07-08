import cv2
import numpy as np
import math
from scipy import signal
from scipy import sparse
from scipy import linalg
from scipy import io as scio
from skimage.util import img_as_float
from sklearn.metrics import mean_squared_error
import utils

def POS_WANG(VideoFile,FS,StartTime,Duration,ECGFile,PPGFile,PlotTF):
    SkinSegmentTF = False
    LPF = 0.7
    HPF = 2.5

    WinSec = 1.6

    #
    if (PlotTF):
        PlotPRPSD = True
        PlotSNR = True
    else:
        PlotPRPSD = False
        PlotSNR = False
    T, RGB = process_video(VideoFile)
    FN = T.shape[0]

    #POS begin
    useFGTransform = False
    if useFGTransform:
        RGBBase = np.mean(RGB,axis=0)
        #TODO:bsxfun
        RGBNorm = np.true_divide(RGB,RGBBase)-1
        FF = np.fft.fft(RGBNorm)
        F = np.linspace(0,RGBNorm.shape[0],RGBNorm.shape[0],endpoint=False)*FS/RGBNorm.shape[0]
        H = np.matmul(FF,np.array([-1/math.sqrt(6),2/math.sqrt(6),-1/math.sqrt(6)]))
        W = np.true_divide(np.multiply(H,np.conj(H)),np.sum(np.multiply(FF,np.conj(FF)),axis=1))
        FMask = (F >= LPF) & (F <= HPF)
        FMask = FMask + np.fliplr(FMask);
        W  = np.multiply(W,FMask.T)#TODO:FMASK.HorT
        FF = np.multiply(FF,np.tile(W,(1,3)))
        RGBNorm = np.real(np.fft.ifft(FF))
        #TODO:bsxfun
        RGBNorm = np.multiply(RGB+1,RGBBase)
        RGB = RGBNorm
    #lines and comments
    N = RGB.shape[0]
    H = np.zeros((1,N))
    l = math.ceil(WinSec*FS)
    #TODO:why len(l)
    C = np.zeros((1,3))


    for n in range(N):

        m = n-l
        if(m>=0):
            Cn = np.true_divide(RGB[m:n,:],np.mean(RGB[m:n,:],axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(np.array([[0,1,-1],[-2,1,1]]),Cn)
            h = S[0,:]+(np.std(S[0,:])/np.std(S[1,:]))*S[1,:]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0,temp] = h[0,temp]-mean_h
            H[0,m:n] = H[0,m:n]+(h[0])

    BVP = H
    T = T[:BVP.shape[0]]
    BVP_mat = scio.loadmat("BVP_pos.mat")["BVP"]
    print(np.sqrt(mean_squared_error(BVP_mat,BVP)))
    PR = utils.prpsd(BVP[0],FS,40,240,PlotPRPSD)

    HR_ECG = utils.parse_ECG(ECGFile,StartTime,Duration)
    PR_PPG = utils.parse_PPG(PPGFile,StartTime,Duration)
    SNR = utils.bvpsnr(BVP[0], FS, HR_ECG, PlotSNR)
    #TODO:plot
    return BVP,PR,HR_ECG,PR_PPG,SNR



def process_video(VideoFile):
    #Standard:
    VidObj = cv2.VideoCapture(VideoFile)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, StartTime * 1000)
    FrameRate = VidObj.get(cv2.CAP_PROP_FPS)
    FramesNumToRead = math.ceil(Duration * FrameRate)

    T = np.zeros((FramesNumToRead, 1))
    RGB = np.zeros((FramesNumToRead, 3))
    FN = 0
    success, frame = VidObj.read()
    CurrentTime = VidObj.get(cv2.CAP_PROP_POS_MSEC)
    EndTime = StartTime + Duration

    while(success and ( CurrentTime <= (EndTime*1000) )):
        T[FN] = CurrentTime
        #TODO: if different region
        frame = cv2.cvtColor(np.array(frame).astype('float32'), cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        sum = np.sum(np.sum(frame,axis=0),axis=0)
        RGB_mat = scio.loadmat("RGB_initial.mat")["VidROI"]
        # loss = RGB_mat-frame
        RGB[FN] = sum/(frame.shape[0]*frame.shape[1])
        success, frame = VidObj.read()
        CurrentTime = VidObj.get(cv2.CAP_PROP_POS_MSEC)
        FN+=1
    # TODO:Skin segement TF

    # T =scio.loadmat("T.mat")["T"]
    # RGB = scio.loadmat("RGB_pos.mat")["RGB"]
    return T,RGB

DataDirectory           = 'test_data\\'
VideoFile               = DataDirectory+ 'video_example.mp4'
FS                      = 30
StartTime               = 0
Duration                = 60
ECGFile                 = DataDirectory+ 'ECGData.mat'
PPGFile                 = DataDirectory+ 'PPGData.mat'
PlotTF                  = False

POS_WANG(VideoFile,FS,StartTime,Duration,ECGFile,PPGFile,PlotTF)
