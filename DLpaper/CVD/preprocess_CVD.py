# This is a sample Python script.
import cv2
import numpy as np
from skimage import draw
from itertools import combinations
import scipy
from scipy import io as scio
import math
import skvideo.io
import matplotlib.pyplot as plt
from scipy import signal
import face_detector
import dlib
import os
import h5py
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w
#6->63
#ff2n
def combine(list,n):
    temp_list = []
    for c in combinations(list,n):
        temp_list.append(c)
    return temp_list

def get_all_combines(num):
    index = [i for i in range(0,num)]
    combine_id_list = []
    for i in range(1,num+1):
        combine_id_list.extend(combine(index,i))
    return combine_id_list


def getCombinedSignalMap(SignalMap,ROInum):
    All_idx = get_all_combines(ROInum.shape[0])#
    SignalMapOut = np.zeros((len(All_idx),SignalMap.shape[1]))
    i = 0
    for tmp_idx in All_idx:#tmp_idx= (0,1,2)[(0),(1),(2)]
        tmp_idx = np.asarray(tmp_idx)
        tmp_signal = SignalMap[tmp_idx]#6 每个区域的平均值 10->1 20->2 30->3 1*10/(10+20+30)+2*(20/(10+20+30))+3*
        tmp_ROI = ROInum[tmp_idx]# 10,20,30
        tmp_ROI = np.true_divide(tmp_ROI,np.sum(tmp_ROI))# 10/10+20+30, 20/10+20+30,30/10+20+30
        tmp_ROI = np.tile(tmp_ROI,(1,6))# (10/10+20+30, 20/10+20+30,30/10+20+30)*6
        SignalMapOut[i,:] = np.sum(np.multiply(tmp_signal,tmp_ROI),axis=0)# (1,2,3)`(1/6,2/6,3/,6) = 1*1/6+2*2/6+3*3/6
        i = i+1
    return SignalMapOut

def get_ROI_signal(frame,mask):
    [m,n,c] = frame.shape

    signal = np.zeros((1,1,c))
    for i in range(c):
        tmp = frame[:,:,i]
        signal[0,0,i] = np.mean(tmp[mask].astype(np.float32))
    return signal

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_col_coords,fill_row_coords] = True
    return mask

def parse_frame(frame,lmk,lmk_num):#w*h*3
    lmk = lmk.astype(np.int)
    R = frame[:,:,0]
    G = frame[:,:,1]
    B = frame[:,:,2]
    #TODO:matlab取int
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U =128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    V =128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    frame = np.concatenate((frame,np.expand_dims(Y,axis=2)),axis=2)
    frame = np.concatenate((frame,np.expand_dims(U,axis=2)),axis=2)
    frame = np.concatenate((frame,np.expand_dims(V,axis=2)),axis=2)
    #TODO:和RGB拼在一起

    #TODO:diff with lmks
    if lmk_num==81:
        ROI_cheek_left1 = [65,66,67,38,36,65]
        ROI_cheek_left2 = [67,68,69,70,46,40,38,67]
        ROI_cheek_right1 = [73,74,75,39,37,73]
        ROI_cheek_right2 = [75,76,77,78,47,41,39,75]
        ROI_mouth = [70,71,72,64,80,79,78,47,59,55,58,46,70]
        ROI_forehead = [18,22,20,24,19,26,30,28,32,27]

        forehead = lmk[:,ROI_forehead]
        eye_distance = np.linalg.norm(lmk[:,0]-lmk[:,9])
        tmp = (np.mean(lmk[:,18:26],axis=1)+np.mean(lmk[:,26:34],axis=1))/2 - (np.mean(lmk[:,0:9],axis=1)+ np.mean(lmk[:,9:18],axis=1))/2
        tmp = eye_distance/np.linalg.norm(tmp)*0.6*np.transpose(tmp)
        ROI_forehead = np.column_stack((forehead,np.expand_dims(forehead[:,-1]+tmp,axis=1),(np.expand_dims(forehead[:,0]+tmp,axis=1)),np.expand_dims(forehead[:,0],axis=1)))
    elif lmk_num==68:
        ROI_cheek_left1 = [0, 1, 2, 31, 41, 0]
        ROI_cheek_left2 = [2, 3, 4, 5, 48, 31, 2]
        ROI_cheek_right1 = [16, 15, 14, 35, 46, 16]
        ROI_cheek_right2 = [14, 13, 12, 11, 54, 35, 14]
        ROI_mouth = [5,6,7,8,9,10,11,54,55,56,57,58,59,48,5]
        #TODO:ROI_forehead = [17:21 22:26] + 1;?
        ROI_forehead = [17,18,19,20,21,22,]

        forehead = lmk[:, ROI_forehead]
        eye_distance = np.linalg.norm(lmk[:, 0] - lmk[:, 9])
        tmp = (np.mean(lmk[:, 18:25]) + np.mean(np.transpose(lmk[:, 26:33]))) / 2 - (
                    np.mean(np.transpose(lmk[:, 0:8])) + np.mean(np.transpose(lmk[:, 9:17]))) / 2
        tmp = eye_distance / np.linalg.norm(tmp) * 0.6 * np.transpose(tmp)
        ROI_forehead = [forehead, forehead[:, -1] + tmp, forehead[:, 0] + tmp, forehead[:, 0]]
    #TODO:这里 xy 反了，但我觉得自己没搞错
    mask_ROI_cheek_left1 = poly2mask(lmk[0,ROI_cheek_left1],lmk[1,ROI_cheek_left1],(frame.shape[0],frame.shape[1]))
    mask_ROI_cheek_left2 = poly2mask(lmk[0,ROI_cheek_left2],lmk[1,ROI_cheek_left2],(frame.shape[0],frame.shape[1]))
    mask_ROI_cheek_right1 = poly2mask(lmk[0,ROI_cheek_right1],lmk[1,ROI_cheek_right2],(frame.shape[0],frame.shape[1]))
    mask_ROI_cheek_right2 = poly2mask(lmk[0,ROI_cheek_right2],lmk[1,ROI_cheek_right2],(frame.shape[0],frame.shape[1]))
    mask_ROI_mouth = poly2mask(lmk[0,ROI_mouth],lmk[1,ROI_mouth],(frame.shape[0],frame.shape[1]))
    mask_ROI_forehead = poly2mask(ROI_forehead[0,:],ROI_forehead[1,:],(frame.shape[0],frame.shape[1]))

    Signal_tmp = np.zeros((6,6))
    ROI_num = np.zeros((6,1))

    Signal_tmp[0,:] = get_ROI_signal(frame,mask_ROI_cheek_left1)
    Signal_tmp[1,:] = get_ROI_signal(frame,mask_ROI_cheek_left2)
    Signal_tmp[2,:] = get_ROI_signal(frame,mask_ROI_cheek_right1)
    Signal_tmp[3,:] = get_ROI_signal(frame,mask_ROI_cheek_right2)
    Signal_tmp[4,:] = get_ROI_signal(frame,mask_ROI_mouth)
    Signal_tmp[5,:] = get_ROI_signal(frame,mask_ROI_forehead)

    ROI_num[0] = np.where(mask_ROI_cheek_left1 == True)[0].shape[0]
    ROI_num[1] = np.where(mask_ROI_cheek_left2 == True)[0].shape[0]
    ROI_num[2] = np.where(mask_ROI_cheek_right1 == True)[0].shape[0]
    ROI_num[3] = np.where(mask_ROI_cheek_right2 == True)[0].shape[0]
    ROI_num[4] = np.where(mask_ROI_mouth == True)[0].shape[0]
    ROI_num[5] = np.where(mask_ROI_forehead == True)[0].shape[0]

    return getCombinedSignalMap(Signal_tmp,ROI_num)#w*h*3,6->6,63

def parse_video(VideoFile,StartTime,Duration):
    '''
    :return: 63*numFrames*6
    '''
    # Standard:

    landmarks = scio.loadmat("landmarks.mat")["landmarks"]
    VidObj = cv2.VideoCapture(VideoFile)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, StartTime * 1000)
    FrameRate = VidObj.get(cv2.CAP_PROP_FPS)
    FramesNumToRead = math.ceil(Duration * FrameRate) + 1  # TODO,refine
    T = np.zeros((FramesNumToRead, 1))
    RGB = np.zeros((63,FramesNumToRead,6))
    FN = 0
    success, frame = VidObj.read()
    CurrentTime = VidObj.get(cv2.CAP_PROP_POS_MSEC)
    EndTime = StartTime + Duration

    while (success and (CurrentTime <= (EndTime * 1000))):
        T[FN] = CurrentTime

        #how to preprocess the frame

        landmarks = face_detector.landmarks_detection(frame)
        frame = cv2.cvtColor(np.array(frame).astype('float32'), cv2.COLOR_BGR2RGB)

        RGB[:,FN,:] = parse_frame(frame,landmarks,81)
        success, frame = VidObj.read()
        CurrentTime = VidObj.get(cv2.CAP_PROP_POS_MSEC)
        FN += 1
    # TODO:Skin segement TF
    T = T[:FN]
    RGB = RGB[:,:FN,:]
    # T =scio.loadmat("T.mat")["T"]
    # RGB = scio.loadmat("RGB_pos.mat")["RGB"]
    return T, RGB
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def read_time(path):
    time = []
    with open(path,"r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            time.append(int(line))
    return np.asarray(time)

def read_gt(path):
    gt = []
    with open(path,"r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            if line == "HR":
                continue
            gt.append(int(line))
    return np.asarray(gt)

def green_channel(video,bvp):
    sample_image = video[0]
    r = cv2.selectROI("face", sample_image)  # Press enter after selecting box
    print('Coordiantes: ', r)
    imCrop = sample_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)  # Press enter again to close both windows
    cv2.destroyWindow("face")
    cv2.destroyWindow("Image")

    ## Region of interest
    print('Start processing ROI')
    length = video.shape[0]
    roi_data = []
    for i in range(length):
        im = video[i]
        imCrop = im[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2]), :]
        roi_data.append(imCrop)
    roi_data = np.array(roi_data)
    print('roi shape', roi_data.shape)

    green_channel = roi_data[:, :, :, 1]
    green_channel_mean = green_channel.mean(axis=(1, 2))
    fs = 30
    b,a = signal.butter(1,[1.5/fs,5/fs],'bandpass')
    green_channel_mean = signal.filtfilt(b,a,green_channel_mean)
    N = next_power_of_2(green_channel_mean.shape[0])
    f, pxx = scipy.signal.periodogram(green_channel_mean, fs=fs, nfft=N, detrend=False)
    fmask = np.argwhere((f >= 0.75) & (f <= 2.5))
    f, pxx = np.take(f, fmask), np.take(pxx, fmask)
    plt.subplot(411)
    plt.plot(bvp)
    plt.title('BVP')
    plt.subplot(412)
    plt.plot(green_channel_mean)
    plt.title('Green Channel')
    plt.subplot(413)
    plt.plot(f, np.reshape(pxx, [-1, 1]))
    plt.title('FFT')
    plt.subplot(414)
    # plt.plot(filted_signal)
    # plt.title('Filted')
    plt.show()



def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()




def read_wave(path):
    wave = []
    with open(path,"r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            if line == "Wave":
                continue
            wave.append(int(line))
    return np.asarray(wave)

def synchronize(frames,bvp,gt,fps,clip_length):
    img_num = frames.shape[0]
    # bpm_p = []
    # bvp_p = []
    # frames_p = []
    clip_num = math.floor((img_num - clip_length) / fps * 2)
    for i in range(clip_num):
        begin_idx = math.floor(0.5 * fps * i)
        if ((begin_idx + clip_length) >= img_num):
            continue
        if math.floor(begin_idx / fps) >= gt.shape[0]:
            continue
        final_signal = frames[begin_idx:(begin_idx + clip_length),:,:,:]
        judge = np.mean(final_signal, axis = (1,2))
        if len(np.where(judge[:, 1] == 0)) <= 0:
            continue

        bvp_begin = math.floor(begin_idx / img_num * bvp.shape[0])
        bvp_len = round(clip_length / img_num * bvp.shape[0])
        if ((bvp_begin + bvp_len) > bvp.shape[0]):
            continue
        bvp_n = bvp[bvp_begin:(bvp_begin + bvp_len)]
        x = np.asarray([j for j in range(bvp_n.shape[0])])
        xx = np.asarray([j for j in range(clip_length)])
        xx = (xx * bvp_n.shape[0]) / clip_length
        # bvp = interp1(x, bvp, xx);
        bvp_n = np.interp(xx, x, bvp_n)
        frames_n = final_signal
        green_channel(frames_n,bvp_n)

        # gt_p.append(gt_temp)
        # fps_p.append(fps)
        # bpm_p.append(bpm_n)
        # bvp_p.append(bvp_n)
        #
        # final_signal_n = np.array(final_signal)
        # for idx in range(final_signal.shape[0]):
        #     for c in range(channel_num):
        #         temp = final_signal[idx, :, c]
        #         temp = moving_average(temp, 3)
        #         final_signal_n[idx, :, c] = (temp - np.min(temp)) / (np.max(temp) - np.min(temp)) * 255
        #
        # img_rgb.append(final_signal_n[:, :, 0:3])
        # img_yuv.append(final_signal_n[:, :, 3:6])


def save_MSTmaps(res,bvp,gt,fps,clip_length):
    img_num = res.shape[1]
    channel_num = res.shape[2]
    gt_p = []
    fps_p = []
    bpm_p = []
    bvp_p = []
    img_rgb = []
    img_yuv = []
    clip_num = math.floor((img_num-clip_length)/fps*2)
    for i in range(clip_num):
        begin_idx = math.floor(0.5*fps*i)
        if((begin_idx + clip_length) >= img_num):
            continue
        if math.floor(begin_idx/fps) >= gt.shape[0]:
            continue
        gt_temp = np.mean(gt[max(0,math.floor(begin_idx/fps)):min(gt.shape[0],math.floor((begin_idx+clip_length)/fps))])
        final_signal = res[:,begin_idx:(begin_idx+clip_length),:]
        judge  = np.mean(final_signal,0)
        if len(np.where(judge[:,1] == 0)) <= 0:
            continue

        bpm_n = gt_temp*clip_length/fps/60
        bvp_begin = math.floor(begin_idx/img_num*bvp.shape[0])
        bvp_len = round(clip_length/img_num*bvp.shape[0])
        if((bvp_begin+bvp_len) > bvp.shape[0]):
            continue
        bvp_n = bvp[bvp_begin:(bvp_begin+bvp_len)]
        x = np.asarray([ j for j in range(bvp_n.shape[0])])
        xx = np.asarray([ j for j in range(clip_length)])
        xx = (xx*bvp_n.shape[0])/clip_length
        # bvp = interp1(x, bvp, xx);
        bvp_n = np.interp(xx,x,bvp_n)

        gt_p.append(gt_temp)
        fps_p.append(fps)
        bpm_p.append(bpm_n)
        bvp_p.append(bvp_n)

        final_signal_n = np.array(final_signal)
        for idx in range(final_signal.shape[0]):
            for c in range(channel_num):
                temp = final_signal[idx,:,c]
                temp = moving_average(temp,3)
                final_signal_n[idx,:,c] = (temp-np.min(temp))/(np.max(temp)-np.min(temp))*255

        img_rgb.append(final_signal_n[:,:,0:3])
        img_yuv.append(final_signal_n[:,:,3:6])
    return gt_p,fps_p,bpm_p,bvp_p,img_rgb,img_yuv


# Press the green button in the gutter to run the script.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#这部分和之前的保持一致
video_path = "./data/video.avi"
landmark_num = 81
landmark_dir = "./data/face_landmarks_81p"
gt_file = "./data_example/gt_HR.csv"
time_file = "./data_example/time.txt"
bvp_file = "./data_example/wave.csv"
time = read_time(time_file)
bvp = read_wave(bvp_file)
gt = read_gt(gt_file)
fps = len(time)/(time[len(time)-1]-time[0])*1000
clip_length = 300;

#TODO:处理输入视频得到fps和time
# fps = ?

DataDirectory           = 'data_example\\'
VideoFile               = DataDirectory+'video.avi'#TODO:deal with files not found error
FS                      = 120
StartTime               = 0
Duration                = 60
ECGFile                 = DataDirectory+ 'ECGData.mat'
PPGFile                 = DataDirectory+ 'PPGData.mat'
PlotTF                  = False

# test green channel
# input: frame_n, T*W*H*3, BVP_n, T
# for whole VIPL

gt_all = []
fps_all = []
bpm_all = []
bvp_all = []
img_rgb_all = []
img_yuv_all = []
res = parse_video(VideoFile,StartTime,Duration)
# videodata = skvideo.io.vread(VideoFile)
# synchronize(videodata,bvp,gt,fps,clip_length)
gt_p,fps_p,bpm_p,bvp_p,img_rgb,img_yuv = save_MSTmaps(res[1],bvp,gt,fps,clip_length)
#
# gt_all.extend(gt_p)
# fps_all.extend(fps_p)
# bpm_all.extend(bpm_p)
# bvp_all.extend(bvp_p)
# img_rgb_all.extend(img_rgb)
# img_yuv_all.extend(img_yuv)
#
#
# file_name = "Demo.h5"
# if not os.path.exists(file_name):
# 	f = h5py.File(file_name, 'w')
# else:
#     f = h5py.File(file_name, 'w')
# f.create_dataset('gt', data=np.asarray(gt_all))
# f.create_dataset('fps', data=np.asarray(fps_all))
# f.create_dataset('bpm',data=np.asarray(bpm_all))
# f.create_dataset('bvp',data = np.asarray(bvp_all))
# f.create_dataset('img_rgb',data = np.asarray(img_rgb_all))
# f.create_dataset('img_yuv',data = np.asarray(img_yuv_all))
# f.close()



