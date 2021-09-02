# This is a sample Python script.
import cv2
import numpy as np
from skimage import draw
from itertools import combinations
from scipy import io as scio
import math
import dlib
from matplotlib import pyplot as plt
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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
    print(combine_id_list)
    return combine_id_list


def getCombinedSignalMap(SignalMap,ROInum):
    All_idx = get_all_combines(ROInum.shape[0])
    SignalMapOut = np.zeros((len(All_idx),SignalMap.shape[1]))
    i = 0
    for tmp_idx in All_idx:
        tmp_idx = np.asarray(tmp_idx)
        tmp_signal = SignalMap[tmp_idx]
        tmp_ROI = ROInum[tmp_idx]
        tmp_ROI = np.true_divide(tmp_ROI,np.sum(tmp_ROI))
        tmp_ROI = np.tile(tmp_ROI,(1,6))
        SignalMapOut[i,:] = np.sum(np.multiply(tmp_signal,tmp_ROI),axis=0)
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
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, (shape[0]+100,shape[1]+100))
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_col_coords,fill_row_coords] = True
    #visualize_mask2(mask, shape)
    return mask

def parse_frame(frame,lmk_num):
     
   # i=1
   # lmk = lmk.astype(np.int)
   
    #TODO:matlab取int
    R = frame[:,:,0]
    G = frame[:,:,1]
    B = frame[:,:,2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U =128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    V =128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    frame = np.concatenate((frame,np.expand_dims(Y,axis=2)),axis=2)
    frame = np.concatenate((frame,np.expand_dims(U,axis=2)),axis=2)
    frame = np.concatenate((frame,np.expand_dims(V,axis=2)),axis=2)

   # mask_ROI_forehead = poly2mask((a[17],a[18],a[19],a[20],a[21],a[22],a[23],a[24],a[25],a[26],a[79],a[72],a[71],a[69],a[76]),([b[17],b[18],b[19],b[20],b[21],b[22],b[23],b[24],b[25],b[26],b[79],b[72],b[71],b[69],b[76]),(frame.shape[0],frame.shape[1]))
   # mask_ROI_cheek_left1 = poly2mask(a[1],b[1],a[21],b[28],a[21],b[30],a[2],b[2],a[1],b[1](frame.shape[0],frame.shape[1]))
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

    return getCombinedSignalMap(Signal_tmp,ROI_num)

def parse_video(VideoFile,StartTime,Duration):
    '''
    :return: 63*numFrames*6
    '''
    # Standard:
    #landmarks = scio.loadmat("landmarks.mat")["landmarks"]
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
        #TODO:get the landmarks
        # landmark
        #TODO:preprocess the frames:
        frame = cv2.cvtColor(np.array(frame).astype('float32'), cv2.COLOR_BGR2RGB)
        RGB[:,FN,:] = parse_frame(frame,81)

        success, frame = VidObj.read()
        CurrentTime = VidObj.get(cv2.CAP_PROP_POS_MSEC)
        FN += 1
    # TODO:Skin segement TF
    T = T[:FN]
    RGB = RGB[:FN]
    # T =scio.loadmat("T.mat")["T"]
    # RGB = scio.loadmat("RGB_pos.mat")["RGB"]
    return T, RGB
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

#这部分和之前的保持一致
video_path = "./data/video.avi"
landmark_num = 81
landmark_dir = "./data/face_landmarks_81p"
gt_file = "./data/gt_HR.csv"
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

#frame = scio.loadmat("frame.mat")["frame"]

#res = parse_video(VideoFile,StartTime,Duration)
cap = cv2.VideoCapture("test.mp4")

predictor_path = "shape_predictor_81_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
a = np.arange(81)
b = np.arange(81)
POINTS=()
i=1

def visualize_mask(mask_ROI_forehead,path,frame_shape):
    mask_ROI_forehead=mask_ROI_forehead.astype(np.int64)
    for i in range(0,frame_shape[0]):
        for j in range(0,frame_shape[1]):
            if (mask_ROI_forehead[i][j][1]+mask_ROI_forehead[i][j][0]+mask_ROI_forehead[i][j][2])>0:
                mask_ROI_forehead[i][j][0]=0
                mask_ROI_forehead[i][j][1]
                mask_ROI_forehead[i][j][2]=0
                print(i,j)
    
    cv2.imwrite(path, mask_ROI_forehead)
def visualize_mask2(mask_ROI_forehead,frame_shape):
    mask_ROI_forehead=mask_ROI_forehead.astype(np.float32)
    for i in range(frame_shape[0]):
        for j in range(frame_shape[1]):
            if (mask_ROI_forehead[i][j][1]+mask_ROI_forehead[i][j][0]+mask_ROI_forehead[i][j][2])>0:
                mask_ROI_forehead[i][j][0]=0
                mask_ROI_forehead[i][j][1]=255
                mask_ROI_forehead[i][j][2]=0
                print(i,j)
    
    cv2.imshow("a",mask_ROI_forehead)


while(1):
    ret, frame = cap.read()
    # 取灰度
    frame=cv2.resize(frame,(525,300))
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # 人脸数rects
    #img_gray=cv2.resize(img_gray,(20,20))
    rects = detector(img_gray, 0)
    dets = detector(img_gray, 0)
    for i in range(len(rects)):
        #frame=cv2.resize(frame,(20,20))
        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 81点的坐标
            pos = (point[0, 0], point[0, 1])
            # 利用cv2.circle给每个特征点画一个圈，共81个
            cv2.circle(frame, pos, 2, color=(0, 255, 0))
            a = a.tolist()
            b = b.tolist()
            a[idx] = point[0, 0]
            b[idx] = point[0, 1]
            a = np.array(a)
            b = np.array(b)
            
            # 利用cv2.putText输出1-81
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(idx + 1), pos, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            print("index=" + str(idx+1) + " x=" + str(pos[0]) + " y=" + str(pos[1]))
            #frame=cv2.resize(frame,(1050,600))
            
            
   # frame=cv2.resize(frame,(1050,600))
    mask_ROI_forehead = poly2mask((a[17],a[18],a[19],a[20],a[21],a[22],a[23],a[24],a[25],a[26],a[79],a[72],a[71],a[69],a[76],a[17]),(b[17],b[18],b[19],b[20],b[21],b[22],b[23],b[24],b[25],b[26],b[79],b[72],b[71],b[69],b[76],b[17]),frame.shape)
    #mask_ROI_forehead = poly2mask((a[79],a[72],a[71]),(b[79],b[72],b[71]),frame.shape)
    #mask_ROI_forehead = poly2mask((a[25],a[26],a[79],a[72],a[71]),(b[24],b[25],b[26],b[79],b[72],b[71]),frame.shape)
    mask_ROI_cheek_left1 = poly2mask((a[1],a[21],a[21],a[2]),(b[1],b[28],b[30],b[2]),frame.shape)
    mask_ROI_cheek_left2 = poly2mask((a[2],a[21],a[31],a[48],a[4],a[3]),(b[2],b[30],b[31],b[48],b[4],b[3]),frame.shape)
    mask_ROI_cheek_right1 = poly2mask((a[15],a[22],a[22],a[14]),(b[15],b[28],b[30],b[14]),frame.shape)
    mask_ROI_cheek_right2 = poly2mask((a[14],a[22],a[35],a[54],a[12],a[13]),(b[14],b[30],b[35],b[54],b[12],b[13]),frame.shape)
    mask_ROI_mouth = poly2mask((a[54],a[55],a[56],a[57],a[58],a[59],a[48],a[4],a[5],a[6],a[9],a[10],a[11],a[12]),(b[54],b[55],b[56],b[57],b[58],b[59],b[48],b[4],b[5],b[6],b[9],b[10],b[11],b[12]),frame.shape)
     #区域划分
  
    cv2.imshow('frame', frame)
    visualize_mask2(mask_ROI_forehead+mask_ROI_cheek_left1+mask_ROI_cheek_left2+mask_ROI_cheek_right1+ mask_ROI_cheek_right2+mask_ROI_mouth,frame.shape)
    plt.imshow(frame)
    plt.show()
    plt.plot([a[29],a[2]],[b[29],b[2]])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("q pressed")
        break
#    mask_ROI_forehead = poly2mask((a[17],a[18],a[19],a[20],a[21],a[22],a[23],a[24],a[25],a[26],a[79],a[72],a[71],a[69],a[76]),([b[17],b[18],b[19],b[20],b[21],b[22],b[23],b[24],b[25],b[26],b[79],b[72],b[71],b[69],b[76]),(frame.shape[0],frame.shape[1]))
   # RGB[:,FN,:] = parse_frame(frame,81)

print()


