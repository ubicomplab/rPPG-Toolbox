from scipy import io as scio
import cv2
import math
import numpy as np

def facial_detection(frame):
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_zone = detector.detectMultiScale(frame)
    if(len(face_zone) < 1):
        print("ERROR:No Face Detected")
    if(len(face_zone) >= 2):
        print("WARN:More than one faces are detected(Only cropping one face)")
    x,y,w,h = face_zone[0]
    return x,y,w,h

def crop_face_down_sample(frame,x,y,w,h):
    frame = frame[y:y + h, x:x + w]
    frame = cv2.resize(frame, (128, 128))
    return frame

def parse_video(VideoFile,StartTime,Duration):
    '''
    :return: 3*numframe*weigh*high
    '''
    # Standard:

    VidObj = cv2.VideoCapture(VideoFile)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, StartTime * 1000)
    FrameRate = VidObj.get(cv2.CAP_PROP_FPS)
    FramesNumToRead = math.ceil(Duration * FrameRate) + 1  # TODO,refine
    T = np.zeros((FramesNumToRead, 1))
    RGB = np.zeros((3,FramesNumToRead,128,128))
    FN = 0
    success, frame = VidObj.read()
    x,y,w,h = facial_detection(frame)

    CurrentTime = VidObj.get(cv2.CAP_PROP_POS_MSEC)
    EndTime = StartTime + Duration

    while (success and (CurrentTime <= (EndTime * 1000))):
        T[FN] = CurrentTime

        #how to preprocess the frame

        #TODO:preprocess the frames:做了调整轴顺序，为了适配rPPG的输入要求，这里可以按照自己的模型做调整
        frame = cv2.cvtColor(np.array(frame).astype('float32'), cv2.COLOR_BGR2RGB)
        frame = crop_face_down_sample(frame,x,y,w,h)
        frame = np.transpose(frame,(2,0,1))
        RGB[:,FN,:,:] = frame


        success, frame = VidObj.read()
        CurrentTime = VidObj.get(cv2.CAP_PROP_POS_MSEC)
        FN += 1

    T = T[:FN]
    RGB = RGB[:,:FN,:,:]
    # T =scio.loadmat("T.mat")["T"]
    # RGB = scio.loadmat("RGB_pos.mat")["RGB"]
    return T, RGB



DataDirectory           = 'data_example\\'
VideoFile               = DataDirectory+'video.avi'#TODO:deal with files not found error
#下面这些参数只有StartTime和Duration有意义，其他不用管
FS                      = 120
StartTime               = 0
Duration                = 60
ECGFile                 = DataDirectory+ 'ECGData.mat'
PPGFile                 = DataDirectory+ 'PPGData.mat'
PlotTF                  = False

T,RGB = parse_video(VideoFile,StartTime,Duration)
print()