import numpy as np
def fake_video(VideoFile,StartTime,Duration,FS,WIDTH,HEIGHT):
    #Standard:

    FramesNumToRead = Duration * FS#TODO,refine
    T = np.zeros((FramesNumToRead, 1))
    RGB = np.zeros((FramesNumToRead, 3))
    FN = 0
    frame = np.random.rand(WIDTH,HEIGHT,3)*255
    CurrentTime = np.random.rand()
    EndTime = StartTime + Duration

    while(FN < FramesNumToRead):
        T[FN] = CurrentTime
        sum = np.sum(np.sum(frame,axis=0),axis=0)
        # loss = RGB_mat-frame
        RGB[FN] = sum/(frame.shape[0]*frame.shape[1])
        frame = np.random.rand(WIDTH,HEIGHT,3)*255
        CurrentTime = np.random.rand()
        FN+=1
    return T,RGB
