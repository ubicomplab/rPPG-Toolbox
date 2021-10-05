from POS_WANG import POS_WANG
from CHROME_DEHAAN import CHROME_DEHAAN
from ICA import ICA_POH
import time
# fake configuration
DataDirectory           = 'test_data/'
VideoFile               = DataDirectory+ 'video_example3.avi'

FS                      = 30
StartTime               = 0
Duration                = 30
ECGFile                 = None
PPGFile                 = None
PlotTF                  = False
WIDTH = 640
HEIGHT = 480


starttime = time.time()
CHROME_DEHAAN(VideoFile,FS,StartTime,Duration,ECGFile,PPGFile,PlotTF,True,WIDTH,HEIGHT)
endtime = time.time()
print("CHROME_DEHAAN duration", (endtime-starttime) * 1000 / (Duration*FS), "ms")

starttime = time.time()
POS_WANG(VideoFile,FS,StartTime,Duration,ECGFile,PPGFile,PlotTF,True,WIDTH,HEIGHT)
endtime = time.time()
print("POS_WANG duration", (endtime-starttime) * 1000 / (Duration*FS),"ms")

starttime = time.time()
ICA_POH(VideoFile,FS,StartTime,Duration,ECGFile,PPGFile,PlotTF,True,WIDTH,HEIGHT)
endtime = time.time()
print("ICA duration", (endtime-starttime) * 1000 / (Duration*FS),"ms")