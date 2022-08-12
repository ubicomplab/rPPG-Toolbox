import argparse
import cv2
import math
import numpy as np
from signal_methods.ICA import *
from signal_methods.POS_WANG import *
from signal_methods.CHROME_DEHAAN import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str,default="pos",choices=["ica","pos","chrome"])
    parser.add_argument('--video_file', type=str,required=True)
    parser.add_argument("--bvp_file",type=str,required=True)
    parser.add_argument("--ppg_file",type=str,required=True)
    parser.add_argument("--plotTF")
    args = parser.parse_args()
    if(args.method == "pos"):
        POS_WANG(args.video_file,args.bvp_file,args.ppg_file,False)
    elif(args.method == "chrome"):
        CHROME_DEHAAN(args.video_file,args.bvp_file,args.ppg_file,False)
    elif(args.method == "ica"):
        ICA_POH(args.video_file,args.bvp_file,args.ppg_file,True)
