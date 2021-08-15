#Transfalte from iphys toolbox

# if (SkinSegmentTF) % skin segmentation - not specified in reference
# YCBCR = rgb2ycbcr(VidROI);
# Yth = YCBCR(:,:, 1) > 80;
# CBth = (YCBCR(:,:, 2) > 77).*(YCBCR(:,:, 2) < 127);
# CRth = (YCBCR(:,:, 3) > 133).*(YCBCR(:,:, 3) < 173);
# ROISkin = VidROI. * repmat(uint8(Yth. * CBth. * CRth), [1, 1, 3]);
# RGB(FN,:) = squeeze(sum(sum(ROISkin, 1), 2). / sum(sum(logical(ROISkin), 1), 2));
# else
# RGB(FN,:) = sum(sum(VidROI, 2)). / (size(VidROI, 1) * size(VidROI, 2));
# end

#ref:https://blog.csdn.net/qq_42722197/article/details/115388593

import cv2
import numpy as np
"""
input a frame
return mask.npy
"""

def exemple():
    img = cv2.imread("save.bmp", cv2.IMREAD_COLOR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    (y, cr, cb) = cv2.split(ycrcb)
    skin2 = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    for i in  range(0, x):
        for j in  range(0, y):
            if (cr[i][j] >  133) and (cr[i][j] <  173) and (cb[i][j] >  77) and (cb[i][j] <  127):
                skin2[i][j] =  255
            else:
                skin2[i][j] =  0
    cv2.imwrite("save1.bmp", skin2)


def visualize_mask(mask,path):
    cv2.imwrite(path, mask * 255)

def frame2skinmask(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    (y, cr, cb) = cv2.split(ycrcb)
    skin2 = np.zeros(cr.shape, dtype=np.uint8)
    (x, y) = cr.shape
    for i in range(0, x):
        for j in range(0, y):
            if (cr[i][j] > 133) and (cr[i][j] < 173) and (cb[i][j] > 77) and (cb[i][j] < 127):
                skin2[i][j] = 1
            else:
                skin2[i][j] = 0
    return skin2

exemple()