import dlib
import numpy as np
import cv2

predictor_path = "shape_predictor_81_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def landmarks_detection(frame):
    dets = detector(frame, 0)
    for k, d in enumerate(dets):
        shape = predictor(frame, d)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        return np.transpose(landmarks)
        # for num in range(shape.num_parts):
        #     cv2.circle(frame, (shape.parts()[num].x, shape.parts()[num].y), 3, (0,255,0), -1)
    #
    # cv2.imwrite('frame_with_landmarks.bmp', frame)
