import cv2
import numpy as np
import time
import datetime
from config import conf
import os

# get camera feed and return image
def capture(cam_number = 0):
    return cv2.imread(conf.sample_folder + 'slika4.jpg')
    cap = cv2.VideoCapture(cam_number)
    ret, img = cap.read()
    cv2.imshow("input", img)
    cv2.VideoCapture(0).release()
    if not ret:
        exit(0)
    return img


def preprocess_image(img):
    #histogram equalization
    imgR = img[:,:,0]
    imgG = img[:,:,1]
    imgB = img[:,:,2]
    imR = cv2.equalizeHist(imgR)
    imG = cv2.equalizeHist(imgG)
    imB = cv2.equalizeHist(imgB)
    im = np.dstack((imR, imG, imB))
    #unsharp masking
    gaussian_3 = cv2.GaussianBlur(im, (9, 9), 10.0)
    unsharp_image = cv2.addWeighted(im, 1.5, gaussian_3, -0.5, 0, im)
    return unsharp_image


def get_detector_params():
    # Set up the detector with default parameters
    # Setup SimpleBlobDetector parameters
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 200

    # Filter by Area
    params.filterByArea = True
    params.minArea = 0.1
    params.maxArea = 20


    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.001

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.001

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.001

    return params

def get_detector(params):
    return cv2.SimpleBlobDetector_create(params)


def keypoint_to_vector(kp):
    features = []
    features.append(kp.angle)#    0
    features.append(kp.class_id)# 1
    features.append(kp.octave)#   2
    features.append(kp.pt[0])#    3
    features.append(kp.pt[1])#    4
    features.append(kp.response)# 5
    features.append(kp.size)#     6
    return features


def get_data_array(keypoints):
    data = [keypoint_to_vector(kp) for kp in keypoints]
    return np.array(data)

#get current timestamp
timestamp = time.strftime(conf.time_format, time.gmtime())

#capture single frame from camera
im = capture()

#process image for better detection
processed_image = preprocess_image(im)

#setup detector parameters and create detector
detector_params = get_detector_params()
detector = get_detector(detector_params)

# detect blobs
keypoints = detector.detect(processed_image)

# Draw detected blobs as yellow circles.
im_with_keypoints = cv2.drawKeypoints(processed_image, keypoints, np.array([]), (0, 255, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#get detected data as a matrix of feature vectors
data = get_data_array(keypoints)

#store data
if conf.save_data:
    np.save(conf.data_folder + timestamp, data)
    cv2.imwrite(conf.image_folder + timestamp + '.jpg', im)
    cv2.imwrite(conf.detected_image_folder + timestamp + '.jpg', im_with_keypoints)

#display results if wanted
if conf.display:
    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
