import numpy as np
import cv2
import sys
from scipy.misc import imresize
from keras.models import load_model
import os
import urllib.request

from darkflow.net.build import TFNet

TEST_VIDEO = 'data/testvid.MOV'
MODEL_PATH = 'cfg/yolo.cfg'
WEIGHTS_PATH = 'bin/yolo.weights'
WEIGHTS_DOWNLOAD_DIR = 'https://s3.amazonaws.com/olegpublicbucket/yolo.weights'
# Load Keras model
model = load_model('full_CNN_model.h5')


def main():
    obj_recognition = False
    lane_recognition = False

    #camera = cv2.VideoCapture(TEST_VIDEO)  # For test videos
    camera = cv2.VideoCapture(0) #For camera usage
    res, frame = camera.read()

    if frame is None:
        print('Error: TEST_VIDEO not found.')
        sys.exit(1)

    if not os.path.exists(WEIGHTS_PATH):
        print('Weights file not found. Downloading the file.')
        urllib.request.urlretrieve(WEIGHTS_DOWNLOAD_DIR, WEIGHTS_PATH)

    height, width, layers = frame.shape
    #height = int((height / 2))  # For test videos, comment out for cameras

    options = {"model": MODEL_PATH, "load": WEIGHTS_PATH, "threshold": 0.3, "gpu": 0.8, "demo":"camera"}

    tfnet = TFNet(options)

    while True:
        res, frame = camera.read()
        frame = frame[:height, :]
        if not res:
            break

        if obj_recognition:
            results = tfnet.return_predict(frame)
        else:
            results = []

        if lane_recognition:
            lane_image = road_lines(frame)
        else:
            lane_image = frame

        for result in results:
            y = result['topleft']['y']
            h = result['bottomright']['y']
            x = result['topleft']['x']
            w = result['bottomright']['x']

            top = max(0, np.floor(y + 0.5).astype(int))
            left = max(0, np.floor(x + 0.5).astype(int))
            right = min(width, np.floor(w + 0.5).astype(int))
            bottom = min(height, np.floor(h + 0.5).astype(int))

            cv2.rectangle(lane_image, (left, top), (right, bottom), (0, 255, 0), 1)
            cv2.putText(lane_image, result['label'], (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1,
                        cv2.LINE_AA)

        cv2.imshow("Press A for obj. detection, B for lane detection, Q for quit", lane_image)

        key = cv2.waitKey(110)
        if key == ord("a"):
            obj_recognition = not obj_recognition
        if key == ord("b"):
            lane_recognition = not lane_recognition
        if key == ord("q") or (key & 0xff) == 27:
            break


def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """
    original_size = image.shape
    # Get image ready for feeding into model
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)

    # # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = imresize(lane_drawn, original_size)
    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result


# Class to average lanes with
class Lanes:
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


lanes = Lanes()

if __name__ == "__main__":
    main()
