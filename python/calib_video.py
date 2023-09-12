#! /usr/bin/env python3

import cv2 as cv

NUM_HORIZON_CORNER  = 10
NUM_VERTICAL_CORNER = 7
NUM_SQUARE_LEN      = 25
FPS = 30



if __name__ == '__main__':

    cap = cv.VideoCapture(0)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT , 1080)
