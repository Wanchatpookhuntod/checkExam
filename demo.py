import numpy as np
import cv2
import imutils
import pandas as pd
import matplotlib.pyplot as plt

answers = pd.read_excel('answerExam.xlsx')
img = cv2.imread('t_image1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def nothing(x):
    pass

# Create a black image, a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('min','image',100,300,nothing)
cv2.createTrackbar('max','image',0,100,nothing)

# create switch for ON/OFF functionality

while True:

    mi = cv2.getTrackbarPos('min', 'image')
    mx = cv2.getTrackbarPos('max', 'image')
    #
    # ret, thresh2 = cv2.threshold(img, mi, mx, cv2.THRESH_BINARY_INV)
    # equalized = cv2.equalizeHist(gray)
    # alpha = 3 # Contrast control (1.0-3.0)
    # beta = 0  # Brightness control (0-100)

    # adjusted = cv2.convertScaleAbs(gray, alpha=1.23, beta=33)
    _, resultThresh = cv2.threshold(gray, mi, mx, cv2.THRESH_BINARY_INV)
    cv2.imshow('image', resultThresh)

    print(f'alpha:{mi} | beta{mx}')

    if cv2.waitKey(1) == 27:
        break


    # # get current positions of four trackbars
    # g = cv2.getTrackbarPos('G','image')
    # b = cv2.getTrackbarPos('B','image')
    # s = cv2.getTrackbarPos(switch,'image')
	#
    # if s == 0:
    #     img[:] = 0
    # else:
    #     img[:] = [b,g,r]

cv2.destroyAllWindows()