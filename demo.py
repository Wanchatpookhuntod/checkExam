import numpy as np
import cv2
import imutils
import pandas as pd
import matplotlib.pyplot as plt

answers = pd.read_excel('answerExam.xlsx')
img = cv2.imread('image/20210216_095847.jpg')


def nothing(x):
    pass

# Create a black image, a window
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('MI','image',0,255,nothing)
cv2.createTrackbar('MX','image',0,255,nothing)

# create switch for ON/OFF functionality

while True:

    mi = cv2.getTrackbarPos('MI', 'image')
    mx = cv2.getTrackbarPos('MX', 'image')

    ret, thresh2 = cv2.threshold(img, mi, mx, cv2.THRESH_BINARY_INV)

    cv2.imshow('image', thresh2)

    print(f'min:{mi} | max{mx}')

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