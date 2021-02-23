import numpy as np
import cv2
import imutils
import pandas as pd
import matplotlib.pyplot as plt
import math

answers = pd.read_excel('answerExam.xlsx')
im = cv2.imread('image/5831.jpg')



def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

ratio = im.shape[0] / 800.0
orig = im.copy()
im = imutils.resize(im, height = 800)

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 75, 200)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:8]




for index, c in enumerate(cnts):
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)

    if len(approx) == 4 and index == 0:
        cv2.drawContours(im, [approx], -1, (0, 255, 0), 2)
        warped = four_point_transform(orig, approx.reshape(4, 2) * ratio)

        answersSheet = cv2.resize(warped, (595, 842))

        resultGray = cv2.cvtColor(answersSheet, cv2.COLOR_BGR2GRAY)
        resultBlur = cv2.blur(resultGray, (3, 3), 5)
        _, resultThres = cv2.threshold(resultBlur, 110, 225, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        im_predict = cv2.dilate(resultThres, kernel, iterations=1)
x = []
xx = []


# for row in range(1,26):
#     space = 180
#     biasRow = int(5.27 * row)
#     Y =
#     W = 175 + (j * 25)
#
#     x = []
#     form = []
#     for i in range(110, 231, 40):
#         crop = im_predict[Y:W, i:i+30]
#         contours = cv2.findContours(crop,
#                                     cv2.RETR_EXTERNAL,
#                                     cv2.CHAIN_APPROX_SIMPLE)[0]
#
#         x.append(len(contours))
#
#     xx.append(x)
# xx = np.array(xx)


# print(xx)

chois = {"A": 100, "B": 140, "C": 180, "D": 220}

ko = 25
ch = "B"
x = chois[ch]
row = ko-1

height = 20
width = 40

biasRow = int(5.27 * row)
biasCol = int(row * (5/ row ) if row != 0 else 0)
y = 180 + (row*height)

point = answersSheet[y+biasRow:y+biasRow+height,
                        x+biasCol:x+biasCol+width]

pointGray = cv2.cvtColor(point, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(pointGray,50,220)
lines = cv2.HoughLines(edges,1,np.pi/180,2)

upStatus = 0
downStatus = 0
if lines is not None:
    for line in lines[0:10]:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            # print(x1,y1,x2,y2)

            if (x1 < 0 and y1 < 0) and (x2 > 0 and y2 > 0):
                downStatus += 1
                if downStatus < 2:
                    p1, p2 = (x1, y1), (x2, y2)
                    slantDown = 90-math.degrees(math.atan2(x2, y2))
            elif (x1 < 0 < y1) and (x2 > 0 > y2):
                upStatus += 1
                if upStatus < 2:
                    p1, p2 = (x1, y1), (x2, y2)
                    slantUp = math.degrees(math.atan2(x1, y1))+90
            else:
                p1, p2 = None, None

            # print(slantUp, slantDown)
            cv2.line(point,p1, p2,(0,0,255),2)

    print(slantDown)
else:
    pass

plt.imshow(point[:,:,::-1]),plt.show()

# p = np.array([p.ravel() for p in corners])
# print(p.shape[0])

# for i in p:
#     print(sum(i))

# cv2.imshow("out",point)
# cv2.waitKey()