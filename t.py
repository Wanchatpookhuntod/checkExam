#%%

import numpy as np
import cv2
import imutils
import pandas as pd
import matplotlib.pyplot as plt
import math
#%%

answers = pd.read_excel('answerExam.xlsx')
im = cv2.imread('image/5831.jpg')
#%%

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
#%%

def four_point_transform(image, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
#%%
def warped(im):
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
            transform = four_point_transform(orig, approx.reshape(4, 2) * ratio)
            im_pedict = cv2.resize(transform, (595, 842))
            return im_pedict


def calculatingDegree(im):
    answersSheet = warped(im)
    height = 20
    width = 40
    listAns = []

    for num in range(25):
        biasRow = int(5.27 * num)
        biasCol = int(num * (5/num) if num != 0 else 0)
        row = 180 + (num*height) + biasRow
        choos = []

        for ans in range(100, 221, 40):
            col = ans + biasCol
            blank = answersSheet[row:row+height, col:col+width]

            blankGray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
            blankEdges = cv2.Canny(blankGray,50,220)
            lines = cv2.HoughLines(blankEdges,1,np.pi/180,2)

            upStatus = 0
            downStatus = 0

            if lines is not None:
                for line in lines[0:20]:
                    for rho, theta in line:
                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a*rho
                        y0 = b*rho
                        x1 = int(x0 + 1000*(-b))
                        y1 = int(y0 + 1000*(a))
                        x2 = int(x0 - 1000*(-b))
                        y2 = int(y0 - 1000*(a))

                        if (x1 < 0 and y1 < 0) and (x2 > 0 and y2 > 0):
                            downStatus += 1
                            if downStatus < 2:
                                p1, p2 = (x1, y1), (x2, y2)
                                slantDown = round(90-math.degrees(math.atan2(x2, y2)))
                        elif (x1 < 0 < y1) and (x2 > 0 > y2):
                            upStatus += 1
                            if upStatus < 2:
                                p1, p2 = (x1, y1), (x2, y2)
                                slantUp = round(math.degrees(math.atan2(x1, y1))+90)
                        else:
                            p1, p2 = None, None
                        # cv2.line(im,p1, p2,(0,0,255),2)
            else:
                slantDown = 0
                slantUp = 0

            if slantUp > 13 and slantDown > 13:
                cross = 1
            else:
                cross = 0

            choos.append(cross)
        listAns.append(choos)
    return np.array(listAns)

if __name__ == '__main__':

    preAnsInSheet = calculatingDegree(im)

    ansDict = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    inv_map = {ansDict[k]: k for k in ansDict}

# fide true answers to score
    scoreList = []
    ansInSheetList = []
    for j, i in enumerate(answers['Answers']):

        ans = ansDict[i]
        score = 1 if preAnsInSheet[j, ans] == 1 else 0
        scoreList.append(score)

        idxs = np.ravel(np.where(preAnsInSheet[j] == 1))

        for xx, idx in enumerate(idxs):
            ansInSheetList.append(inv_map[idx])
            PR

    print(len(scoreList))
    print(len(ansInSheetList))


    # allSheet = pd.DataFrame({"T-Ans":answers['Answers'],
    #                          "Ans": ansInSheetList,
    #                          "Score": scoreList})
    # print(allSheet)
    #
