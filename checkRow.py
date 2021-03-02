#%%

import numpy as np
import cv2
import imutils
import pandas as pd
import matplotlib.pyplot as plt
import math
#%%

answers = pd.read_excel('answerExam.xlsx')
im = cv2.imread('image/20210216_095831.jpg')
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

def rhoTheta(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    return x1, y1, x2,y2

def calculatingDegree(im):
    global pDown1New, pDown2New, pUp1New, pUp2New
    answersSheet = warped(im)
    height = 20
    width = 40
    dd = []

    for num in range(25):
        biasRow = int(5.27 * num)
        biasCol = int(num * (5/num) if num != 0 else 0)
        row = 180 + (num*height) + biasRow

        d = []
        for ans in range(100, 221, 40):
            col = ans + biasCol
            blank = answersSheet[row:row+height, col:col+width]

            blankGray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
            blankEdges = cv2.Canny(blankGray,50,220)
            lines = cv2.HoughLines(blankEdges,2,np.pi/180,10)

            if lines is not None:
                dx = []
                ux = []
                for line in lines:
                    for rho, theta in line:
                        x1, y1, x2, y2 = rhoTheta(rho, theta)

                        if (x1 < 0 and y1 < 0) and (x2 > 0 and y2 > 0):
                            slantDown = round(90-math.degrees(math.atan2(x2, y2)))

                            if 10 < slantDown < 75:
                                pDown1, pDown2 = (x1, y1), (x2, y2)
                                dx.append([pDown1, pDown2])

                        elif (x1 < 0 < y1) and (x2 > 0 > y2):
                            slantUP = round(math.degrees(math.atan2(x1, y1))+90)

                            if 10 < slantUP < 75:
                                pUp1, pUp2 = (x1, y1), (x2, y2)
                                ux.append([pUp1, pUp2])

                    if len(sorted(dx)) > 0:
                        pDown1New, pDown2New = sorted(dx[0])

                    if len(sorted(dx)) > 0:
                        pUp1New, pUp2New = sorted(dx[0])
            else:
                pDown1New, pDown2New = ([],[]),([],[])
                pUp1New, pUp2New = ([],[]),([],[])

            check = 0 if np.array(pDown1New).size == 0 and \
                         np.array(pDown2New).size == 0 and \
                         np.array(pUp1New).size == 0 and \
                         np.array(pUp2New).size == 0 else 1

            d.append(check)
        dd.append(d)

    return np.array(dd, dtype=int)

if __name__ == '__main__':

    preAnsInSheet = calculatingDegree(im)

    ansDict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'nan': 'nan'}

    scoreList = []
    ansInSheetList = []

    for j, i in enumerate(answers['Answers']):
        ans = ansDict[i]
        score = 1 if preAnsInSheet[j, ans] == 1 else 0
        scoreList.append(score)

    for ansInSheet in preAnsInSheet:
        if ansInSheet[0] == 1:
            ansChar = 'a'
        elif ansInSheet[1] == 1:
            ansChar = 'b'
        elif ansInSheet[2] == 1:
            ansChar = 'c'
        elif ansInSheet[3] == 1:
            ansChar = 'd'
        else:
            ansChar = 'nan'

        ansInSheetList.append(ansChar)

    allSheet = pd.DataFrame({"Ans": ansInSheetList,
                             "T-Ans": answers['Answers'],
                             "Score": scoreList})

    print('='*20)
    print(allSheet)
    print('='*20)
    print(f'Total Score: {sum(scoreList)}')
