import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

imOrigin = cv2.imread("image/de-3.jpg")
im_test = imOrigin.copy()
gray = cv2.cvtColor(im_test, cv2.COLOR_BGR2GRAY)
im_predict = np.ones((gray.shape[0], gray.shape[1]), dtype=np.uint8) * 255

answers = pd.read_excel('checkExam.xlsx')

A = [5, 45]
B = [55, 95]
C = [105, 145]
D = [155, 195]

covY = lambda j : j * 50 - 45
covH = lambda j : j * 50 - 5

print(answers['answers'])

for j, choose in enumerate(answers['answers'], start=1):
    numY = covY(j)
    numH = covH(j)

    chooseIndex = ''
    if choose == "a":
        chooseIndex = A
    elif choose == "b":
        chooseIndex = B
    elif choose == "c":
        chooseIndex = C
    elif choose == "d":
        chooseIndex = D
    else:
        print(f'Numbers {j} answers not in a b c d')

    im_predict[numY: numH, chooseIndex[0]: chooseIndex[1]] = \
        gray[numY: numH, chooseIndex[0]: chooseIndex[1]]

thresh = cv2.threshold(im_predict, 170, 255, cv2.THRESH_BINARY_INV)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

for c in contours:
    # x, y, w, h = cv2.boundingRect(c)
    # cv2.rectangle(im_tes, (x, y), (x + w, y + h), (36, 255, 12), 2)
    (x, y), r = cv2.minEnclosingCircle(c)
    center = (int(x), int(y))
    radius = int(r)
    cv2.circle(imOrigin, center, int(radius / 2), (0, 0, 255), 2)

print(len(contours))
cv2.putText(imOrigin, f"score: {len(contours)}",
            (imOrigin.shape[1] - 80, 20),
            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)

plt.imshow(imOrigin[:, :, ::-1])
# plt.imshow(im_predict, 'gray')
plt.show()

# cv2.imshow("out", im_tes)
# cv2.imshow("out_t", thresh)
# cv2.waitKey()
