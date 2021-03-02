import cv2
import numpy as np

img = cv2.imread('image/cross4.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,220)

lines = cv2.HoughLines(edges,1,np.pi/180,65)
#
# print(lines)

# for rho,theta in lines[2]:
for line in lines:
    for rho, theta in line:
        print(rho, theta)
        a = np.cos(theta)
        b = np.sin(theta)

        print(a, b)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        # print(x1,y1,x2,y2)
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
#
cv2.imshow('out', img)
cv2.waitKey()


# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt
# img = cv.imread('image/cross4.jpg')
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# corners = cv.goodFeaturesToTrack(gray,25,0.01,10)
# corners = np.int0(corners)
# l = np.array([])
# for j, i in enumerate(corners):
#     x,y = i.ravel()
#     cv.circle(img,(x,y),3,255,-1)
#     cv.putText(img, f"{j}", (x,y), cv.FONT_HERSHEY_SIMPLEX, .4, 255)
# plt.imshow(img),plt.show()
# print(l)