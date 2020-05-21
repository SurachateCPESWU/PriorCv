import cv2
import os
import numpy as np

file_name = os.path.abspath('200dpi/200dpi_3.jpg')
imgOrigin = cv2.imread(file_name)
gray = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 3, 25, 25)
cv2.imshow('out',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.adaptiveThreshold(gray, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 4)

cv2.imshow('out',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


img = cv2.medianBlur(img, 11)
img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
thresh = cv2.erode(img, None, iterations=7)
thresh = cv2.dilate(thresh, None, iterations=7)


cv2.imshow('out',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()