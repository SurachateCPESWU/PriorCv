import cv2
import os
import numpy as np

file_name = os.path.abspath('200dpi/200dpi_3.jpg')
imgOrigin = cv2.imread(file_name)

denoise = cv2.fastNlMeansDenoisingColored(imgOrigin,None,3,21,7,21)

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
filterimg = cv2.filter2D(denoise, -1, kernel)

imgGray = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
ret, binImg = cv2.threshold(imgGray, 200, 255, cv2.THRESH_BINARY_INV)


cv2.imshow('imageOri',imgOrigin)
cv2.imshow('image',denoise)
cv2.imshow('filter2D',filterimg)
cv2.imshow('binImg',binImg)
cv2.waitKey(0)
cv2.destroyAllWindows()