import cv2
import os
import numpy as np

file_name = os.path.abspath('200dpi/200dpi_3.jpg')
img = cv2.imread(file_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,19,10)


cv2.imshow('origin',th3)
cv2.waitKey(0)
cv2.destroyAllWindows()