import cv2

imgOrigin = cv2.imread("2.png")
imgOutput = imgOrigin
imgGray = cv2.cvtColor(imgOrigin, cv2.COLOR_BGR2GRAY)
ret, binImg = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY_INV)

mask = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))

dilation = cv2.dilate(binImg, mask)

contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    rect = cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('image',imgOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()