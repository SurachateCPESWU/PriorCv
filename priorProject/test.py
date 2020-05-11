import cv2

imgOrigin = cv2.imread("2.png")
img = cv2.bitwise_not(imgOrigin)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))

dilation = cv2.dilate(thresh, rect_kernel)

contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    rect = cv2.rectangle(imgOrigin, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('image',imgOrigin)
cv2.waitKey(0)
cv2.destroyAllWindows()